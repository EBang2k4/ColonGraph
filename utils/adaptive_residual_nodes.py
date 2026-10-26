"""Function-preserving XYZ-only local controllers over an unchanged J8 backbone."""
import types
from collections import Counter

import torch
from torch import nn

from utils.adaptive_deformation_graph import AdaptiveNodeConfig
from utils.adaptive_g3_graph import AdaptiveG3NodeManager
from utils.adaptive_j8 import import_pre_split_statistics
from utils.exact_split_branchpoint import digest


class ResidualNodes(nn.Module):
    FORMAT='adaptive_residual_translation_linear4_v1'

    def __init__(self,n,device,ramp_steps=500):
        super().__init__()
        self.ramp_steps=int(ramp_steps)
        if self.ramp_steps<1:raise ValueError('Activation ramp must be positive')
        self.coefficients=nn.ParameterList()
        self.register_buffer('assignment',torch.full((n,),-1,device=device,dtype=torch.long))
        self.register_buffer('influence',torch.zeros(n,device=device))
        self.register_buffer('positions',torch.empty((0,3),device=device))
        self.register_buffer('parents',torch.empty(0,device=device,dtype=torch.long))
        self.register_buffer('births',torch.empty(0,device=device,dtype=torch.long))
        self.register_buffer('radii',torch.empty(0,device=device))
        self.register_buffer('iteration',torch.zeros((),device=device,dtype=torch.long))

    def activation(self):
        return ((self.iteration-self.births).float()/self.ramp_steps).clamp(0,1)

    def translations(self,time):
        if not len(self.coefficients):return self.positions.new_empty((0,3))
        t=torch.as_tensor(time,device=self.positions.device).reshape(-1)[0].clamp(0,1)*3
        lo=t.floor().long().clamp(max=2);fraction=t-lo
        values=torch.cat(list(self.coefficients),0)
        return ((1-fraction)*values[:,lo]+fraction*values[:,lo+1])*self.activation()[:,None]

    def correction(self,time):
        if not len(self.positions):return self.influence.new_zeros((len(self.assignment),3))
        return self.influence[:,None]*self.translations(time)[self.assignment.clamp_min(0)]

    def apply(self,xyz,time):
        if not len(self.positions):return xyz
        return xyz+self.correction(time)

    @torch.no_grad()
    def insert(self,proposals,iteration):
        if not proposals:return None
        if len(self.positions)+len(proposals)>400:raise ValueError('Residual node cap400 exceeded')
        occupied=self.assignment>=0
        for proposal in proposals:
            rows=proposal['rows'];weights=proposal['weights']
            if occupied[rows].any():raise ValueError('Insertion must not replace an existing controller')
            if len(rows)!=len(weights) or not torch.isfinite(weights).all() or (weights<=0).any() or (weights>1).any():
                raise ValueError('Invalid residual support weights')
            occupied[rows]=True
        start=len(self.positions)
        for offset,p in enumerate(proposals):
            self.assignment[p['rows']]=start+offset
            self.influence[p['rows']]=p['weights']
        for key,values in [('positions',torch.stack([p['position'] for p in proposals])),
            ('parents',self.parents.new_tensor([p['parent'] for p in proposals])),
            ('births',self.births.new_full((len(proposals),),iteration)),
            ('radii',self.radii.new_tensor([p['radius'] for p in proposals]))]:
            setattr(self,key,torch.cat((getattr(self,key),values)))
        parameter=nn.Parameter(self.positions.new_zeros((len(proposals),4,3)))
        self.coefficients.append(parameter)
        self.iteration.fill_(iteration)
        return parameter

    def payload(self):
        return dict(format=self.FORMAT,ramp_steps=self.ramp_steps,batch_sizes=[len(x) for x in self.coefficients],
                    state={k:v.detach().cpu() for k,v in self.state_dict().items()})

    @classmethod
    def from_payload(cls,payload,device):
        if payload['format']!=cls.FORMAT:raise ValueError('Unknown residual node representation')
        module=cls(len(payload['state']['assignment']),device,payload['ramp_steps'])
        for key in module._buffers:setattr(module,key,payload['state'][key].to(device).clone())
        for n in payload['batch_sizes']:module.coefficients.append(nn.Parameter(torch.zeros((n,4,3),device=device)))
        module.load_state_dict(payload['state'],strict=True)
        return module


def install(model,module=None):
    if not hasattr(model,'_adaptive_original_geometry'):
        model._adaptive_original_geometry=model.deform_geometry_from_nodes
        def geometry(self,time):
            xyz,scale,rotation=self._adaptive_original_geometry(time)
            return self.adaptive_residual.apply(xyz,time),scale,rotation
        model.deform_geometry_from_nodes=types.MethodType(geometry,model)
    model.adaptive_residual=module if module is not None else ResidualNodes(len(model._xyz),model._xyz.device)
    return model.adaptive_residual


def register_optimizer(model):
    params=list(model.adaptive_residual.parameters())
    if not params:return
    group=next((g for g in model.optimizer.param_groups if g.get('name')=='adaptive_residual'),None)
    if group is None:
        lr=model.deformation_scheduler_args(int(model.adaptive_residual.iteration))
        model.optimizer.add_param_group(dict(params=params,lr=lr,name='adaptive_residual'))
    else:
        known={id(p) for p in group['params']}
        group['params'].extend(p for p in params if id(p) not in known)


def structure(model):
    """Immutable topology only; all existing geometry/appearance parameters train."""
    return dict(base_geometry=model.deformation_graph.state_dict(),appearance=model.appearance_graph.state_dict(),
        G3={k:v for k,v in model.graph_motion_residual.state_dict().items() if not k.startswith('decoder.')},
        J8={k:getattr(model.skinning,k) for k in ('indices','initial','target')})


class A20ResidualManager:
    def __init__(self,model,output,source_statistics=None):
        self.model=model
        self.config=AdaptiveNodeConfig(mode='split',max_nodes=2400,max_splits=60)
        self.observer=AdaptiveG3NodeManager(model,self.config,str(output),score='current',placement='centroid')
        if source_statistics is not None:import_pre_split_statistics(self.observer,source_statistics)
        self.events=[]
        self.structure_sha256=digest(structure(model)) if model.skinning is not None else None

    def update_statistics(self,gradient,visible):
        self.observer.update_statistics(gradient,visible)

    def is_event(self,i):return self.observer.is_event(i)

    @torch.no_grad()
    def proposals(self,iteration):
        residual=self.model.adaptive_residual
        stats=self.observer.node_statistics(iteration)
        candidates=self.observer._split_candidates(stats)
        room=400-len(residual.positions)
        quota=min(60,room)
        accepted=[];reasons=Counter()
        occupied=residual.assignment>=0
        for parent in candidates.tolist():
            if len(accepted)>=quota:reasons['cap_or_event_quota']+=1;continue
            p=self.observer._proposal(parent,stats)
            if p is None:reasons['original_centroid_or_separation_gate']+=1;continue
            position,support,minimum=p
            positions=[residual.positions]+([torch.stack([x['position'] for x in accepted])] if accepted else [])
            positions=torch.cat(positions)
            if len(positions) and ((positions-position).norm(dim=1)<minimum).any():
                reasons['residual_spatial_NMS']+=1;continue
            graph=self.model.deformation_graph
            membership=graph.gaussian_node_indices[support]==parent
            slot=membership.long().argmax(1)
            parent_weight=graph.gaussian_node_weights[support,slot]
            distance=(self.model._xyz.detach()[support]-position).norm(dim=1)
            radius=2*distance.median().clamp_min(1e-4)
            q=distance/radius
            radial=(1-q.clamp(0,1)).pow(4)*(4*q.clamp(0,1)+1)
            weights=radial*parent_weight/parent_weight.max().clamp_min(1e-8)
            valid=(weights>0)&~occupied[support]
            if int(valid.sum())<self.config.min_coverage:
                reasons['free_local_support_below32']+=1;continue
            rows=support[valid]
            accepted.append(dict(parent=parent,position=position,rows=rows,weights=weights[valid],radius=float(radius),
                                 score=float(stats['normalized'][parent])))
            occupied[rows]=True
        return accepted,dict(candidates=len(candidates),accepted=len(accepted),rejected=dict(reasons),
                             score_floor=self.config.split_z,cap_total_geometry=2400,
                             selector='A20 current RGB gradient EMA/log-z/centroid; fixed-backbone K4 support',
                             local_weight='compact Wendland radius2x median support-to-centroid; immutable disjoint support')

    def state_dict(self,i):
        observer=self.observer.state_dict(i);observer.pop('optimizer')
        return dict(observer=observer,events=self.events,structure_sha256=self.structure_sha256)

    @torch.no_grad()
    def load_statistics(self,state):
        core=state['observer']
        for key in ['gaussian_score','gaussian_observations','running_log_mean','running_log_var']:
            value=core[key];setattr(self.observer,key,None if value is None else value.to(self.model._xyz.device).clone())
        for key in ['direction_sum','direction_count','last_split','surface_weights']:
            value=core['adaptive_g3'][key];setattr(self.observer,key,value.to(getattr(self.observer,key)).clone())
        self.events=list(state['events']);self.structure_sha256=state['structure_sha256']
        assert digest(structure(self.model))==self.structure_sha256
