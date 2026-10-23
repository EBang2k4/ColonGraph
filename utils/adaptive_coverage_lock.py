"""TRAIN-only RGB/depth under-resolution selection and bounded XYZ residuals.

U1 and U2 share a single10k branchpoint, parent IDs and base architecture. The
legacy A20 files are intentionally unchanged. Selection gradients are proxies,
not measured physical motion or proof of missing geometric capacity.
"""
from collections import Counter
from pathlib import Path
import types

import numpy as np
import torch

from utils.adaptive_residual_nodes import ResidualNodes
from utils.adaptive_deformation_graph import AdaptiveNodeConfig,AdaptiveNodeManager
from utils.exact_split_branchpoint import digest


POLICY=dict(format='rgb_depth_vector_underres_coverage_v1',seed=0,window=[9001,10000],insertion=10000,
    child_count=132,activation_ramp=500,ema=.99,minimum_support=32,weight_threshold=.05,
    minimum_separation=.1,log_z_floor=1.,phi='positive log-difficulty z; eligibility z>=1',
    difficulty='K4-weighted mean EMA(||d RGB-L1/d XYZ|| + ||d native weighted depth-L1/d XYZ||)',
    vector='EMA(d (RGB-L1 + native weighted depth-L1)/d transformed XYZ), no KNN/TV/KL gradient',
    H='unweighted intra-support standard deviation of EMA vectors',score='phi(z)*sqrt(D*H)',
    U1_placement='Original A20 RGB-gradient-weighted centroid/PCA and compact K4 support',
    U2_placement='Two equal-count halves along gradient-vector PCA; farther unweighted spatial centroid; subcluster support',
    common_admission='Descending common score, require both placements to pass original spacing and free-support gates',
    U2_bound='radial tanh norm<=.35*nearest incident G3/base-neighbor spacing',
    trust_ratio=.35,TEST_for_selection=False,pruning=False,new_loss=False,ARAP=False)


def radial_bound(raw,radii):
    """Smooth Euclidean ball map, including a nonzero derivative at raw=0."""
    r=radii[:,None].clamp_min(1e-12)
    q=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)/r
    factor=torch.where(q>1e-4,torch.tanh(q)/q.clamp_min(1e-12),1-q.square()/3)
    return raw*factor


class CoverageResidualNodes(ResidualNodes):
    FORMAT='coverage_lock_radial_translation_linear4_v1'

    def __init__(self,n,device,ramp_steps=500):
        super().__init__(n,device,ramp_steps)
        self.register_buffer('trust_radii',torch.empty(0,device=device))

    @torch.no_grad()
    def insert(self,proposals,iteration):
        param=super().insert(proposals,iteration)
        if proposals:
            self.trust_radii=torch.cat((self.trust_radii,self.positions.new_tensor([p['trust_radius'] for p in proposals])))
        return param

    def translations(self,time):
        if not len(self.coefficients):return self.positions.new_empty((0,3))
        t=torch.as_tensor(time,device=self.positions.device).reshape(-1)[0].clamp(0,1)*3
        lo=t.floor().long().clamp(max=2);fraction=t-lo
        values=torch.cat(list(self.coefficients),0)
        raw=(1-fraction)*values[:,lo]+fraction*values[:,lo+1]
        return radial_bound(raw,self.trust_radii)*self.activation()[:,None]

    @classmethod
    def from_payload(cls,payload,device):
        if payload['format']!=cls.FORMAT:raise ValueError('Wrong coverage residual format')
        module=cls(len(payload['state']['assignment']),device,payload['ramp_steps'])
        for key in module._buffers:setattr(module,key,payload['state'][key].to(device).clone())
        for n in payload['batch_sizes']:
            module.coefficients.append(torch.nn.Parameter(torch.zeros((n,4,3),device=device)))
        module.load_state_dict(payload['state'],strict=True)
        return module


class SelectionStatistics:
    def __init__(self,n,device):
        self.rgb=torch.zeros(n,device=device)
        self.depth=torch.zeros(n,device=device)
        self.vector=torch.zeros(n,3,device=device)
        self.count=torch.zeros(n,device=device,dtype=torch.long)
        self.steps=0

    @torch.no_grad()
    def update(self,rgb_gradient,depth_gradient,visible):
        valid=visible&torch.isfinite(rgb_gradient).all(1)&torch.isfinite(depth_gradient).all(1)
        first=self.count[valid]==0;decay=POLICY['ema']
        for key,value in [('rgb',rgb_gradient.norm(dim=1)),('depth',depth_gradient.norm(dim=1)),
                          ('vector',rgb_gradient+depth_gradient)]:
            old=getattr(self,key)[valid];new=value[valid]
            mask=first if old.ndim==1 else first[:,None]
            getattr(self,key)[valid]=torch.where(mask,new,decay*old+(1-decay)*new)
        self.count[valid]+=1;self.steps+=1

    def payload(self):
        return dict(steps=self.steps,**{key:getattr(self,key).cpu() for key in ['rgb','depth','vector','count']})

    @classmethod
    def from_payload(cls,payload,device):
        m=cls(len(payload['count']),device);m.steps=payload['steps']
        for key in ['rgb','depth','vector','count']:setattr(m,key,payload[key].to(device).clone())
        return m


@torch.no_grad()
def parent_scores(graph,statistics):
    idx=graph.gaussian_node_indices.cpu().numpy();weights=graph.gaussian_node_weights.cpu().numpy()
    counts=statistics.count.cpu().numpy();vectors=statistics.vector.cpu().numpy()
    difficulty=(statistics.rgb+statistics.depth).cpu().numpy()
    n=len(graph.node_xyz);mass=np.zeros(n);sums=np.zeros(n);coverage=np.zeros(n,dtype=int)
    for column in range(4):
        np.add.at(mass,idx[:,column],weights[:,column])
        np.add.at(sums,idx[:,column],weights[:,column]*difficulty)
        np.add.at(coverage,idx[:,column],(weights[:,column]>.05)&(counts>0))
    D=sums/np.maximum(mass,1e-8);eligible=coverage>=32
    log=np.log(np.maximum(D,1e-8));mu=log[eligible].mean();sd=max(log[eligible].std(),1e-6)
    z=(log-mu)/sd
    gids,slots=np.nonzero((counts[:,None]>0)&(weights>.05));parents=idx[gids,slots]
    order=np.argsort(parents,kind='stable');gids,parents=gids[order],parents[order]
    bounds=np.searchsorted(parents,np.arange(n+1))
    rows=[]
    for parent in range(n):
        ids=np.unique(gids[bounds[parent]:bounds[parent+1]])
        v=vectors[ids].astype(np.float64)
        H=float(np.sqrt(((v-v.mean(0))**2).sum(1).mean())) if len(v)>=32 else 0.
        allowed=bool(eligible[parent] and z[parent]>=POLICY['log_z_floor'] and H>0)
        score=max(float(z[parent]),0)*np.sqrt(D[parent]*H) if allowed else 0.
        rows.append(dict(parent=parent,D=float(D[parent]),H=H,z=float(z[parent]),score=float(score),
                         coverage=int(coverage[parent]),eligible=allowed))
    return sorted(rows,key=lambda x:(-x['score'],x['parent']))


@torch.no_grad()
def local_support(model,position,support,occupied):
    graph=model.deformation_graph
    # Caller sets the candidate parent, preserving its original fixed K4 weight.
    parent=model._coverage_parent
    slot=(graph.gaussian_node_indices[support]==parent).long().argmax(1)
    parent_weight=graph.gaussian_node_weights[support,slot]
    distance=(model._xyz.detach()[support]-position).norm(dim=1)
    radius=2*distance.median().clamp_min(1e-4)
    q=(distance/radius).clamp(0,1)
    w=(1-q).pow(4)*(4*q+1)*parent_weight/parent_weight.max().clamp_min(1e-8)
    valid=(w>0)&~occupied[support]
    if int(valid.sum())<32:return None
    return support[valid],w[valid],float(radius)


@torch.no_grad()
def coverage_position(model,statistics,parent,support):
    if len(support)<64:return None
    vectors=statistics.vector[support].double();centered=vectors-vectors.mean(0)
    _,axes=torch.linalg.eigh(centered.T@centered/len(centered))
    axis=axes[:,-1]
    axis=axis*torch.where(axis[axis.abs().argmax()]<0,axis.new_tensor(-1.),axis.new_tensor(1.))
    projections=centered@axis
    # Stable Gaussian-ID tie-breaks; eigenvector sign cannot change the choice
    # of the farther spatial half, except equal-distance ties resolved by ID.
    order=torch.argsort(projections,stable=True);middle=len(support)//2
    halves=[support[order[:middle]],support[order[middle:]]]
    centers=[model._xyz.detach()[ids].mean(0) for ids in halves]
    point=model.deformation_graph.node_xyz[parent]
    distance=[float((c-point).norm()) for c in centers]
    choice=0 if distance[0]>distance[1] or (distance[0]==distance[1] and int(halves[0].min())<int(halves[1].min())) else 1
    return centers[choice],halves[choice]


@torch.no_grad()
def select_matched_parents(model,statistics):
    """Admit each parent only when both branches can preserve the shared budget."""
    graph=model.deformation_graph;xyz=model._xyz.detach();device=xyz.device
    observer=AdaptiveNodeManager.__new__(AdaptiveNodeManager)
    observer.model=model;observer.config=AdaptiveNodeConfig(mode='split')
    observer.gaussian_observations=statistics.count
    observer.gaussian_score=statistics.rgb  # U1 retains original RGB placement.
    rows=parent_scores(graph,statistics);accepted={'U1':[],'U2':[]};reasons=Counter()
    occupied={k:torch.zeros(len(xyz),device=device,dtype=torch.bool) for k in accepted}
    edge=model.graph_motion_residual.edge_indices
    for row in rows:
        parent=row['parent']
        if not row['eligible']:
            row['reason']='original_z_or_coverage_or_H_gate';continue
        if len(accepted['U1'])==132:
            row['reason']='matched_budget_reached';continue
        original=observer._proposal(parent,{})
        if original is None:
            row['reason']='U1_original_placement_gate';reasons[row['reason']]+=1;continue
        p1,support,minimum=original
        balanced=coverage_position(model,statistics,parent,support)
        if balanced is None:
            row['reason']='U2_two_halves_below32';reasons[row['reason']]+=1;continue
        p2,subsupport=balanced
        if torch.cdist(p2[None],graph.node_xyz).min()<minimum:
            row['reason']='U2_original_separation_gate';reasons[row['reason']]+=1;continue
        incident=(edge[0]==parent)|(edge[1]==parent)
        neighbors=torch.where(edge[0,incident]==parent,edge[1,incident],edge[0,incident]).unique()
        if not len(neighbors):raise ValueError('Base parent has no G3 neighbor')
        hp=float((graph.node_xyz[neighbors]-graph.node_xyz[parent]).norm(dim=1).min())
        proposals={};reason=None
        model._coverage_parent=parent
        for branch,pos,ids in [('U1',p1,support),('U2',p2,subsupport)]:
            previous=accepted[branch]
            if previous and (torch.stack([x['position'] for x in previous])-pos).norm(dim=1).min()<minimum:
                reason=branch+'_residual_NMS';break
            local=local_support(model,pos,ids,occupied[branch])
            if local is None:reason=branch+'_free_support_below32';break
            gids,w,radius=local
            proposals[branch]=dict(parent=parent,position=pos,rows=gids,weights=w,radius=radius,
                trust_radius=POLICY['trust_ratio']*hp,parent_spacing=hp,score=row['score'])
        if reason:
            row['reason']=reason;reasons[reason]+=1;continue
        for branch in accepted:
            p=proposals[branch];accepted[branch].append(p);occupied[branch][p['rows']]=True
        row.update(reason='accepted',U1_position=p1.tolist(),U2_position=p2.tolist(),parent_spacing=hp,
                   U1_support=len(proposals['U1']['rows']),U2_support=len(proposals['U2']['rows']))
    if hasattr(model,'_coverage_parent'):del model._coverage_parent
    return accepted,dict(policy=POLICY,eligible=sum(r['eligible'] for r in rows),accepted=len(accepted['U1']),
        requested=132,rejected=dict(reasons),candidate_ledger=rows,
        parents=[p['parent'] for p in accepted['U1']],matched_parents=True,
        rank_and_admission_note='Common feasibility gate couples both placements; U1 is not an isolated historical-schedule ablation of A20.')
