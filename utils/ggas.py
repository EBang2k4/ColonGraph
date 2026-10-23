"""Explicit, fixed-candidate graph-guided skinning. Opt-in; original K4 stays intact."""
from pathlib import Path
import json
import torch
from torch import nn
from utils.exact_split_branchpoint import digest


class AdaptiveSkinning(nn.Module):
    def __init__(self, indices, initial, target, learned=False, ramp_end=6000, bounded=False, delta_max=1.5):
        super().__init__()
        if indices.shape != initial.shape or initial.shape != target.shape:
            raise ValueError('Candidate/prior shape mismatch')
        if not torch.isfinite(target).all() or (target < 0).any():raise ValueError('Invalid prior')
        if not torch.allclose(target.sum(-1),torch.ones_like(target[:,0]),atol=2e-6):raise ValueError('Prior not normalized')
        self.register_buffer('indices',indices.detach().long().clone())
        self.register_buffer('initial',initial.detach().clone())
        self.register_buffer('target',target.detach().clone())
        self.register_buffer('phase',target.new_zeros(()))
        self.ramp_end=int(ramp_end)
        self.delta=nn.Parameter(torch.zeros_like(target)) if learned else None
        self.bounded=bool(bounded)
        if bounded:
            if not learned or not 0 < float(delta_max) < float('inf'):
                raise ValueError('Bounded skinning requires learned parameters and a finite positive cap')
            # Stored in the model digest as well as the native format metadata.
            self.register_buffer('correction_cap',target.new_tensor(delta_max))

    def set_iteration(self,iteration):
        self.phase.fill_(min(1.,max(0.,(iteration-5000)/max(1,self.ramp_end-5000))))

    def prior(self):return self.initial + self.phase*(self.target-self.initial)

    def centered_logits(self):
        if self.delta is None:return torch.zeros_like(self.target)
        valid=self.target>0
        mean=(self.delta*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        return (self.delta-mean)*valid

    def corrections(self):
        if self.delta is None:return torch.zeros_like(self.target)
        if not self.bounded:return self.delta
        eta=self.centered_logits()
        return self.correction_cap*torch.tanh(eta/self.correction_cap)

    def weights(self):
        p=self.prior()
        if self.delta is None:return p
        logp=torch.where(p>0,p.clamp_min(1e-38).log(),torch.full_like(p,-torch.inf))
        return torch.softmax(logp+self.corrections(),dim=-1)

    def kl(self):
        if self.delta is None:return self.target.new_zeros(())
        p=self.prior();w=self.weights()
        return (w*(w.clamp_min(1e-38).log()-p.clamp_min(1e-38).log())).sum(-1).mean()

    def blend(self,*values):
        weights=self.weights().unsqueeze(-1)
        return tuple((v[self.indices]*weights).sum(1) for v in values)

    def payload(self):
        p=dict(format_version=1,learned=self.delta is not None,ramp_end=self.ramp_end,state_dict=self.state_dict())
        if self.bounded:p.update(format_version=2,parameterization='centered_tanh',delta_max=float(self.correction_cap))
        return p

    @classmethod
    def from_payload(cls,payload,device):
        version=payload['format_version']
        if version not in (1,2):raise ValueError('Unknown skinning checkpoint format')
        if version==2 and payload['parameterization']!='centered_tanh':raise ValueError('Unknown bounded parameterization')
        s=payload['state_dict'];m=cls(s['indices'],s['initial'],s['target'],payload['learned'],payload['ramp_end'],
            bounded=version==2,delta_max=payload.get('delta_max',1.5)).to(device)
        m.load_state_dict(s,strict=True)
        if version==2 and float(m.correction_cap)!=float(payload['delta_max']):raise ValueError('Bound cap metadata/state mismatch')
        return m


def add_optimizer(model):
    m=getattr(model,'skinning',None)
    if m is not None and m.delta is not None and not any(g['name']=='skinning' for g in model.optimizer.param_groups):
        model.optimizer.add_param_group(dict(params=[m.delta],lr=float(model.args.ggas_lr),name='skinning'))


def initialize(model,output):
    if model.args.ggas_variant=='none':return
    if model.args.geometry_regularizer!='knn99' or model.args.geometry_study_refresh!='none':raise ValueError('GGAS requires unchanged KNN99/fixed candidates')
    if model.args.ggas_variant=='A_BASE':return
    if model.skinning is not None:return
    p=torch.load(model.args.ggas_candidates,map_location=model._xyz.device,weights_only=True)
    g=model.deformation_graph
    if digest(g.gaussian_node_indices)!=p['original_indices_sha256'] or digest(g.gaussian_node_weights)!=p['original_weights_sha256']:
        raise ValueError('Candidate source differs from exact original repaired K4')
    if digest(model._xyz)!=p['source_xyz_sha256']:raise ValueError('Candidates must initialize on original5k geometry')
    variant=model.args.ggas_variant
    if variant=='B_EUCLIDEAN8':ids=p['euclidean_indices'];target=p['euclidean_weights'];initial=target
    elif variant in ('C_GRAPH8_FIXED','D_GRAPH8_LEARNED','D_GGAS_V1','G_BOUNDED'):
        ids=p['graph_indices'];target=p['graph_prior'];initial=torch.cat((g.gaussian_node_weights,torch.zeros_like(g.gaussian_node_weights)),1)
    elif variant in ('H_NO_MOTION_PRIOR','I_EUCLIDEAN8_LEARNED','J_NO_MOTION_SELECTION'):
        key={'H_NO_MOTION_PRIOR':('graph_indices','graph_spatial_prior'),
             'I_EUCLIDEAN8_LEARNED':('euclidean_indices','euclidean_motion_prior'),
             'J_NO_MOTION_SELECTION':('spatial_graph_indices','spatial_graph_prior')}[variant]
        ids=p[key[0]];target=p[key[1]]
        initial=torch.cat((g.gaussian_node_weights,torch.zeros_like(g.gaussian_node_weights)),1)
    elif variant=='E_K4_LEARNED':ids=g.gaussian_node_indices;target=g.gaussian_node_weights;initial=target
    else:raise ValueError('Unsupported/unverified GGAS variant '+variant)
    learned=variant not in ('B_EUCLIDEAN8','C_GRAPH8_FIXED')
    model.skinning=AdaptiveSkinning(ids,initial,target,learned,model.args.ggas_ramp_end,
        bounded=variant=='G_BOUNDED',delta_max=getattr(model.args,'ggas_delta_max',1.5))
    before=digest(model.optimizer.state_dict());add_optimizer(model)
    Path(output,'skinning_initial.json').write_text(json.dumps(dict(variant=variant,original_optimizer_sha256=before,
        candidates_sha256=digest(ids),initial_prior_sha256=digest(initial),target_prior_sha256=digest(target),
        learned=model.skinning.delta is not None,lr=model.args.ggas_lr,lambda_skin=model.args.ggas_lambda,
        parameterization='centered_tanh' if model.skinning.bounded else 'unbounded',
        delta_max=float(model.skinning.correction_cap) if model.skinning.bounded else None),indent=2)+'\n')


@torch.no_grad()
def diagnostic(model,iteration,output,losses):
    m=model.skinning
    row=dict(iteration=iteration,**{k:float(v.detach()) for k,v in losses.items()})
    row['lr']={g['name']:g['lr'] for g in model.optimizer.param_groups}
    if m is not None:
        w=m.weights();row.update(entropy=float(-(w*w.clamp_min(1e-38).log()).sum(-1).mean()),
            effective_K=float((w.square().sum(-1)).reciprocal().mean()),new_mass=float(w[:,4:].sum(-1).mean()),
            max_weight=float(w.max()),KL=float(m.kl()),phase=float(m.phase))
        if m.delta is not None:
            grad=m.delta.grad
            if grad is not None and not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite skinning gradient')
            if not torch.isfinite(m.delta).all():raise FloatingPointError('Nonfinite learned logits')
            row.update(logit_abs_max=float(m.delta.abs().max()),logit_RMS=float(m.delta.square().mean().sqrt()),
                logit_grad_norm=0. if grad is None else float(grad.norm()),logit_grad_absmax=0. if grad is None else float(grad.abs().max()))
            if getattr(model.args,'ggas_study_version',1)==2:
                centered=m.centered_logits();correction=m.corrections()
                row.update(centered_logit_abs_max=float(centered.abs().max()),centered_logit_RMS=float(centered.square().mean().sqrt()),
                    correction_abs_max=float(correction.abs().max()),correction_span_max=float((correction.max(1).values-correction.min(1).values).max()),
                    bound_saturation_fraction=float((correction.abs()>.95*m.correction_cap).float().mean()) if m.bounded else 0.)
    with Path(output,'skinning_diagnostics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
