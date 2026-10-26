"""TRAIN-selected harmonic XYZ controllers. No changes to backbone/appearance topology."""
import torch
from torch import nn
from utils.adaptive_coverage_lock import radial_bound

class HarmonicNodes(nn.Module):
    FORMAT='stage12_harmonic_pe4_radial_v1'
    def __init__(self,n,device,handshake_ramp=True,count=132):
        super().__init__()
        self.handshake_ramp=bool(handshake_ramp)
        assert 1<=count<=150
        self.coefficients=nn.Parameter(torch.zeros(count,3,9,device=device))
        self.register_buffer('assignment',torch.full((n,),-1,dtype=torch.long,device=device))
        self.register_buffer('influence',torch.zeros(n,device=device))
        self.register_buffer('support_ids',torch.empty(0,dtype=torch.long,device=device))
        self.register_buffer('positions',torch.zeros(count,3,device=device))
        self.register_buffer('parents',torch.zeros(count,dtype=torch.long,device=device))
        self.register_buffer('trust_radii',torch.ones(count,device=device))
        self.register_buffer('iteration',torch.tensor(10000,device=device))
        self.register_buffer('frequencies',2.**torch.arange(4,device=device))
    def activation(self):return ((self.iteration-10000)/500.).clamp(0,1)
    def translations(self,time):
        t=torch.as_tensor(time,device=self.coefficients.device,dtype=self.coefficients.dtype).reshape(-1)[0]
        phase=t*self.frequencies
        pe=torch.cat((t.reshape(1),phase.sin(),phase.cos()))
        raw=torch.matmul(self.coefficients,pe)
        # U2's smooth radial tanh, with analytic zero limit (nonzero learning gradient).
        return self.activation()*radial_bound(raw,self.trust_radii)
    def correction(self,time):
        ids=self.support_ids
        values=self.influence[ids,None]*self.translations(time)[self.assignment[ids]]
        return self.influence.new_zeros((len(self.assignment),3)).index_copy(0,ids,values)
    def micro_factor(self):
        a=self.activation() if self.handshake_ramp else 1.
        return 1-.5*a*self.influence[:,None]
    def regularization(self):return 1e-4*self.coefficients.square().sum((1,2)).mean()
    @torch.no_grad()
    def insert(self,proposals):
        assert len(proposals)==len(self.coefficients)
        for slot,p in enumerate(proposals):
            ids=p['rows'];w=p['weights']
            assert len(ids)>=32 and (self.assignment[ids]<0).all()
            assert torch.isfinite(w).all() and ((w>0)&(w<=1)).all()
            self.assignment[ids]=slot;self.influence[ids]=w
            self.parents[slot]=p['parent'];self.positions[slot]=p['position'];self.trust_radii[slot]=p['trust_radius']
        self.support_ids=(self.assignment>=0).nonzero().flatten()
        assert len(self.parents.unique())==len(proposals) and (self.trust_radii>0).all()
    def payload(self):
        return dict(format=self.FORMAT,handshake_ramp=self.handshake_ramp,state={k:v.detach().cpu() for k,v in self.state_dict().items()})
    @classmethod
    def from_payload(cls,payload,device):
        assert payload['format']==cls.FORMAT
        m=cls(len(payload['state']['assignment']),device,payload['handshake_ramp'],len(payload['state']['coefficients']))
        m.support_ids=payload['state']['support_ids'].to(device).clone()
        m.load_state_dict(payload['state']);return m


def register_optimizer(m):
    if not hasattr(m,'harmonic_nodes'):return
    assert not any(g['name']=='harmonic_nodes' for g in m.optimizer.param_groups)
    m.optimizer.add_param_group(dict(params=list(m.harmonic_nodes.parameters()),lr=.001,name='harmonic_nodes'))
