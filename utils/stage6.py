"""Stage6: exact requested Charbonnier schedules and WideR6-96 appearance."""
import math
import torch
from torch import nn
from scene.deformation import NodeConditionedResidualAppearance,NodeDeformationMLP


def photometric(iteration):
    t=min(1.,max(0.,(iteration-15000)/15000))
    return .80-.65*t,.20,2.+12.*t


def dynamics(iteration):
    t=min(1.,max(0.,(iteration-15000)/15000))
    eta=.12+.44*(1+math.cos(math.pi*t))
    t=min(1.,max(0.,(iteration-8000)/14000))
    return eta,.25-.245*t,.01-.0098*t


class R6AdditiveWide96(NodeConditionedResidualAppearance):
    def __init__(self,cfg,num_nodes):
        super().__init__(cfg,num_nodes)
        if num_nodes!=2000 or self.node_appearance_dim!=16:
            raise ValueError('WideR6-96 requires original2000 K4 appearance nodes with16D features')
        if self.eta!=.5 or not self.use_tanh:
            raise ValueError('WideR6-96 uses .50 tanh multiplicative gain')
        self.additive_cap=float(getattr(cfg,'r6_additive_cap',.25))
        if self.additive_cap!=.25:raise ValueError('WideR6-96 uses .25 tanh additive cap')
        with torch.random.fork_rng(devices=[]):
            self.residual_mlp=nn.Sequential(nn.Linear(45,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,3))
            self.additive_head=nn.Linear(96,3)
            generator=torch.Generator(device='cpu').manual_seed(int(cfg.appearance_seed))
            for layer in (self.residual_mlp[0],self.residual_mlp[2]):
                nn.init.kaiming_uniform_(layer.weight,a=math.sqrt(5),generator=generator)
                bound=1/math.sqrt(layer.weight.shape[1]);nn.init.uniform_(layer.bias,-bound,bound,generator=generator)
            nn.init.zeros_(self.node_appearance)
            for head in (self.residual_mlp[4],self.additive_head):
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    @torch.no_grad()
    def bind_fine0_features(self,original):
        if original.shape!=(2000,8):raise ValueError('Expected exact sharedfine0 appearance features2000x8')
        self.node_appearance[:,:8].copy_(original)
        self.node_appearance[:,8:].zero_()

    def components(self,times,colors,embedding,indices,weights):
        t=torch.as_tensor(times,device=embedding.device,dtype=embedding.dtype).reshape(-1,1).expand(len(embedding),1)
        x=torch.cat((embedding,self.node_context(indices,weights),NodeDeformationMLP.encode(t,self.time_frequencies)),-1)
        h=self.residual_mlp[:4](x)
        return .5*self.residual_mlp[4](h).tanh(),self.additive_cap*self.additive_head(h).tanh()

    def forward(self,points,times,colors,embedding,node_indices=None,node_weights=None):
        gain,beta=self.components(times,colors,embedding,node_indices,node_weights)
        return colors*(1+gain)+beta,gain
