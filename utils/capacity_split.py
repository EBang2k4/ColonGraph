"""Rcap: unchanged persistent pool/ordering + isolated-child capacity gates."""
import json
from pathlib import Path
import torch
from utils.persistent_split import PersistentSplitManager


class CapacitySplitManager(PersistentSplitManager):
    def __init__(self,*args,capacity_reference='',**kwargs):
        if kwargs.get('signal') not in ('persistent_capacity', 'persistent_underres'):
            raise ValueError('Rcap requires explicit persistent_capacity signal')
        super().__init__(*args,**kwargs)
        self.capacity_reference = str(capacity_reference)
        self.capacity_policy = (json.loads(Path(capacity_reference).read_text()) if capacity_reference else None)
        if self.capacity_policy is not None:
            if self.capacity_policy.get('format')!='rcap_train_only_capacity_v1':
                raise ValueError('Invalid capacity calibration')
            if (self.capacity_policy['min_child_support'] != 32 or
                    self.capacity_policy['min_rendered_mass_share'] <= 0):
                raise ValueError('Preregistered support/mass floor mismatch')
        self._admission_proposal_details={}

    @torch.no_grad()
    def expected_capacity(self,parent,position):
        if self._bank is None or 'control_mass' not in self._bank:
            raise RuntimeError('Need frozen training probe contribution bank')
        graph=self.graph;m=len(graph.node_xyz)
        nodes=torch.cat((graph.node_xyz,position[None]))
        _,rows,idx,w=self._local_child_update(graph.gaussian_node_indices,
            graph.gaussian_node_weights,nodes,parent,position,m)
        child_weight=(w*(idx==m)).sum(1)
        mass=self._bank['control_mass']
        control=(mass[rows]*child_weight).sum()
        return dict(expected_child_support=int((child_weight>.05).sum()),
                    expected_child_influence=float(child_weight.sum()),
                    expected_child_enters_fraction=float((child_weight>0).sum()/len(graph.gaussian_node_indices)),
                    expected_rendered_mass=float(control),
                    expected_rendered_mass_share=float(control/mass.sum().clamp_min(1e-30)))

    @torch.no_grad()
    def _proposal(self,parent,stats):
        proposal=super()._proposal(parent,stats)
        if proposal is None:return None
        if self.capacity_policy is None:raise RuntimeError('Rcap gate must be calibrated before admission')
        row=self.expected_capacity(parent,proposal[0])
        row['capacity_reject_reason']=None
        if row['expected_child_support']<self.capacity_policy['min_child_support']:
            row['capacity_reject_reason']='expected_child_support'
        elif row['expected_rendered_mass_share']<self.capacity_policy['min_rendered_mass_share']:
            row['capacity_reject_reason']='expected_rendered_control_mass'
        self._admission_proposal_details[parent]=row
        return None if row['capacity_reject_reason'] else proposal

    def adapt(self,*args,**kwargs):
        self._admission_proposal_details={}
        return super().adapt(*args,**kwargs)

    def _record(self,event):
        event['capacity_gate']=dict(policy=self.capacity_policy,
            mass='sum(A_training^T valid * hypothetical child K4 weight), foreground-mass normalized',
            isolated_potential=True,admission_counts=self._admission_audit['counts'])
        return super()._record(event)

    def state_dict(self,iteration):
        state=super().state_dict(iteration)
        state['capacity_split']=self.capacity_policy
        return state

    def load_state_dict(self,state):
        if state.get('capacity_split')!=self.capacity_policy:
            raise ValueError('Capacity gate changed on resume')
        super().load_state_dict(state)
