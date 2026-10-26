"""Fixed shared-node alignment; only learned correction logits receive gradients."""
import json,hashlib
from pathlib import Path
import torch
from utils.exact_split_branchpoint import digest


def shared_disagreement(correction,edges,right_slot,mask):
    """Per-edge mean squared, shared-set centered differences, aligned by node ID."""
    i,j=edges.long().unbind(1)
    left=correction[i];right=correction[j].gather(1,right_slot.long())
    count=mask.sum(1)
    if bool((count<2).any()):raise ValueError('At least two valid shared nodes required')
    diff=(left-right).masked_fill(~mask,0)
    centered=(diff-diff.sum(1,keepdim=True)/count[:,None]).masked_fill(~mask,0)
    return centered.square().sum(1)/count


class SurfaceConsistency:
    """External immutable training-loss cache; no trainable module/optimizer group."""
    def __init__(self,path,skinning,batch_size=65536):
        p=torch.load(path,map_location=skinning.indices.device,weights_only=True)
        if p['format_version']!=1 or p['candidates_sha256']!=digest(skinning.indices):
            raise ValueError('SC cache/candidate mismatch')
        self.edges=p['edges'];self.right_slot=p['right_slot'];self.mask=p['shared_mask'];self.order=p['permutation']
        self.batch_size=int(batch_size);self.count=len(self.edges)
        if self.count<1 or self.batch_size<1:raise ValueError('Empty graph or invalid sample size')
        if (skinning.target<=0).any():raise ValueError('SC requires all J target candidates positive')
        # Verify alignment once, not in every training forward.
        for lo in range(0,self.count,65536):
            e=self.edges[lo:lo+65536].long();m=self.mask[lo:lo+65536];slot=self.right_slot[lo:lo+65536].long()
            left=skinning.indices[e[:,0]];right=skinning.indices[e[:,1]].gather(1,slot)
            if (m.sum(1)<2).any() or (left[m]!=right[m]).any():raise ValueError('SC shared-node alignment failed')
        self.file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def values(self,correction,index):
        return shared_disagreement(correction,self.edges[index],self.right_slot[index],self.mask[index])

    def loss(self,correction,iteration):
        offset=(int(iteration)-6001)*self.batch_size
        index=self.order[(torch.arange(self.batch_size,device=self.order.device)+offset)%self.count].long()
        value=self.values(correction,index).mean()
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite surface-consistency loss')
        return value

    @torch.no_grad()
    def full_statistics(self,correction):
        means=[];total=0.
        for lo in range(0,self.count,self.batch_size):
            idx=self.order[lo:lo+self.batch_size].long();v=self.values(correction,idx)
            means.append(float(v.mean()));total+=float(v.double().sum())
        return dict(full_mean=total/self.count,partition_means=means,edges=self.count,
            note='Fixed-permutation partition variation; descriptive sampling variance, not independent random draws')


def initialize(model,output):
    args=model.args
    if not args.ggas_sc_edges:return
    if args.ggas_sc_lambda<0:raise ValueError('Negative SC coefficient')
    if model.skinning is None:
        if args.ggas_sc_lambda!=0:raise ValueError('SC requires learned J skinning')
        return
    if model.skinning.bounded or args.ggas_variant!='J_NO_MOTION_SELECTION':raise ValueError('SC is defined for unchanged unbounded J only')
    model.skinning_consistency=SurfaceConsistency(args.ggas_sc_edges,model.skinning,args.ggas_sc_batch_size)
    row=dict(cache_sha256=model.skinning_consistency.file_sha256,edges=model.skinning_consistency.count,
        batch_size=args.ggas_sc_batch_size,lambda_SC=args.ggas_sc_lambda,start_iteration=args.ggas_sc_start,
        candidates_sha256=digest(model.skinning.indices),no_new_parameters=True,
        sampler='Fixed private-seeded permutation, cyclic65536 edge mean; iteration indexed, no RNG consumption')
    Path(output,'skinning_consistency_initial.json').write_text(json.dumps(row,indent=2)+'\n')


def training_term(model,existing_loss,knn_loss,kl_loss,iteration,output):
    args=model.args;sc=getattr(model,'skinning_consistency',None)
    if sc is None:return existing_loss+args.ggas_lambda*kl_loss
    active=iteration>=args.ggas_sc_start and not getattr(args,'geometry_study_frozen',False)
    sampled=(iteration%100==0 or iteration==6001)
    calibrate=args.ggas_sc_calibration and active and (iteration==6001 or iteration%50==0)
    raw=existing_loss.new_zeros(());elapsed=None
    if active and (args.ggas_sc_lambda>0 or calibrate or sampled):
        if float(model.skinning.phase)!=1.:raise RuntimeError('SC activated before prior ramp completed')
        if sampled:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        if args.ggas_sc_lambda>0 or calibrate:raw=sc.loss(model.skinning.corrections(),iteration)
        else:
            with torch.no_grad():raw=sc.loss(model.skinning.corrections(),iteration)
        if sampled:end.record();end.synchronize();elapsed=start.elapsed_time(end)
    # Exact lambda=0 path: no zero-valued extra term in the backward graph.
    loss=existing_loss+args.ggas_lambda*kl_loss
    if active and args.ggas_sc_lambda>0:loss=loss+args.ggas_sc_lambda*raw
    row=dict(iteration=iteration,active=active,lambda_SC=args.ggas_sc_lambda,SC=float(raw.detach()),
        weighted_SC=float(raw.detach())*args.ggas_sc_lambda,SC_forward_ms=elapsed)
    if calibrate:
        terms=dict(reconstruction=existing_loss-knn_loss,KNN99=knn_loss,KL=args.ggas_lambda*kl_loss,SC=raw)
        row['losses']={k:float(v.detach()) for k,v in terms.items()};row['gradient_norms']={};row['gradient_ms']={}
        for name,value in terms.items():
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
            grad=torch.autograd.grad(value,model.skinning.delta,retain_graph=True,allow_unused=False)[0]
            end.record();end.synchronize()
            if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite SC calibration gradient')
            row['gradient_norms'][name]=float(grad.norm());row['gradient_ms'][name]=start.elapsed_time(end)
        if iteration%100==0:row['sampling']=sc.full_statistics(model.skinning.corrections())
    if sampled or calibrate:
        with Path(output,'skinning_consistency_diagnostics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    return loss
