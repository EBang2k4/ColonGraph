"""Single frozen-TRAIN admission; rigid-invariant edge-length/hinge statistics."""
import numpy as np
import torch
from collections import Counter
from utils.adaptive_coverage_lock import parent_scores,coverage_position,local_support

@torch.no_grad()
def intrinsic_strain(z,trajectories,edges):
    pairs=torch.unique(torch.sort(edges.T,dim=1).values,dim=0)
    pairs=pairs[pairs[:,0]!=pairs[:,1]];a,b=pairs.T
    v=trajectories[:,b]-trajectories[:,a];length=v.norm(dim=-1)
    rest=(z[a]-z[b]).norm(dim=1)
    edge_stretch=((length-length[:1])/(rest[None]+1e-6)).square().mean(0).sqrt()
    sums=z.new_zeros(len(z));counts=z.new_zeros(len(z))
    for ids in (a,b):
        sums.index_add_(0,ids,edge_stretch);counts.index_add_(0,ids,torch.ones_like(edge_stretch))
    bends=z.new_zeros(len(z));neighbors=[[] for _ in range(len(z))]
    for p,q in pairs.cpu().tolist():neighbors[p].append(q);neighbors[q].append(p)
    for p,ns in enumerate(neighbors):
        if len(ns)<2:continue
        ns=sorted(ns);delta=trajectories[:,ns]-trajectories[:,p,None]
        unit=delta/(delta.norm(dim=-1,keepdim=True)+1e-6)
        q,r=torch.triu_indices(len(ns),len(ns),offset=1,device=z.device)
        cosine=(unit[:,q]*unit[:,r]).sum(-1)
        bends[p]=(cosine-cosine[:1]).square().mean(0).sqrt().mean()
    return sums/counts.clamp_min(1),bends,pairs,counts

@torch.no_grad()
def select(model,statistics,times):
    graph=model.deformation_graph;xyz=model._xyz.detach();z=graph.node_xyz
    rows=parent_scores(graph,statistics);rows.sort(key=lambda r:r['parent'])
    photo=np.asarray([np.sqrt(r['D']*r['H']) for r in rows])
    positions=torch.stack([z+model._query_node_motion(float(t))[0] for t in times])
    stretch,bend,edges,degree=intrinsic_strain(z,positions,model.graph_motion_residual.edge_indices)
    normalization={}
    def normalize(v,key):
        lo,hi=np.percentile(v,[5,95]);normalization[key]=dict(p5=float(lo),p95=float(hi),population='all backbone nodes')
        return np.clip((v-lo)/max(float(hi-lo),1e-12),0,1)
    stretch=stretch.cpu().numpy();bend=bend.cpu().numpy()
    score=.65*normalize(photo,'photo')+.35*(.5*normalize(stretch,'stretch')+.5*normalize(bend,'bend'))
    zs=(score-score.mean())/(score.std()+1e-6)
    for p,r in enumerate(rows):r.update(photo=float(photo[p]),stretch=float(stretch[p]),bend=float(bend[p]),score=float(score[p]),z_Psi=float(zs[p]))
    rows.sort(key=lambda r:(-r['score'],r['parent']))
    occupied=torch.zeros(len(xyz),dtype=torch.bool,device=xyz.device)
    proposals=[];reasons=Counter();significant=0
    for r in rows:
        parent=r['parent']
        if len(proposals)==150:r['reason']='cap150';continue
        if r['z_Psi']<1 and len(proposals)>=16:r['reason']='below_z_threshold';continue
        support=((graph.gaussian_node_indices==parent).any(1)&(statistics.count>0)).nonzero().flatten()
        if len(support)<64 or degree[parent]==0:r['reason']='insufficient_support_or_G3';reasons[r['reason']]+=1;continue
        vec=statistics.vector[support].double();cov=(vec-vec.mean(0)).T@(vec-vec.mean(0))
        if not torch.isfinite(cov).all() or torch.linalg.eigvalsh(cov)[-1]<=0:
            r['reason']='degenerate_PCA';reasons[r['reason']]+=1;continue
        pos,sub=coverage_position(model,statistics,parent,support)
        minimum=.1*(xyz[support]-z[parent]).norm(dim=1).median().clamp_min(1e-4)
        if torch.cdist(pos[None],z).min()<minimum:
            r['reason']='base_spacing';reasons[r['reason']]+=1;continue
        if proposals and (torch.stack([p['position'] for p in proposals])-pos).norm(dim=1).min()<minimum:
            r['reason']='child_spacing';reasons[r['reason']]+=1;continue
        model._coverage_parent=parent;local=local_support(model,pos,sub,occupied)
        if local is None:r['reason']='free_support_below32';reasons[r['reason']]+=1;continue
        ids,w,radius=local;incident=(edges[:,0]==parent)|(edges[:,1]==parent)
        neighbors=edges[incident].flatten().unique();neighbors=neighbors[neighbors!=parent]
        hp=float((z[neighbors]-z[parent]).norm(dim=1).min())
        proposals.append(dict(parent=parent,position=pos,rows=ids,weights=w,radius=radius,trust_radius=.35*hp,parent_spacing=hp,score=r['score']))
        occupied[ids]=True;significant+=int(r['z_Psi']>=1)
        r.update(reason='accepted_significant' if r['z_Psi']>=1 else 'accepted_min16_fallback',support=len(ids),minimum_spacing=float(minimum),hp=hp)
    if hasattr(model,'_coverage_parent'):del model._coverage_parent
    audit=dict(minimum=16,maximum=150,accepted=len(proposals),significant_admitted=significant,fallback_admitted=len(proposals)-significant,geometry_backbone=len(z),appearance_nodes=len(model.appearance_graph.node_xyz),times=list(map(float,times)),normalization=normalization,score_mean=float(score.mean()),score_std=float(score.std()),z_threshold=1.,rejected=dict(reasons),ledger=rows,TEST_used=False,
       strain_caveat='Rigid-invariant graph length/hinge proxy; G3 adjacency does not certify anatomical topology')
    return proposals,audit
