import copy
import unittest
import torch
from utils.connectivity_ablation import euclidean_reference,restore_optimizer_independent
from utils.exact_split_branchpoint import digest
from utils.graph_forward import GraphMotionResidual


class ConnectivityTests(unittest.TestCase):
    def test_smoke_updates_do_not_mutate_cached_source_optimizer(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
        source_optimizer=torch.optim.Adam([parameter])
        parameter.grad=torch.ones_like(parameter);source_optimizer.step()
        source=source_optimizer.state_dict();before=digest(source)
        initial=[]
        for _ in range(2):
            child=torch.nn.Parameter(parameter.detach().clone())
            optimizer=torch.optim.Adam([child])
            restore_optimizer_independent(optimizer,source)
            initial.append(digest(optimizer.state_dict()))
            self.assertNotEqual(optimizer.state[child]['step'].data_ptr(),source['state'][0]['step'].data_ptr())
            child.grad=torch.ones_like(child);optimizer.step()
            self.assertEqual(digest(source),before)
        self.assertEqual(initial,[before,before])

    def fixture(self):
        generator=torch.Generator().manual_seed(82)
        nodes=torch.randn(20,3,generator=generator)
        displacement=torch.randn(16,20,3,generator=generator)
        surface=dict(canonical_node_xyz=nodes,graph_k4_indices=torch.arange(4).repeat(7,1),
                     graph_k4_weights=torch.ones(7,4)/4,trajectory_times=torch.linspace(0,1,16),
                     audit=dict(trajectory_sigma=2.7))
        geodesic=torch.cdist(nodes,nodes)*1.5
        return surface,displacement,geodesic,2.

    def test_only_connectivity_fields_change_and_source_not_mutated(self):
        surface,motion,geodesic,sigma=self.fixture();saved=copy.deepcopy(surface);result=euclidean_reference(surface,motion,geodesic,sigma)
        for k in ('canonical_node_xyz','graph_k4_indices','graph_k4_weights','trajectory_times'):
            self.assertTrue(torch.equal(surface[k],saved[k]));self.assertTrue(torch.equal(result[k],saved[k]))
        self.assertEqual(surface['audit'],saved['audit'])
        self.assertEqual(result['audit']['trajectory_sigma'],2.7)
        self.assertFalse(result['audit']['connectivity_ablation']['trajectory_sigma_recalibrated'])

    def test_edges_equal_symmetric_union_of_euclidean_knn6(self):
        surface,motion,geodesic,sigma=self.fixture();result=euclidean_reference(surface,motion,geodesic,sigma)
        distances=torch.cdist(surface['canonical_node_xyz'],surface['canonical_node_xyz']);distances.fill_diagonal_(float('inf'))
        expected=set()
        for i,neighbors in enumerate(torch.argsort(distances,stable=True)[:,:6].tolist()):
            expected.update(tuple(sorted((i,j))) for j in neighbors)
        self.assertEqual(set(map(tuple,result['surface_edge_indices'].T.tolist())),expected)
        e=result['surface_edge_indices'];delta=motion[:,e[0]]-motion[:,e[1]]
        expected_gate=torch.exp(-delta.square().sum(-1).mean(0)/(2*2.7**2))
        self.assertTrue(torch.allclose(result['trajectory_similarity'],expected_gate))
        expected_spatial=torch.exp(-geodesic[e[0],e[1]].square()/(2*sigma**2)).clamp_min(1e-12)
        self.assertTrue(torch.equal(result['surface_edge_weights'],expected_spatial))

    def test_zero_decoder_and_seed_unchanged_when_connectivity_changes(self):
        surface,motion,geodesic,sigma=self.fixture();result=euclidean_reference(surface,motion,geodesic,sigma)
        e=result['surface_edge_indices'];w=result['trajectory_gated_edge_weights'];nodes=surface['canonical_node_xyz']
        rng=torch.get_rng_state().clone()
        a=GraphMotionResidual(nodes,e,w,seed=4242,mode='g3')
        b=GraphMotionResidual(nodes,e,torch.ones_like(w),seed=4242,mode='g3')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a.decoder.state_dict()[k],b.decoder.state_dict()[k]) for k in a.decoder.state_dict()))
        self.assertTrue(torch.equal(a(motion[0])[0],motion[0]));self.assertTrue(torch.equal(b(motion[0])[0],motion[0]))


if __name__=='__main__':unittest.main()
