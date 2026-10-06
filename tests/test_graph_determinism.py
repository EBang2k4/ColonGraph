import unittest
import torch
from utils.graph_forward import GraphMotionResidual, graph_index_add_


class GraphDeterminismTests(unittest.TestCase):
    def module(self, device="cpu", enabled=False):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]], device=device)
        edges = torch.tensor([[0, 0, 1], [1, 2, 2]], device=device)
        weights = torch.tensor([1., 3., 2.], device=device)
        return GraphMotionResidual(xyz, edges, weights, seed=4, mode="g3",
                                   deterministic_aggregation=enabled).to(device)

    def test_default_legacy_and_roundtrip(self):
        legacy = self.module()
        safe = self.module(enabled=True)
        base = torch.tensor([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]], requires_grad=True)
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, safe.state_dict()[key]))
        self.assertTrue(torch.equal(legacy.aggregate(base), safe.aggregate(base)))
        safe(base)[0].square().mean().backward()
        self.assertTrue(torch.isfinite(base.grad).all())
        with torch.no_grad():
            safe.decoder[-1].bias.fill_(.25)
        restored = GraphMotionResidual.from_state_dict_payload(safe.state_dict_payload(), "cpu")
        self.assertTrue(restored.deterministic_aggregation)
        self.assertTrue(torch.equal(safe(base)[0], restored(base)[0]))
        payload = legacy.state_dict_payload()
        payload.pop("deterministic_aggregation")
        old = GraphMotionResidual.from_state_dict_payload(payload, "cpu")
        self.assertFalse(old.deterministic_aggregation)
        self.assertTrue(torch.equal(legacy(base)[0], old(base)[0]))

    def test_scope_restored_including_warn_only_and_failure(self):
        original = torch.are_deterministic_algorithms_enabled()
        original_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for enabled, warn in ((False, False), (True, True), (True, False)):
                torch.use_deterministic_algorithms(enabled, warn_only=warn)
                self.module(enabled=True).aggregate(torch.ones(3, 3))
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn)
                with self.assertRaises((RuntimeError, IndexError)):
                    graph_index_add_(torch.zeros(1), 0, torch.tensor([8]), torch.ones(1), True)
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn)
        finally:
            torch.use_deterministic_algorithms(original, warn_only=original_warn)

    def test_no_rng_consumption(self):
        cpu = torch.get_rng_state().clone()
        model = self.module(enabled=True)
        model.aggregate(torch.ones(3, 3))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_same_input_repeat_and_reload(self):
        module = self.module("cuda", True)
        base = torch.tensor([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]], device="cuda")
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone()
        original = module.aggregate(base)
        for _ in range(10):
            self.assertTrue(torch.equal(original, module.aggregate(base)))
        restored = GraphMotionResidual.from_state_dict_payload(module.state_dict_payload(), "cuda")
        self.assertTrue(torch.equal(original, restored.aggregate(base)))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
