import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('g3_surf_launcher', ROOT / 'tools/run_g3_surf_baseline.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class BaselineConfigTests(unittest.TestCase):
    def test_baseline_contract(self):
        config = json.loads(launcher.CONFIG.read_text())
        launcher.validate_config(config)
        self.assertEqual(config['ModelHiddenParams']['experiment_seed'], 0)
        self.assertEqual(config['ModelHiddenParams']['graph_forward_seed'], 4242)
        self.assertTrue(config['ModelHiddenParams']['connectivity_ablation_audit'])

    def test_relocation_does_not_change_numeric_settings(self):
        original = json.loads(launcher.CONFIG.read_text())
        actual = launcher.make_config(Path('/external/artifacts'), Path('/external/data'), Path('/new/run'))
        for group, values in original.items():
            for name, value in values.items():
                if not isinstance(value, str):
                    self.assertEqual(actual[group][name], value, (group, name))
        h = actual['ModelHiddenParams']
        self.assertTrue(h['fine_warmup_state'].startswith('/external/artifacts/output/'))
        self.assertEqual(h['fine_stage_state_save'], '/new/run/fine_state_30000.pth')

    def test_rejects_adaptive_or_new_appearance(self):
        for key, value in [('graph_forward_adaptive', True), ('surface_child_mode', 'adaptive'),
                           ('node_appearance_dim', 16), ('color_residual_scale', 0.3),
                           ('graph_forward_deterministic', True)]:
            config = json.loads(launcher.CONFIG.read_text())
            config['ModelHiddenParams'][key] = value
            with self.assertRaises(ValueError):
                launcher.validate_config(config)


if __name__ == '__main__':
    unittest.main()
