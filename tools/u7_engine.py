"""U7 training/inference engine; native Stage11 equations and checkpoint format."""
import copy,gc,json,os,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import torch,numpy as np
from tools import run_stage8 as base
from gaussian_renderer import render
from utils.exact_split_branchpoint import model_digest,digest,RAW_NAMES
from utils.stage9 import reference_offset,cosine
from utils.stage11 import compensated_knn
from utils.stage10 import depth_loss
from utils.adaptive_residual_nodes import install,register_optimizer,ResidualNodes
from utils.adaptive_coverage_lock import CoverageResidualNodes
write,sha=base.write,base.sha
OUT=ROOT/'output'
DATA=None
ASSET=None
SOURCES={}


def configure(key):
    assert key in ('U7','R') and DATA is not None
    base.configure('transverse')
    base.full.DATA=base.legacy.DATA=base.shared.DATA=DATA
    return 'rectum' if key=='R' else 'transverse'


def gate(key):
    p=ASSET/'scale_gate.json';q=json.loads(p.read_text())
    return dict(ratio=q['ratio'],enabled=.5<=q['ratio']<=2.,source=str(p),source_sha=sha(p))


def apply_boundary(m,key,out):
    path=ASSET/'boundary/selection.pth';q=torch.load(path,map_location='cuda',weights_only=True)
    m.stage11_boundary=q['mask'];m.stage11_floor=q['floor'];m.stage11_gate=gate(key)
    assert m.stage11_gate['enabled']
    with torch.no_grad():
        previous=m._opacity.clone();m._opacity.copy_(torch.maximum(m._opacity,q['fine0_opacity']))
    write(out/'opacity_restore.json',dict(selection_sha=sha(path),restored_fraction=float((m._opacity>previous).float().mean()),boundary_fraction=float(q['mask'].float().mean()),Adam_policy='preserve existing opacity moments; project logits after each Adam',metric_enabled=True,lambda_raw=.30))


def opt_setup(m, c):
    m.training_setup(c)
    m.add_graph_residual_optimizer_group()
    if m.skinning is not None:
        from utils.ggas import add_optimizer
        add_optimizer(m)
    if hasattr(m, 'adaptive_residual'):
        register_optimizer(m)

def migrate(m, c, old):
    opt_setup(m, c)
    fresh = m.optimizer.state_dict()
    oldgroups = {g['name']: g for g in old['param_groups']}
    for g in fresh['param_groups']:
        og = oldgroups[g['name']]
        n = len(og['params'])
        assert len(g['params']) >= n
        for (previous, current) in zip(og['params'], g['params']):
            if previous in old['state']:
                fresh['state'][current] = copy.deepcopy(old['state'][previous])
        for (k, v) in og.items():
            if k != 'params':
                g[k] = v
    m.optimizer.load_state_dict(fresh)
    restored = m.optimizer.state_dict()
    for g in restored['param_groups']:
        og = oldgroups[g['name']]
        for (previous, current) in zip(og['params'], g['params']):
            assert digest(old['state'].get(previous, {})) == digest(restored['state'].get(current, {}))
    return restored

def load(path, i, key):
    configure(key)
    (m, s, p, b, c, st) = base.full.load(path, i)
    if 'adaptive_residual' in st:
        payload = st['adaptive_residual']
        cls = CoverageResidualNodes if payload['format'] == CoverageResidualNodes.FORMAT else ResidualNodes
        install(m, cls.from_payload(payload, 'cuda'))
    if 'stage11' in st:
        m.stage11_ref = st['stage11']['reference'].cuda()
        m.stage11_key = key
        m.stage11_gate = st['stage11']['gate']
        m.stage11_boundary = st['stage11']['boundary'].cuda()
        m.stage11_floor = st['stage11']['floor'].cuda()
    return (m, s, p, b, c, st)

def save(path, m, s, c, i):
    base.full.save(path, m, s, c, i)
    f = path / f'fine_state_{i}.pth'
    st = torch.load(f, map_location='cpu', weights_only=False)
    st['stage11'] = {'reference': m.stage11_ref.detach().cpu(), 'key': m.stage11_key, 'gate': m.stage11_gate}
    if m.stage11_key != 'P':
        st['stage11'].update(boundary=m.stage11_boundary.cpu(), floor=m.stage11_floor.cpu())
    if hasattr(m, 'adaptive_residual'):
        st['adaptive_residual'] = m.adaptive_residual.payload()
    torch.save(st, f)
    q = json.loads((path / f'checkpoint_{i}.json').read_text())
    q['sha256'] = sha(f)
    write(path / f'checkpoint_{i}.json', q)

def reload_probe(m, cam, p, b):
    residual = m.graph_motion_residual
    old = None if residual is None else residual.deterministic_aggregation
    try:
        if residual is not None:
            residual.deterministic_aggregation = True
        with torch.no_grad():
            q = render(cam, m, p, b)
            return {k: q[k].detach().cpu().clone() for k in ['render', 'depth', 'alpha', 'transformed_points']}
    finally:
        if residual is not None:
            residual.deterministic_aggregation = old

def prepare_model(key, out):
    scratch = OUT / 'reference_snapshots' / key
    if not scratch.exists():
        scratch.mkdir(parents=True)
        for name in ['cfg_args', 'fine_state_10000.pth', 'checkpoint_10000.json']:
            shutil.copy2(SOURCES[key] / name, scratch / name)
        shutil.copytree(SOURCES[key] / 'point_cloud/iteration_10000', scratch / 'point_cloud/iteration_10000')
    (m, s, p, b, c, st) = load(scratch, 10000, key)
    before = {k: getattr(m, k).detach().clone() for k in RAW_NAMES}
    cam = s.getTrainCameras()[0]
    original = reload_probe(m, cam, p, b)
    oldapp = copy.deepcopy(m.appearance_model.state_dict())
    from utils.stage7 import R6AdditiveMicro
    c.appearance_mode = m.args.appearance_mode = 'r6_additive_micro'
    c.r6_additive_cap = m.args.r6_additive_cap = 0.25
    with base.preserve_rng():
        app = R6AdditiveMicro(c, 2000).cuda()
    result = app.load_state_dict(oldapp, strict=False)
    assert set(result.missing_keys) == {'additive_head.weight', 'additive_head.bias', 'micro_head.weight', 'micro_head.bias', 'micro_iteration'} and (not result.unexpected_keys)
    m.appearance_model = app
    app.micro_iteration.fill_(10000)
    m.stage11_key = key
    m.stage11_ref = reference_offset(m._xyz, m.closest_point_indices)
    new = reload_probe(m, cam, p, b)
    errors = {k: float((new[k] - original[k]).abs().max()) for k in ['render', 'depth', 'alpha', 'transformed_points']}
    assert max(errors.values()) < 2e-06 if key == 'R' else all((errors[k] == 0 for k in ['render', 'depth', 'alpha'])) and errors['transformed_points'] < 2e-06, errors
    for (k, v) in before.items():
        assert torch.equal(v, getattr(m, k))
    apply_boundary(m, key, out)
    st = copy.deepcopy(st)
    st['optimizer'] = migrate(m, c, st['optimizer'])
    write(out / 'migration.json', dict(pass_=True, source=str(SOURCES[key]), source_sha256=sha(SOURCES[key] / 'fine_state_10000.pth'), render_zero_head_errors=errors, old_Adam_exact=True, added_parameters=390, reference='canonical10k detached'))
    c.model_path = str(out)
    s.model_path = str(out)
    for (k, v) in dict(stage3_dynamics=False, loss_rgb_mode='l1', tv_weight=0.0005, col_smooth_weight=0.0001, position_lr_max_steps=30000, position_lr_final=1.6e-06, depth_weight=0.25, knn_weight=0.01).items():
        setattr(c, k, v)
        setattr(m.args, k, v)
    m.gats_joint = True
    m.fine_resume_state = st
    return (m, s, p, b, c, st)

class Hooks:

    def __init__(self, out, m, s, p, b, c, st, key, end):
        (self.out, self.m, self.s, self.p, self.b, self.c) = (out, m, s, p, b, c)
        (self.key, self.end, self.state) = (key, end, st)
        self.started = time.monotonic()
        self.overhead = 0
        self.app = digest(m.appearance_graph.state_dict())
        self.i = 10000
        self.base_lrs = {g['name']: g['lr'] for g in st['optimizer']['param_groups']}

    def setup(self, m):
        opt_setup(m, self.c)

    def begin(self, i):
        self.i = i
        self.m.appearance_model.micro_iteration.fill_(i)
        if hasattr(self.m, 'adaptive_residual'):
            self.m.adaptive_residual.iteration.fill_(i)
        self.c.knn_weight = 0.01
        self.c.depth_weight = 0.25
        if self.key == 'P' and i == 1:
            for (k, v) in self.state['coarse_raw_gradients'].items():
                getattr(self.m, k).grad = None if v is None else v.cuda().clone()
        if self.key == 'P' and i == 5001:
            import random
            random.seed(0)
            base.base.update(self.m, self.c, ggas_variant='J_NO_MOTION_SELECTION', ggas_study_version=2, ggas_candidates=str(self.out / 'candidates.pth'), ggas_lr=0.001, ggas_lambda=0.01, ggas_ramp_end=6000, geometry_study_refresh='none')
            from utils.ggas import initialize
            with base.preserve_rng():
                initialize(self.m, self.out)
            old = ROOT / 'output/experiments/stage8_fixedknn_modality/run_20261023/sigmoid2/runs/P/J/point_cloud/iteration_10000/skinning.pth'
            payload = torch.load(old, map_location='cuda', weights_only=True)['state_dict']
            for k in ['indices', 'initial', 'target']:
                assert torch.equal(self.m.skinning.state_dict()[k], payload[k]), k
            write(self.out / 'P_topology_match.json', dict(pass_=True, indices_initial_target_bitwise=True, source=str(old), source_sha=sha(old), G3='exact saved5k loaded; no rebuild'))

    def schedule(self):
        constant = {'f_dc': self.c.feature_lr, 'deformation_table': self.c.feature_lr, 'opacity': self.c.opacity_lr, 'scaling': self.c.scaling_lr, 'rotation': self.c.rotation_lr, 'skinning': self.m.args.ggas_lr}
        for g in self.m.optimizer.param_groups:
            lr = constant.get(g['name'], g['lr'])
            if g['name'] == 'adaptive_residual':
                lr = self.m.deformation_scheduler_args(self.i)
            g['lr'] = lr * cosine(self.i)

    def depth(self, pred, gt, valid, weight):
        raw_weight = 0.3 if self.m.stage11_gate['enabled'] else 0.0
        (total, norm, raw) = depth_loss(pred, gt, valid, weight, raw_weight)
        self.depth_stats = dict(metric_enabled=bool(raw_weight), lambda_raw=raw_weight, normalized_weighted=float(norm.detach()), raw_weighted=float(raw.detach()))
        return total

    def render(self, cam, m, *a, **kw):
        assert self.key == 'P' or cam.source_split == 'TRAIN'
        with (self.out / 'camera_sequence.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=self.i, name=cam.image_name)) + '\n')
        return render(cam, m, *a, **kw)

    def rgb(self, x, y):
        from utils.loss_utils import ssim
        return 0.8 * ((x - y).square() + 1e-06).sqrt().mean() + 0.15 * (1 - ssim(x, y)) / 2 + 3 * (x - y).square().mean()

    def backward(self, m, i, loss, rgb, depth, knn):
        assert torch.isfinite(loss)
        if i % 100 == 0:
            norms = {k: float(v.grad.norm()) if v.grad is not None else 0 for (k, v) in m.appearance_model.named_parameters()}
            assert all((np.isfinite(v) for v in norms.values()))
            if i == 10100:
                assert norms['micro_head.weight'] > 0 and norms['additive_head.weight'] > 0
                assert torch.count_nonzero(m._micro_offset[m.stage11_boundary]) == 0
            with (self.out / 'training_metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(iteration=i, loss=float(loss), RGB_L1=float(rgb), depth=float(depth), knn=float(knn), depth_components=self.depth_stats, gradients=norms, lr={g['name']: g['lr'] for g in m.optimizer.param_groups})) + '\n')

    def after(self, m, s, cameras, i):
        with torch.no_grad():
            m._opacity[m.stage11_boundary] = torch.maximum(m._opacity[m.stage11_boundary], m.stage11_floor[m.stage11_boundary])
        if i % 100 == 0:
            assert torch.all(m._opacity[m.stage11_boundary] >= m.stage11_floor[m.stage11_boundary])
        if self.key == 'P' and i == 2000:
            with base.preserve_rng():
                save(self.out / 'setup/warmup', m, s, self.c, i)
                helper = base.graph_helper(self.out)
                helper.reference()
                base.base.update(m, self.c, graph_forward_mode='g3', graph_forward_reference=str(helper.REF))
                m.initialize_graph_forward(str(helper.REF))
                m.add_graph_residual_optimizer_group()
        if self.key == 'P' and i == 5000:
            with base.preserve_rng():
                save(self.out / 'common_5000', m, s, self.c, i)
                base.candidates(self.out)
        if i % 100 == 0:
            write(self.out / 'status.json', dict(state='RUNNING', iteration=i))
        if i in [10200, 15000, 20000, 25000, 30000] or i == self.end:
            assert digest(m.appearance_graph.state_dict()) == self.app
            start = time.monotonic()
            with base.preserve_rng():
                save(self.out, m, s, self.c, i)
            self.overhead += time.monotonic() - start
        if i == self.end:
            write(self.out / 'runtime.json', dict(training_seconds=time.monotonic() - self.started - self.overhead, peak_VRAM_MiB=torch.cuda.max_memory_allocated() / 2 ** 20, **base.parameters(m)))
            write(self.out / 'status.json', dict(state='PILOT_COMPLETE' if i < 30000 else 'TRAINED', iteration=i))

def instrument(h):
    import train
    base.native_function(h)
    s = h.patched_source
    start = s.index('        from utils.photometric_unlock import rgb_loss')
    end = s.index('        ### DEPTH LOSS', start)
    s = s[:start] + '        loss_rgb = gats_hooks.rgb(images_concealing,gt_images)\n        loss += loss_rgb\n\n' + s[end:]
    start = s.index('        if valid_depth_mask.any():')
    end = s.index('        loss +=depth_loss', start)
    s = s[:start] + '        depth_loss = gats_hooks.depth(rendered_depths,gt_depths,valid_depth_mask,opt.depth_weight)\n' + s[end:]
    needle = '            loss_clusters = opt.knn_weight * per_point_sq_dist.mean()'
    assert s.count(needle) == 1
    s = s.replace(needle, '            loss_clusters = unified_knn(transformed_means,idx,gaussians.stage11_ref,iteration,gaussians.stage11_boundary)')
    needle = '        update_training(gaussians, opt, iteration, stage)'
    assert s.count(needle) == 1
    s = s.replace(needle, needle + '\n        gats_hooks.schedule()')
    needle = '                if not (stage == "fine" and getattr(opt, "stage3_dynamics", False)) and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):'
    assert s.count(needle) == 1
    s = s.replace(needle, '                if iteration in (3000,6000,9000):')
    ns = dict(train.__dict__, gats_hooks=h, render=h.render, unified_depth=depth_loss, unified_knn=compensated_knn)
    exec(compile(s, __file__ + ':native', 'exec'), ns)
    (h.out / 'native_training.py').write_text(s)
    return ns['scene_reconstruction']
