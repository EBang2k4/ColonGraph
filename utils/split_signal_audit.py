"""Preregistered split ranking candidates, independent of evaluation frames."""
import numpy as np
from scipy.stats import rankdata


def observed_median(values, observed):
    result = np.zeros(values.shape[1], dtype=np.float64)
    for i in range(values.shape[1]):
        if observed[:, i].any():
            result[i] = np.median(values[observed[:, i], i])
    return result


def percentiles(values, eligible):
    out = np.zeros_like(values, dtype=np.float64)
    out[eligible] = (rankdata(values[eligible])-1)/max(1, eligible.sum()-1)
    return out


def make_signals(discovery, eligible, rgb, depth_gradient, U, H):
    """No validation/test input is accepted by this function.

    impact is integrated alpha*T*depth-error assigned to a node, divided by
    integrated image error. Error remains error per contributing pixel.
    """
    error = discovery['depth_error']
    observed = discovery['observed'].astype(bool)
    thresholds = np.array([np.quantile(error[t, observed[t] & eligible], .75)
                           for t in range(len(error))])
    n = observed.sum(0)
    persistent = (((error > thresholds[:, None]) & observed).sum(0)/np.maximum(n, 1))
    reliability = np.minimum(n/8., 1.)
    residual = observed_median(error, observed)
    impact = observed_median(discovery['error_impact'], observed)
    # A quarter of observed frames must be difficult for persistent candidates.
    quality = eligible & (persistent >= .25)
    p = residual*persistent*reliability
    signals = {
        'rgb_gradient': (rgb, eligible),
        'depth_gradient': (depth_gradient, eligible),
        'gradient_capacity': (percentiles(depth_gradient, eligible)*percentiles(U, eligible), eligible),
        'depth_error': (residual, eligible),
        'persistent_error': (p, quality),
        'persistent_impact': (impact*persistent*reliability, quality),
        'persistent_U': (p*(.5+.5*percentiles(U, eligible)), quality),
        'persistent_H': (p*(.5+.5*percentiles(H, eligible)), quality),
    }
    return signals, dict(residual=residual, persistence=persistent,
                         reliability=reliability, impact=impact, quality=quality)


def evaluate_selection(nodes, bank, eligible):
    """Coverage is additive because normalized K4 partitions contribution."""
    nodes = np.asarray(nodes, dtype=int)
    if not len(nodes):
        return dict(count=0, captured_depth_error=0., captured_fold_error=0.,
                    captured_raw_depth_error=0., mean_support_depth_error=0.,
                    depth_enrichment=0., fold_fraction=0., highlight_fraction=0.)
    observed = bank['observed'].astype(bool)
    sums = {}
    for key in ('depth_error', 'fold_fraction', 'highlight_fraction'):
        value = (bank[key]*observed).sum(0)/np.maximum(observed.sum(0), 1)
        sums[key] = value
    evaluable = observed[:, nodes].sum(0) > 0
    selected = nodes[evaluable]
    reference = eligible & (observed.sum(0) > 0)
    return dict(count=len(nodes), observed_count=len(selected),
        captured_depth_error=float(bank['error_impact'][:, nodes].sum(1).mean()),
        captured_fold_error=float(bank['fold_error_impact'][:, nodes].sum(1).mean()),
        captured_raw_depth_error=float(bank['raw_error_impact'][:, nodes].sum(1).mean()),
        mean_support_depth_error=float(sums['depth_error'][selected].mean()) if len(selected) else 0.,
        depth_enrichment=float(sums['depth_error'][selected].mean()/max(sums['depth_error'][reference].mean(), 1e-30)) if len(selected) else 0.,
        fold_fraction=float(sums['fold_fraction'][selected].mean()) if len(selected) else 0.,
        highlight_fraction=float(sums['highlight_fraction'][selected].mean()) if len(selected) else 0.)


def choose_signal(rows):
    """Primary validation error coverage after equal maximum birth budget.

    Geometry-proxy gate: >=16 accepted at every checkpoint and positive mean
    exact-K4 WLS gain in at least two of three checkpoints. If nobody passes,
    return an explicit exploratory choice rather than claim validated capacity.
    """
    names = list(rows[0]['signals'])
    summaries = {}
    for name in names:
        runs = [r['signals'][name] for r in rows]
        gains = [r['validation_wls_gain_mean'] for r in runs]
        coverage = [r['admitted_validation']['captured_depth_error'] for r in runs]
        summaries[name] = dict(mean_validation_error_coverage=float(np.mean(coverage)),
            mean_wls_gain=float(np.mean(gains)), positive_wls_checkpoints=int(np.sum(np.array(gains) > 0)),
            minimum_birth_count=min(r['birth_count'] for r in runs))
    qualified = [k for k, v in summaries.items() if v['minimum_birth_count'] >= 16 and
                 v['positive_wls_checkpoints'] >= 2 and v['mean_wls_gain'] > 0]
    candidates = qualified or names
    winner = max(candidates, key=lambda k: summaries[k]['mean_validation_error_coverage'])
    return dict(winner=winner, qualified=qualified, capacity_proxy_qualified=winner in qualified,
                summaries=summaries,
                criterion='validation depth-error contribution captured after <=32 safe births; WLS feasibility gate before selection; test not consulted')
