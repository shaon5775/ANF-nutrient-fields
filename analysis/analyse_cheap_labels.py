# Summarises 05_cheap_partial_labels.py output. Usage: python analyse_cheap_labels.py ea_results.json ea_preds.npz
import json, os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(sys.argv[1])); P = np.load(sys.argv[2]); y = P['y']
KS = ('R_pool', 'R_sum', 'R_soft', 'M'); ARMS = ('A0', 'AE', 'AM', 'AEM'); DR = (200, 201, 202)
def pm(p, i, idx): return 100 * np.abs(p[idx, i] - y[idx, i]).sum() / y[idx, i].sum()
def mac(p, idx=slice(None)): return float(np.mean([pm(p, i, idx) for i in (2, 3, 4)]))
def boot(fa, fb, n=2000):  # CI of mac(b) - mac(a) on draw-averaged predictions
    rng = np.random.default_rng(0); N = len(y); o = [mac(fb, j) - mac(fa, j) for j in (rng.integers(0, N, N) for _ in range(n))]
    return [round(float(np.percentile(o, 2.5)), 3), round(float(np.percentile(o, 97.5)), 3)]
out = {'table': {}, 'hyp': {}}
for f in ('0.1', '0.05'):
    T = {}
    for arm in ARMS:
        for k in KS:
            g = [R[f'f{f}_{arm}_{k}_d{d}'] for d in DR]
            T[(arm, k)] = dict(macro=float(np.mean([v['macro'] for v in g])), pmae=np.mean([v['pmae'] for v in g], 0).round(2).tolist(),
                               factors=(np.mean([v['factors'] for v in g], 0).round(3).tolist() if g[0]['factors'] else None))
    for arm in ARMS:
        for k in KS:
            c = T[(arm, k)]; c['rel_gain_vs_A0_%'] = round(100 * (T[('A0', k)]['macro'] - c['macro']) / T[('A0', k)]['macro'], 2)
            out['table'][f'f{f}_{arm}_{k}'] = c
    for arm in ('AE', 'AM', 'AEM'):
        avg = lambda k, a: np.mean([P[f'f{f}_{a}_{k}_d{d}'] for d in DR], 0)
        best = min(('R_pool', 'R_sum', 'R_soft'), key=lambda k: T[(arm, k)]['macro'])
        a_ok = T[(arm, 'M')]['rel_gain_vs_A0_%'] >= 5.0; b_ok = all(T[(arm, 'M')]['macro'] < T[(arm, k)]['macro'] for k in ('R_pool', 'R_sum', 'R_soft'))
        out['hyp'][f'f{f}_{arm}'] = dict(anf_gain_pct=T[(arm, 'M')]['rel_gain_vs_A0_%'], a_gain_ge5=a_ok, b_beats_all=b_ok, PASS=bool(a_ok and b_ok), best_baseline=best,
                                         ci_anfA0_minus_anfArm=boot(avg('M', arm), avg('M', 'A0')), ci_best_minus_anf=boot(avg('M', arm), avg(best, arm)))
out['H1_PASS'] = out['hyp']['f0.1_AE']['PASS']; out['H2_PASS'] = out['hyp']['f0.1_AEM']['PASS']; out['TEST_PASS'] = bool(out['H1_PASS'] or out['H2_PASS'])
json.dump(out, open(os.path.join(H, '..', 'results', 'cheap_labels_summary.json'), 'w'), indent=1)
for f in ('0.1', '0.05'):
    print(f'\n== f={f}  macro PMAE (fat/carb/prot mean) | gain vs A0 % | kcal | mass')
    for arm in ARMS:
        print(arm, '  '.join(f"{k}:{out['table'][f'f{f}_{arm}_{k}']['macro']:.2f}({out['table'][f'f{f}_{arm}_{k}']['rel_gain_vs_A0_%']:+.1f}) k{out['table'][f'f{f}_{arm}_{k}']['pmae'][0]:.1f} m{out['table'][f'f{f}_{arm}_{k}']['pmae'][1]:.1f}" for k in KS))
for k, v in out['hyp'].items(): print(k, v)
print('H1', out['H1_PASS'], 'H2', out['H2_PASS'], 'TEST', 'PASS' if out['TEST_PASS'] else 'FAIL')
