# Summarises 04_factor_starts.py output. Usage: python analyse_factor_starts.py dz_results.json
import json, os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(sys.argv[1]))
ATW = np.array([9., 4., 4.]); ST = ['S0', 'S1', 'S2', 'S3', 'S4']
def group(ds, st, rn): return [v for k, v in R.items() if k.startswith(f'{ds}_{st}_{rn}_')]
out = {'cells': {}}; checks = []
for ds in ('N5k', 'NVR'):
    for rn in ('R1', 'R10'):
        for st in ST:
            g = group(ds, st, rn)
            if not g: continue
            fin = np.mean([v['final'] for v in g], 0); start = np.array(g[0]['start']); kcal = float(np.mean([v['pmae'][0] for v in g]))
            ratio = fin / ATW; inband = bool(np.all(np.abs(ratio - 1) <= 0.15))
            toward = [bool(abs(f - a) < abs(s - a) or abs(s - a) < 1e-9) for f, s, a in zip(fin, start, ATW)]
            out['cells'][f'{ds}_{rn}_{st}'] = dict(n=len(g), start=start.tolist(), final=np.round(fin, 3).tolist(), ratio=np.round(ratio, 3).tolist(),
                                                   in_band=inband, moved_toward=toward, order_ok=bool(fin[0] > fin[1] and fin[0] > fin[2]), kcal_pmae=round(kcal, 2))
            if rn == 'R10' and st != 'S0': checks.append(inband)
for ds in ('N5k', 'NVR'):
    for rn in ('R1', 'R10'):
        k0 = out['cells'].get(f'{ds}_{rn}_S0')
        for st in ST[1:]:
            c = out['cells'].get(f'{ds}_{rn}_{st}')
            if c and k0: c['kcal_minus_S0'] = round(c['kcal_pmae'] - k0['kcal_pmae'], 2); c['dist_to_S0'] = round(float(np.linalg.norm(np.array(c['final']) - np.array(k0['final']))), 3)
out['H1_checks_passed'] = f'{sum(checks)} / {len(checks)}'; out['H1_PASS'] = bool(len(checks) == 8 and all(checks))
json.dump(out, open(os.path.join(H, '..', 'results', 'factor_starts_summary.json'), 'w'), indent=1)
for k, v in out['cells'].items(): print(k, v)
print('H1', out['H1_checks_passed'], 'PASS' if out['H1_PASS'] else 'FAIL')
