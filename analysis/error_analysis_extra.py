# Descriptive error analysis by dish mass and ingredient count (no training; uses the saved full-study predictions).
# Output: results/FA/tables/T8c_error_extra.json + printout. Ingredient names from public Nutrition5k metadata.
import csv, io, json, os, numpy as np, requests, certifi
H = os.path.dirname(os.path.abspath(__file__)); FA = os.path.join(H, '..', 'results', 'FA')
R = list(csv.DictReader(open(os.path.join(FA, 'preds_n5k_base.csv'), encoding='utf-8')))
M = {'Pooled': 'R_pool', 'Summed maps': 'R_sum', 'Maps+cons.': 'R_soft', 'ANF': 'M'}
y = np.array([float(r['true_kcal']) for r in R]); mass = np.array([float(r['true_mass']) for r in R])
P = {k: np.array([float(r[v + '_kcal']) for r in R]) for k, v in M.items()}
t1, t2 = np.quantile(mass, [1 / 3, 2 / 3]); grp = np.where(mass < t1, 'light', np.where(mass < t2, 'medium', 'heavy'))
out = {'mass_cutoffs_g': [round(float(t1), 1), round(float(t2), 1)], 'groups': {}}
for g in ['light', 'medium', 'heavy']:
    m = grp == g; d = {'n': int(m.sum()), 'mean_true_kcal': round(float(y[m].mean()), 1)}
    for k, p in P.items():
        d[k] = dict(PMAE=round(float(100 * np.abs(p[m] - y[m]).sum() / y[m].sum()), 1), MAE_kcal=round(float(np.abs(p[m] - y[m]).mean()), 1),
                    bias_pct=round(float(100 * (p[m] - y[m]).sum() / y[m].sum()), 1))
    out['groups'][g] = d
# whole test set: share of dishes where ANF has the smallest absolute error, and small-dish share of the worst errors
ae = {k: np.abs(p - y) for k, p in P.items()}; best = np.argmin(np.vstack(list(ae.values())), 0)
out['share_ANF_best'] = round(float(np.mean(best == list(M).index('ANF'))), 3)
rel = np.abs(P['ANF'] - y) / y; worst = np.argsort(-rel)[:25]
out['worst25'] = dict(median_true_kcal=round(float(np.median(y[worst])), 1), median_mass=round(float(np.median(mass[worst])), 1),
                      share_single_ingredient=round(float(np.mean([int(R[i]['n_ingredients']) == 1 for i in worst])), 2),
                      share_over=round(float(np.mean(P['ANF'][worst] > y[worst])), 2))
# ingredient names of the 6 worst dishes
S = requests.Session(); S.verify = certifi.where(); names = {}
for f in ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']:
    txt = S.get('https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/' + f, timeout=120).text
    for row in csv.reader(io.StringIO(txt)):
        if row and row[0].startswith('dish_'):
            names[row[0]] = [row[6 + 7 * j + 1] for j in range((len(row) - 6) // 7)]
out['worst6'] = [dict(dish=R[i]['dish'], true_kcal=float(R[i]['true_kcal']), ANF_kcal=round(float(P['ANF'][i]), 1), mass=float(R[i]['true_mass']),
                      ingredients=names.get(R[i]['dish'], [])[:4]) for i in np.argsort(-rel)[:6]]
json.dump(out, open(os.path.join(FA, 'tables', 'T8c_error_extra.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
