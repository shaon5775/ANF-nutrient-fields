# Paper figures (learning curves, factors, energy maps) from the full-study outputs in results/FA (run 01_full_study.py first for the map thumbnails).
# Inputs: results/FA (results.json, factors.json, maps, preds; thumbs_n5k_test.npz from 01_full_study.py). Outputs: figures_out/*.pdf/png
import os, csv, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); FA = os.path.join(H, '..', 'results', 'FA'); OUT = os.path.join(H, '..', 'figures_out'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False})
KS = ['R_pool', 'R_sum', 'R_soft', 'M']; COL = {'R_pool': '#8c8c8c', 'R_sum': '#4c72b0', 'R_soft': '#dd8452', 'M': '#c44e52'}
NAME = {'R_pool': 'Pooled regression', 'R_sum': 'Summed maps', 'R_soft': 'Maps + consistency loss', 'M': 'ANF (ours)'}
MK = {'R_pool': 'o', 'R_sum': 's', 'R_soft': '^', 'M': 'D'}
RES = json.load(open(os.path.join(FA, 'results.json'))); FACT = json.load(open(os.path.join(FA, 'factors.json')))
def save(fig, name): fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
# ---- Learning curve (full text width) ----
LC = RES['E2']; fr = sorted(LC, key=float); x = [100 * float(f) for f in fr]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
for k in KS:
    for a_, q in zip(ax, ('kcal', 'macro')):
        mu = [np.mean(LC[f][k][q]) for f in fr]; sd = [np.std(LC[f][k][q]) for f in fr]
        # 100%: use the five-seed single-model scores of Table 2 (same scoring as the subsets: each model scored separately)
        E1 = RES['E1_base'][k]; mu[-1] = E1['mean'][0] if q == 'kcal' else float(np.mean(E1['mean'][1:])); sd[-1] = E1['sd'][0] if q == 'kcal' else 0.0
        a_.errorbar(x, mu, yerr=sd, marker=MK[k], ms=6, capsize=3, color=COL[k], lw=2.6 if k == 'M' else 1.6, label=NAME[k])
for a_, t, ttl in zip(ax, ('Energy PMAE (%)', 'Mean PMAE (%), mass/fat/carb/protein'), ('(a) Energy', '(b) Mass and macronutrients')):
    a_.set_xscale('log'); a_.set_xticks([5, 10, 25, 50, 100]); a_.set_xticklabels(['5', '10', '25', '50', '100']); a_.set_xlabel('Training dishes used (%)'); a_.set_ylabel(t); a_.set_title(ttl); a_.grid(alpha=0.3)
ax[0].legend(frameon=False); fig.tight_layout(); save(fig, 'F_learning_curve')
# ---- Learned factors (single column) ----
groups = [('N5k\nbase', FACT['E1_base']['M']), ('N5k\nlarge', FACT['E1_large']['M']), ('NVR\nbase', FACT['E3_base']), ('NVR\nlarge', FACT['E3_large'])] + [(f'N5k\n{int(round(100 * float(f)))}%', [LC[f]['factors']]) for f in fr]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for j, (lab, lst) in enumerate(groups):
    a_ = np.array(lst)
    for c, (ref, mk, colr, nm) in enumerate(zip((9, 4, 4), ('o', 's', '^'), ('#c44e52', '#4c72b0', '#55a868'), ('fat / 9', 'carbohydrate / 4', 'protein / 4'))):
        ax.scatter(np.full(len(a_), j) + (c - 1) * 0.22, a_[:, c] / ref, marker=mk, s=40, color=colr, edgecolor='black', lw=0.4, label=nm if j == 0 else None, zorder=3)
ax.axhspan(0.85, 1.15, color='grey', alpha=0.13); ax.axhline(1, color='black', lw=1)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=10.5); ax.set_ylabel('Learned factor / Atwater value'); ax.set_ylim(0.8, 1.2)
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.16)); ax.grid(axis='y', alpha=0.3); fig.tight_layout(); save(fig, 'F_factors')
# ---- Example maps (full text width): 2 rows x 4 dishes, each dish = photo / ANF energy / summed-map energy ----
mp = np.load(os.path.join(FA, 'maps_n5k_base.npz')); th = np.load(os.path.join(FA, 'thumbs_n5k_test.npz'))['rgb']; R = list(csv.DictReader(open(os.path.join(FA, 'preds_n5k_base.csv'))))
err = np.array([abs(float(r['M_kcal']) - float(r['true_kcal'])) / float(r['true_kcal']) for r in R]); o = np.argsort(err)
pick = [int(o[0]), int(o[1]), int(o[len(o) // 2]), int(o[len(o) // 2 + 1]), int(o[-1])]
fig, ax = plt.subplots(3, len(pick), figsize=(10, 6.3))
for c, j in enumerate(pick):
    e_anf = mp['M'][j][:, 1].astype(float).reshape(16, 16); e_sum = mp['R_sum'][j][:, 0].astype(float).reshape(16, 16)
    ax[0, c].imshow(th[j]); ax[0, c].set_title(f'true {float(R[j]["true_kcal"]):.0f} / ANF {float(R[j]["M_kcal"]):.0f} kcal', fontsize=11)
    for rr, (m_, cm) in enumerate(((e_anf, 'magma'), (e_sum, 'magma')), start=1):
        ax[rr, c].imshow(th[j]); ax[rr, c].imshow(np.kron(m_, np.ones((16, 16))), cmap=cm, alpha=0.62)
    for rr in range(3): ax[rr, c].set_xticks([]); ax[rr, c].set_yticks([])
for rr, t in enumerate(('Photo', 'ANF energy', 'Summed-map\nenergy')): ax[rr, 0].set_ylabel(t, fontsize=13)
fig.tight_layout(); save(fig, 'F_maps_examples')
print('saved')
