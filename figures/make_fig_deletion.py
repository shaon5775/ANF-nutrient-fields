# Deletion-test figure (supplementary Fig. S1) from the numbers printed by experiments/03_map_checks.py (mean of three seeds).
import os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
k = [8, 16, 32, 64]
D = {'ANF, top-energy cells': ([38.7, 61.3, 83.0, 96.4], '#c44e52', '-', 'o'),
     'Summed maps, top-energy cells': ([30.7, 49.8, 71.6, 89.3], '#4c72b0', '-', 's'),
     'Maps + consistency loss, top-energy cells': ([30.4, 49.7, 71.6, 89.5], '#dd8452', '--', '^'),
     'ANF, random cells': ([3.2, 6.3, 12.4, 25.4], '#9e9e9e', ':', 'D')}
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures_out'), exist_ok=True)
fig, ax = plt.subplots(figsize=(3.6, 3.3))
for n, (y, c, ls, mk) in D.items(): ax.plot(k, y, ls, color=c, marker=mk, ms=4, lw=1.4, label=n)
ax.set_xlabel('cells replaced (of 256)'); ax.set_ylabel('fall in predicted energy (%)'); ax.set_xticks(k); ax.set_ylim(0, 100)
ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=1)
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures_out', 'F_deletion.' + e), dpi=300, bbox_inches='tight')
print('ok')
