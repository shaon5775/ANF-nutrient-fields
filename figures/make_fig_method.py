# Figure 1 (method) built around a real Nutrition5k test dish (test index 150 = dish_1565123881).
# Inputs: results/FA (thumbs_n5k_test.npz from 01_full_study.py). Output: figures_out/F_method.pdf (+ .png).
import os, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
H = os.path.dirname(os.path.abspath(__file__)); FA = os.path.join(H, '..', 'results', 'FA'); OUT = os.path.join(H, '..', 'figures_out'); os.makedirs(OUT, exist_ok=True)
J = 150
img = np.load(os.path.join(FA, 'thumbs_n5k_test.npz'))['rgb'][J]
mp = np.load(os.path.join(FA, 'maps_n5k_base.npz'))['M'][J].astype(np.float32)
mass, energy = mp[:, 0].reshape(16, 16), mp[:, 1].reshape(16, 16)
dens = np.where(mass > 0.05 * mass.max(), energy / np.maximum(mass, 1e-6), np.nan)
r = list(csv.DictReader(open(os.path.join(FA, 'preds_n5k_base.csv'))))[J]
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 11, 'mathtext.fontset': 'stix'})
fig = plt.figure(figsize=(9.0, 3.4))
def panel(x, y, w, h): return fig.add_axes([x, y, w, h])
def arrow(x0, y0, x1, y1):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=14, lw=1.8, color='black'))
# 1) photo with grid
a = panel(0.005, 0.20, 0.17, 0.58); a.imshow(img); a.set_xticks([]); a.set_yticks([])
for k in range(1, 16): a.axhline(k * 16, color='white', lw=0.3, alpha=0.7); a.axvline(k * 16, color='white', lw=0.3, alpha=0.7)
a.set_title('Photo, 16×16 cells', fontsize=12, pad=4)
arrow(0.18, 0.49, 0.225, 0.49)
# 2) frozen backbone
b = panel(0.225, 0.22, 0.10, 0.54); b.axis('off'); b.set_xlim(0, 1); b.set_ylim(0, 1)
b.fill([0.05, 0.95, 0.95, 0.05], [0.0, 0.22, 0.78, 1.0], color='#dbe6f4', ec='black', lw=1.4)
b.text(0.5, 0.5, 'DINOv2\n(frozen)', ha='center', va='center', fontsize=12)
b.text(0.5, -0.08, 'one feature\nper cell', ha='center', va='top', fontsize=10.5, color='0.25')
arrow(0.33, 0.49, 0.37, 0.49)
# 3) small head
c = panel(0.37, 0.30, 0.075, 0.38); c.axis('off'); c.set_xlim(0, 1); c.set_ylim(0, 1)
c.add_patch(Rectangle((0.03, 0.03), 0.94, 0.94, fc='#f6dcdc', ec='black', lw=1.4))
c.text(0.5, 0.5, 'Small\nhead\n(17k)', ha='center', va='center', fontsize=11.5)
arrow(0.448, 0.62, 0.50, 0.70); arrow(0.448, 0.36, 0.50, 0.28)
# 4) per-cell outputs
d1 = panel(0.50, 0.54, 0.13, 0.36); d1.imshow(img, alpha=0.45); d1.imshow(np.kron(mass, np.ones((16, 16))), cmap='viridis', alpha=0.65); d1.set_xticks([]); d1.set_yticks([])
d1.set_title('Grams per cell', fontsize=11.5, pad=3)
d2 = panel(0.50, 0.07, 0.13, 0.36); d2.imshow(img, alpha=0.45); d2.imshow(np.kron(dens, np.ones((16, 16))), cmap='magma', alpha=0.65); d2.set_xticks([]); d2.set_yticks([])
d2.set_title('Make-up → kcal per gram', fontsize=11.5, pad=3)
pc = panel(0.635, 0.12, 0.075, 0.21); pc.pie([0.10, 0.12, 0.10, 0.68], colors=['#e07b5f', '#f2c14e', '#7fb069', '#d9d9d9'], startangle=90, wedgeprops=dict(lw=0.9, ec='black'))
pc.set_title('One cell', fontsize=10.5, pad=1)
fig.text(0.672, 0.0, 'fat / carb /\nprotein / other', fontsize=9.5, ha='center', va='bottom', color='0.25')
arrow(0.64, 0.72, 0.735, 0.62); arrow(0.712, 0.24, 0.735, 0.36)
# 5) dish totals as an academic results table (white background)
e = panel(0.74, 0.04, 0.26, 0.90); e.axis('off'); e.set_xlim(0, 1); e.set_ylim(0, 1)
e.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fc='white', ec='black', lw=1.4))
e.text(0.5, 0.92, 'Dish totals = sum over cells', ha='center', va='center', fontsize=12, fontweight='bold')
e.plot([0.05, 0.95], [0.84, 0.84], color='black', lw=1.0)
e.text(0.06, 0.78, 'Output', fontsize=11, fontweight='bold', va='center'); e.text(0.60, 0.78, 'ANF', fontsize=11, fontweight='bold', ha='center', va='center'); e.text(0.82, 0.78, 'Weighed', fontsize=11, fontweight='bold', ha='center', va='center')
e.plot([0.05, 0.95], [0.73, 0.73], color='black', lw=0.6)
rows = [('Mass (g)', 'mass'), ('Fat (g)', 'fat'), ('Carbohydrate (g)', 'carb'), ('Protein (g)', 'protein'), ('Energy (kcal)', 'kcal')]
for i, (nm, k) in enumerate(rows):
    y = 0.665 - i * 0.085
    e.text(0.06, y, nm, fontsize=11, va='center'); e.text(0.60, y, f'{float(r["M_" + k]):.0f}', fontsize=11, ha='center', va='center'); e.text(0.82, y, f'{float(r["true_" + k]):.0f}', fontsize=11, ha='center', va='center')
e.plot([0.05, 0.95], [0.27, 0.27], color='black', lw=1.0)
e.text(0.5, 0.17, r'$E = a_F\,F + a_C\,C + a_P\,P$', ha='center', va='center', fontsize=13)
e.text(0.5, 0.07, 'learned factors $a_F, a_C, a_P \\geq 0$', ha='center', va='center', fontsize=10, color='0.25')
fig.savefig(os.path.join(OUT, 'F_method.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'F_method.png'), dpi=300, bbox_inches='tight')
print('saved')
