# IAF ingredient-map figure (Test EB, seed 10 maps). Dishes chosen by a fixed rule: best, median and worst dish-level
# mean (H - H0) among annotated test dishes with 2-4 groups. Columns: photo | true ingredient masks | IAF group map.
import os, io, json, csv, urllib.request, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(H, '..', 'results', 'IAF'); C = os.path.join(E, 'cache'); os.makedirs(C, exist_ok=True)
z = np.load(os.path.join(E, 'iaf_maps_s0.npz')); gm = z['gm'].astype(np.float32); m = z['m'].astype(np.float32); GR = [str(g) for g in z['groups']]; gi = {g: i for i, g in enumerate(GR)}
order = json.load(open(os.path.join(E, 'test_order.json'))); tpos = {d: j for j, d in enumerate(order)}
GH = 'https://raw.githubusercontent.com/liamhand/ingredient-level-nutrition-regression/main/data'
def fetch(url, fn):
    p = os.path.join(C, fn)
    if not os.path.exists(p): open(p, 'wb').write(urllib.request.urlopen(url, timeout=60).read())
    return p
rows = list(csv.DictReader(open(fetch(f'{GH}/ingredient_nutrition.csv', 'ingr.csv'), encoding='utf-8')))
LAB = {}
for r in rows:
    if r['dish_id'] in tpos: LAB.setdefault(r['dish_id'], {})[int(r['label_value'])] = r['annotation_name']
score = {}; REG = {}
for d, lab in LAB.items():
    gs = {}
    for lv, nm in lab.items():
        if nm in gi: gs.setdefault(gi[nm], []).append(lv)
    if len(gs) < 2: continue
    mk = np.asarray(Image.open(fetch(f'{GH}/masks/{d}.png', f'{d}.png')).resize((448, 448), Image.NEAREST)); j = tpos[d]; v = []
    for k, lvs in gs.items():
        w = np.isin(mk, lvs).astype(np.float32).reshape(16, 28, 16, 28).mean((1, 3)).reshape(256)
        if w.sum() >= 2: g = gm[j, :, k]; v.append((w * g).sum() / max(g.sum(), 1e-9) - (w * m[j]).sum() / max(m[j].sum(), 1e-9))
    if v: score[d] = float(np.mean(v)); REG[d] = (gs, mk)
assert set(tpos[d] for d in score) == set(int(r) for r in z['rows']), 'row order mismatch'
cand = sorted([d for d in score if 2 <= len(REG[d][0]) <= 4], key=lambda d: -score[d])
pick = [('Best', cand[0]), ('Median', cand[len(cand) // 2]), ('Worst', cand[-1])]
print('dishes scored', len(score), 'candidates', len(cand), {k: (d, round(score[d], 3)) for k, d in pick})
PAL = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860']
fig, ax = plt.subplots(3, 3, figsize=(7.6, 6.0))
for r, (tag, d) in enumerate(pick):
    gs, mk = REG[d]; j = tpos[d]; img = np.asarray(Image.open(fetch(f'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/{d}/rgb.png', f'{d}_rgb.png')).convert('RGB').resize((448, 448), Image.BICUBIC))
    ks = sorted(gs, key=lambda k: -gm[j, :, k].sum()); col = {k: to_rgb(PAL[i % len(PAL)]) for i, k in enumerate(ks)}
    ov = img.astype(np.float32) / 255 * 0.35
    for k in ks:
        sel = np.isin(mk, gs[k]); ov[sel] = 0.45 * ov[sel] / 0.35 * 0.35 + 0.55 * np.array(col[k])
    g = gm[j].reshape(16, 16, -1); arg = g.argmax(-1); amt = g.sum(-1); a = np.clip(amt / np.percentile(amt, 98), 0, 1)
    pm = np.ones((16, 16, 3))
    for y in range(16):
        for x in range(16):
            k = arg[y, x]; c = np.array(col.get(k, (0.6, 0.6, 0.6))); pm[y, x] = 1 - a[y, x] * (1 - c)
    pm = np.kron(pm, np.ones((28, 28, 1)))
    for c_, im, tt in zip(range(3), (img, ov, pm), ('Photo', 'True ingredient masks', 'IAF ingredient map')):
        ax[r, c_].imshow(im); ax[r, c_].set_xticks([]); ax[r, c_].set_yticks([])
        if r == 0: ax[r, c_].set_title(tt, fontsize=11)
    ax[r, 0].set_ylabel(f'{tag} (H$-$H$_0$ = {score[d]:.2f})', fontsize=10)
    hs = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in ks]; tm = {k: sum(float(r['mass'] or 0) for r in rows if r['dish_id'] == d and r['annotation_name'] == GR[k]) for k in ks}; lab = [f"{GR[k].replace('_', ' ')}: {gm[j, :, k].sum():.0f} g (true {tm[k]:.0f} g)" for k in ks]
    ax[r, 2].legend(hs, lab, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
plt.tight_layout(); plt.subplots_adjust(wspace=0.04, hspace=0.06); os.makedirs(os.path.join(H, '..', 'figures_out'), exist_ok=True); out = os.path.join(H, '..', 'figures_out', 'F_iaf_maps')
fig.savefig(out + '.pdf', bbox_inches='tight'); fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
json.dump({k: dict(dish=d, score=score[d], groups=[GR[x] for x in REG[d][0]]) for k, d in pick}, open(os.path.join(E, 'fig_iaf_pick.json'), 'w'), indent=1)
print('saved', out)
