# ANF full study: main results (both backbones, 5 seeds), learning curves, NutritionVerse-Real, cross-dataset, factors, ablations, cost, error analysis. Kaggle GPU T4.
import os, io, re, glob, time, json, csv, zipfile, platform, concurrent.futures as cf, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, requests
from PIL import Image
DRY = os.environ.get('DRY'); SMOKE = os.environ.get('SMOKE')
OUT = '/kaggle/working/FA' if not DRY else DRY; W = '/kaggle/temp/fa' if not DRY else DRY + '/tmp'; os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True); T0 = time.time()
L = open(OUT + '/log.txt', 'a')
def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(map(str, a)); L.write(s + '\n'); L.flush(); print(round((time.time() - T0) / 60, 1), 'min', *a, flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
EP = 150 if not SMOKE else 3; SEEDS = (10, 11, 12); TG = ['kcal', 'mass', 'fat', 'carb', 'protein']
BB = {'base': 'facebook/dinov2-base', 'large': 'facebook/dinov2-large'}
def get(u, binary=True):
    for k in range(5):
        try:
            r = requests.get(u, timeout=60)
            if r.status_code == 404: return None
            r.raise_for_status(); return r.content if binary else r.text
        except Exception: time.sleep(5 * 2 ** k)
    return None
# ---------------- Nutrition5k ----------------
B = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset'
Y5 = {}; NI = {}
for f in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
    for line in get(f'{B}/metadata/{f}', False).splitlines():
        p = line.strip().split(',')
        if len(p) > 6:
            try: Y5[p[0]] = [float(x) for x in p[1:6]]; NI[p[0]] = (len(p) - 6) // 7
            except ValueError: pass
tr_ids = get(f'{B}/dish_ids/splits/rgb_train_ids.txt', False).split(); te_ids = get(f'{B}/dish_ids/splits/rgb_test_ids.txt', False).split()
ids = [d for d in tr_ids + te_ids if d in Y5 and Y5[d][0] > 0]; tes = set(te_ids)
if SMOKE: ids = [d for d in ids if d not in tes][:30] + [d for d in ids if d in tes][:12]
def fetch(d):
    a = get(f'{B}/imagery/realsense_overhead/{d}/rgb.png')
    if a is None: return None
    try: return np.asarray(Image.open(io.BytesIO(a)).convert('RGB').resize((448, 448), Image.BICUBIC), np.uint8)
    except Exception: return None
IM5, OK5 = [], []
with cf.ThreadPoolExecutor(12) as ex:
    for d, im in zip(ids, ex.map(fetch, ids)):
        if im is not None: IM5.append(im); OK5.append(d)
IM5 = np.stack(IM5); te5 = np.array([d in tes for d in OK5]); Yn5 = np.array([Y5[d] for d in OK5], np.float32); log('N5k images', len(OK5), 'test', int(te5.sum()))
# ---------------- NutritionVerse-Real ----------------
NV = W + '/nvr'; z = W + '/nvr.zip'
if not glob.glob(NV + '/**/*.csv', recursive=True):
    with requests.get('https://www.kaggle.com/api/v1/datasets/download/nutritionverse/nutritionverse-real', stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(z, 'wb') as fh:
            for ch in r.iter_content(1 << 20): fh.write(ch)
    zipfile.ZipFile(z).extractall(NV); os.remove(z)
Mrows = list(csv.DictReader(open(glob.glob(NV + '/**/nutritionverse_dish_metadata3.csv', recursive=True)[0], encoding='utf-8', errors='ignore'))); cal = [c for c in Mrows[0] if 'calor' in c.lower()][0]; log('NV columns', list(Mrows[0])[:12], 'calorie col', cal)
NVY = {}
for r in Mrows:
    try: v = [float(r[cal]), float(r['total_food_weight']), float(r['total_fats']), float(r['total_carbohydrates']), float(r['total_protein'])]; d = int(float(r['dish_id']))
    except (ValueError, KeyError): continue
    if min(v[:2]) > 0 and min(v) >= 0: NVY[d] = v
NVP = []
for p in sorted(glob.glob(NV + '/**/images/*.jpg', recursive=True)):
    m = re.match(r'^dish_(\d+)_', os.path.basename(p))
    if m and int(m.group(1)) in NVY: NVP.append((int(m.group(1)), p))
if SMOKE: NVP = NVP[:40]
IMV = np.stack([np.asarray(Image.open(p).convert('RGB').resize((448, 448), Image.BICUBIC), np.uint8) for _, p in NVP]); dV = np.array([d for d, _ in NVP]); YnV = np.array([NVY[d] for d in dV], np.float32)
ud = np.unique(dV); fd = {d: i % 5 for i, d in enumerate(np.random.default_rng(0).permutation(ud))}; foldV = np.array([fd[d] for d in dV]); log('NV images', len(dV), 'dishes', len(ud))
# ---------------- features ----------------
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); SD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
def cells(IM, name):
    if SMOKE: return torch.randn(len(IM), 256, 1024 if name == 'large' else 768).half()
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(BB[name]).to(dev).eval().half(); out = []
    for b in range(0, len(IM), 32):
        x = ((torch.tensor(IM[b:b + 32], dtype=torch.float32).permute(0, 3, 1, 2) / 255) - MEAN) / SD
        with torch.no_grad(): t = enc(pixel_values=x.to(dev).half()).last_hidden_state[:, 1:].float()
        D = t.shape[-1]; out.append(t.reshape(-1, 32, 32, D).reshape(-1, 16, 2, 16, 2, D).mean((2, 4)).reshape(-1, 256, D).half().cpu())
    del enc; torch.cuda.empty_cache(); return torch.cat(out)
# ======================= ANF full study body =======================
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs(OUT + '/tables', exist_ok=True); os.makedirs(OUT + '/figures', exist_ok=True)
KS = ('R_pool', 'R_sum', 'R_soft', 'M'); COL = {'R_pool': '#9e9e9e', 'R_sum': '#4c72b0', 'R_soft': '#dd8452', 'M': '#c44e52', 'M_fixed': '#8172b3', 'M_nocomp': '#55a868'}
NAME = {'R_pool': 'Pooled regression', 'R_sum': 'Summed maps', 'R_soft': 'Maps + consistency loss', 'M': 'ANF (ours)', 'M_fixed': 'ANF, fixed 9/4/4', 'M_nocomp': 'ANF without composition'}
def pca_fit(Cf, trm):
    C = Cf.float(); D = C.shape[-1]; Z = C[torch.tensor(trm)].reshape(-1, D); mu = Z.mean(0); Zs = Z[torch.randperm(len(Z))[:60000]] - mu
    _, _, V = torch.linalg.svd(Zs, full_matrices=False); V = V[:128].T; s = ((C[torch.tensor(trm)] - mu) @ V).std()
    return lambda Cg: (((Cg.float() - mu) @ V) / s).to(dev)
def regrid(Cf, g):  # 16x16 cells -> g x g
    if g == 16: return Cf
    k = 16 // g; D = Cf.shape[-1]; return Cf.float().reshape(-1, g, k, g, k, D).mean((2, 4)).reshape(-1, g * g, D).half()
class Pool(nn.Module):
    def __init__(s, scale): super().__init__(); s.sc = scale; s.f = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, 5))
    def forward(s, x): return Fn.softplus(s.f(torch.cat([x.mean(1), x.amax(1)], 1))) * s.sc, None
class Sum(nn.Module):
    def __init__(s, scale):
        super().__init__(); s.sc = scale; s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5)); s.a = nn.Parameter(torch.log(torch.expm1(torch.tensor([9., 4., 4.]))))
    def forward(s, x): mp = Fn.softplus(s.f(x)) * s.sc / x.shape[1]; return mp.sum(1), mp
class ANF(nn.Module):
    def __init__(s, scale, fixed=False):
        super().__init__(); s.sc = scale; s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5)); s.a = nn.Parameter(torch.log(torch.expm1(torch.tensor([9., 4., 4.]))), requires_grad=not fixed)
        with torch.no_grad(): s.f[-1].bias[1:] = torch.log(torch.tensor([.06, .09, .08, .77]))
    def forward(s, x):
        o = s.f(x); m = Fn.softplus(o[..., 0]) * s.sc[1] / x.shape[1]; pi = torch.softmax(o[..., 1:], -1); comp = m[..., None] * pi[..., :3]; e = (comp * Fn.softplus(s.a)).sum(-1)
        return torch.stack([e.sum(1), m.sum(1), comp[..., 0].sum(1), comp[..., 1].sum(1), comp[..., 2].sum(1)], 1), torch.stack([m, e], -1)
class ANFnc(nn.Module):  # ablation: mass x free energy density per cell; macros from free maps (no shared composition)
    def __init__(s, scale): super().__init__(); s.sc = scale; s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5))
    def forward(s, x):
        o = Fn.softplus(s.f(x)); m = o[..., 0] * s.sc[1] / x.shape[1]; e = m * o[..., 1] * (s.sc[0] / s.sc[1]); mac = o[..., 2:] * s.sc[2:] / x.shape[1]
        return torch.cat([torch.stack([e.sum(1), m.sum(1)], 1), mac.sum(1)], 1), torch.stack([m, e], -1)
def build(kind, scale): return {'R_pool': lambda: Pool(scale), 'R_sum': lambda: Sum(scale), 'R_soft': lambda: Sum(scale), 'M': lambda: ANF(scale), 'M_fixed': lambda: ANF(scale, True), 'M_nocomp': lambda: ANFnc(scale)}[kind]().to(dev)
def train(kind, X, Yt, scale, itr, seed, tag):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = build(kind, scale); opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP); cur = []; t1 = time.time()
    for e in range(EP):
        net.train(); tot = []
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // 64)):
            b = torch.tensor(b, device=dev); p, _ = net(X[b]); loss = ((p - Yt[b]).abs() / scale).sum(1).mean()
            if kind == 'R_soft': loss = loss + ((p[:, 0] - (p[:, 2:] * Fn.softplus(net.a)).sum(1)).abs() / scale[0]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot.append(loss.item())
        sch.step(); cur.append(round(float(np.mean(tot)), 5))
    CURVES[f'{tag}_{kind}_s{seed}'] = cur; TRAIN_S.setdefault(kind, []).append(time.time() - t1); net.eval(); return net
def predict(net, X, idx):
    with torch.no_grad(): p, mp = net(X[torch.tensor(idx, device=dev)])
    return p.cpu().numpy(), (None if mp is None else mp.cpu().numpy())
def factors(net): return Fn.softplus(net.a).tolist() if hasattr(net, 'a') else None
def run(X, Yn, trm, tem, kinds, seeds, tag, keep=None):
    Yt = torch.tensor(Yn, device=dev); scale = Yt[torch.tensor(trm, device=dev)].mean(0); itr = np.where(trm)[0]; ite = np.where(tem)[0]; PS, FA, MP = {}, {}, {}
    for k in kinds:
        PS[k] = []
        for s in seeds:
            net = train(k, X, Yt, scale, itr, s, tag); p, mp = predict(net, X, ite); PS[k].append(p)
            if k in ('M', 'M_fixed', 'R_soft'): FA.setdefault(k, []).append(factors(net))
            if keep is not None and s == seeds[0]: keep[k] = net
            if mp is not None and s == seeds[0]: MP[k] = mp
    return PS, FA, MP
def pm(p, y, i, idx=slice(None)): return float(100 * np.abs(p[idx, i] - y[idx, i]).sum() / y[idx, i].sum())
def gap(p, af): return 100 * np.abs(p[:, 0] - p[:, 2:] @ af) / np.maximum(p[:, 0], 1)
def boot(pa, pb, y, idxs, n=2000):
    rng = np.random.default_rng(0); N = len(y); o = []
    for _ in range(n):
        j = rng.integers(0, N, N); o.append(np.mean([pm(pb, y, i, j) - pm(pa, y, i, j) for i in idxs]))
    return [round(float(np.percentile(o, 2.5)), 3), round(float(np.percentile(o, 97.5)), 3)]
CURVES, TRAIN_S, RES, FACT = {}, {}, {}, {}
TBL = {}
def save_csv(name, header, rows):
    with open(f'{OUT}/tables/{name}.csv', 'w', newline='') as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)
def savefig(fig, name): fig.savefig(f'{OUT}/figures/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/figures/{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)
y5 = Yn5[te5]; tid = [OK5[i] for i in np.where(te5)[0]]
# ---------- T1 datasets ----------
save_csv('T1_datasets', ['dataset', 'images', 'dishes', 'train', 'test_or_folds', 'mean_kcal', 'mean_mass_g', 'mean_fat_g', 'mean_carb_g', 'mean_protein_g', 'licence'],
         [['Nutrition5k (overhead RGB)', len(OK5), len(OK5), int((~te5).sum()), int(te5.sum())] + [round(float(v), 1) for v in Yn5.mean(0)] + ['CC BY 4.0'],
          ['NutritionVerse-Real', len(dV), len(np.unique(dV)), '5-fold', 'grouped by dish'] + [round(float(v), 1) for v in YnV.mean(0)] + ['CC BY-NC-SA 4.0']])
# ---------- E1 main + E6 inference timing ----------
TIMING = {}; KEEP = {}; CB = None
for bb in ('base', 'large'):
    t1 = time.time(); C = cells(IM5, bb); TIMING[f'feat_ms_per_img_{bb}'] = 1000 * (time.time() - t1) / len(IM5)
    proj = pca_fit(C, ~te5); X = proj(C)
    if bb == 'base': CB = C; PROJB = proj; XB = X
    else: del C
    PS, FA, MP = run(X, Yn5, ~te5, te5, KS, (0, 1, 2, 3, 4), f'main_{bb}', keep=(KEEP if bb == 'base' else None))
    af = np.mean(FA['M'], 0); R = {}
    for k in KS:
        per = np.array([[pm(p, y5, i) for i in range(5)] for p in PS[k]]); pmean = np.mean(PS[k], 0)
        R[k] = dict(mean=per.mean(0).tolist(), sd=per.std(0).tolist(), seedmean_pmae=[pm(pmean, y5, i) for i in range(5)], kcal_mae=float(np.abs(pmean[:, 0] - y5[:, 0]).mean()), gap=float(gap(pmean, af).mean()))
        if k != 'M': R[k]['ci_kcal_rival_minus_anf'] = boot(np.mean(PS['M'], 0), pmean, y5, [0]); R[k]['ci_macro_rival_minus_anf'] = boot(np.mean(PS['M'], 0), pmean, y5, [1, 2, 3, 4])
    RES[f'E1_{bb}'] = R; FACT[f'E1_{bb}'] = FA; log('E1', bb, {k: [round(v, 2) for v in R[k]['mean']] for k in KS}, 'factors', np.round(af, 2).tolist())
    if bb == 'base':
        PM_BASE = {k: np.mean(PS[k], 0) for k in KS}; MAPS = MP
        with open(OUT + '/preds_n5k_base.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['dish', 'n_ingredients'] + [f'true_{t}' for t in TG] + [f'{k}_{t}' for k in KS for t in TG])
            for j, d in enumerate(tid): w.writerow([d, NI.get(d, '')] + [round(float(v), 2) for v in y5[j]] + [round(float(PM_BASE[k][j, i]), 2) for k in KS for i in range(5)])
rows = []
for bb in ('base', 'large'):
    for k in KS:
        R = RES[f'E1_{bb}'][k]; rows.append([bb, NAME[k]] + [f"{R['mean'][i]:.2f} ± {R['sd'][i]:.2f}" for i in range(5)] + [f"{np.mean(R['mean'][1:]):.2f}", round(R['kcal_mae'], 1), round(R['gap'], 3), R.get('ci_kcal_rival_minus_anf', ''), R.get('ci_macro_rival_minus_anf', '')])
save_csv('T2_main_n5k', ['backbone', 'model', 'kcal PMAE', 'mass PMAE', 'fat PMAE', 'carb PMAE', 'protein PMAE', 'macro mean', 'kcal MAE', 'consistency gap %', 'CI kcal (rival-ANF, pts)', 'CI macro (rival-ANF, pts)'], rows)
# ---------- E2 learning curve ----------
LC = {}
for f in (0.05, 0.10, 0.25, 0.50, 1.0):
    draws = (200, 201, 202) if f < 1 else (200,); acc = {k: [] for k in KS}; fa = []
    for dr in draws:
        itr_all = np.where(~te5)[0]; sub = itr_all if f == 1 else np.random.default_rng(dr).choice(itr_all, max(20, int(round(f * len(itr_all)))), replace=False)
        trm = np.zeros(len(OK5), bool); trm[sub] = True; X = pca_fit(CB, trm)(CB)
        PS, FA, _ = run(X, Yn5, trm, te5, KS, ((dr - 200,) if f < 1 else (0, 1, 2)), f'lc_{f}_{dr}')
        for k in KS: acc[k].append(np.mean(PS[k], 0))
        fa += FA['M']
    LC[f] = {k: dict(kcal=[pm(p, y5, 0) for p in acc[k]], macro=[np.mean([pm(p, y5, i) for i in (1, 2, 3, 4)]) for p in acc[k]]) for k in KS}; LC[f]['factors'] = np.mean(fa, 0).tolist(); LC[f]['n'] = len(sub)
    log('E2', f, {k: round(float(np.mean(LC[f][k]['kcal'])), 2) for k in KS})
RES['E2'] = LC
save_csv('T3_learning_curve', ['fraction', 'n_train'] + [f'{NAME[k]} kcal' for k in KS] + [f'{NAME[k]} macro' for k in KS] + ['ANF factors'],
         [[f, LC[f]['n']] + [f"{np.mean(LC[f][k]['kcal']):.2f} ± {np.std(LC[f][k]['kcal']):.2f}" for k in KS] + [f"{np.mean(LC[f][k]['macro']):.2f} ± {np.std(LC[f][k]['macro']):.2f}" for k in KS] + [np.round(LC[f]['factors'], 2).tolist()] for f in LC])
# ---------- E3 NV-Real, E4 cross-dataset ----------
for bb in ('base', 'large'):
    CV = cells(IMV, bb); PV = {k: np.zeros((len(dV), 5)) for k in KS}; FV = []
    for k_ in range(5):
        trm, tem = foldV != k_, foldV == k_; X = pca_fit(CV, trm)(CV); PS, FA, _ = run(X, YnV, trm, tem, KS, (0, 1, 2), f'nv_{bb}_f{k_}')
        for k in KS: PV[k][tem] = np.mean(PS[k], 0)
        FV += FA['M']
    af = np.mean(FV, 0); RES[f'E3_{bb}'] = {k: dict(pmae=[pm(PV[k], YnV, i) for i in range(5)], gap=float(gap(PV[k], af).mean())) for k in KS}; FACT[f'E3_{bb}'] = FV
    for k in KS: RES[f'E3_{bb}'][k]['ci_kcal_rival_minus_anf'] = boot(PV['M'], PV[k], YnV, [0]) if k != 'M' else ''
    log('E3', bb, {k: round(RES[f'E3_{bb}'][k]['pmae'][0], 2) for k in KS}, 'factors', np.round(af, 2).tolist())
    if bb == 'base':
        CVB = CV; PVB = PV
        with open(OUT + '/preds_nvr_base.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['dish', 'fold'] + [f'true_{t}' for t in TG] + [f'{k}_{t}' for k in KS for t in TG])
            for j in range(len(dV)): w.writerow([int(dV[j]), int(foldV[j])] + [round(float(v), 2) for v in YnV[j]] + [round(float(PV[k][j, i]), 2) for k in KS for i in range(5)])
    else: del CV
save_csv('T5_nvr', ['backbone', 'model'] + [f'{t} PMAE' for t in TG] + ['macro mean', 'consistency gap %', 'CI kcal (rival-ANF)'],
         [[bb, NAME[k]] + [round(v, 2) for v in RES[f'E3_{bb}'][k]['pmae']] + [round(float(np.mean(RES[f'E3_{bb}'][k]['pmae'][1:])), 2), round(RES[f'E3_{bb}'][k]['gap'], 3), RES[f'E3_{bb}'][k]['ci_kcal_rival_minus_anf']] for bb in ('base', 'large') for k in KS])
# E4: train on all N5k train dishes, test on all NV photos (PCA from N5k train)
Xall = torch.cat([XB, PROJB(CVB)]); Yall = np.concatenate([Yn5, YnV]); trm = np.r_[~te5, np.zeros(len(dV), bool)]; tem = np.r_[np.zeros(len(OK5), bool), np.ones(len(dV), bool)]
PS, FA, _ = run(Xall, Yall, trm, tem, KS, (0, 1, 2), 'cross'); RES['E4'] = {k: [pm(np.mean(PS[k], 0), YnV, i) for i in range(5)] for k in KS}; log('E4 cross', {k: round(RES['E4'][k][0], 2) for k in KS})
save_csv('T5b_cross_n5k_to_nvr', ['model'] + [f'{t} PMAE' for t in TG], [[NAME[k]] + [round(v, 2) for v in RES['E4'][k]] for k in KS])
# ---------- E5 ablations ----------
AB = {}
for f in (1.0, 0.10):
    itr_all = np.where(~te5)[0]; sub = itr_all if f == 1 else np.random.default_rng(200).choice(itr_all, int(round(f * len(itr_all))), replace=False)
    trm = np.zeros(len(OK5), bool); trm[sub] = True; X = pca_fit(CB, trm)(CB); PS, FA, _ = run(X, Yn5, trm, te5, ('M', 'M_fixed', 'M_nocomp'), (0, 1, 2), f'abl_{f}')
    for k in PS: AB[f'{k}_f{f}'] = [pm(np.mean(PS[k], 0), y5, i) for i in range(5)]
for g in (4, 8):
    Cg = regrid(CB, g); X = pca_fit(Cg, ~te5)(Cg); PS, _, _ = run(X, Yn5, ~te5, te5, ('R_sum', 'M'), (0, 1, 2), f'grid{g}')
    for k in PS: AB[f'{k}_grid{g}'] = [pm(np.mean(PS[k], 0), y5, i) for i in range(5)]
for k in ('R_sum', 'M'): AB[f'{k}_grid16'] = RES['E1_base'][k]['seedmean_pmae']
RES['E5'] = AB; save_csv('T6_ablations', ['variant'] + [f'{t} PMAE' for t in TG], [[k] + [round(v, 2) for v in AB[k]] for k in AB]); log('E5', {k: round(v[0], 2) for k, v in AB.items()})
# ---------- E6 cost ----------
nets = {k: build(k, torch.ones(5, device=dev)) for k in KS}; xb = XB[:256]
for k, n_ in nets.items():
    n_.eval(); t1 = time.time()
    with torch.no_grad():
        for _ in range(20): n_(xb)
    if dev == 'cuda': torch.cuda.synchronize()
    TIMING[f'head_ms_per_img_{k}'] = 1000 * (time.time() - t1) / (20 * len(xb))
save_csv('T7_cost', ['model', 'trainable params', 'mean train s per run', 'head ms per image'], [[NAME[k], sum(p.numel() for p in nets[k].parameters() if p.requires_grad), round(float(np.mean(TRAIN_S[k])), 1), round(TIMING[f'head_ms_per_img_{k}'], 4)] for k in KS])
# ---------- E7 error analysis ----------
mt = np.quantile(y5[:, 1], [1 / 3, 2 / 3]); tert = np.digitize(y5[:, 1], mt); ni = np.array([NI.get(d, 0) for d in tid]); ng = np.where(ni <= 2, 0, np.where(ni <= 5, 1, 2))
rows = []
for nm, grp, labels in (('mass tertile', tert, ['light', 'medium', 'heavy']), ('ingredients', ng, ['1-2', '3-5', '6+'])):
    for g_, lab in enumerate(labels):
        idx = np.where(grp == g_)[0]
        if len(idx): rows.append([nm, lab, len(idx)] + [round(pm(PM_BASE[k], y5, 0, idx), 2) for k in KS] + [round(float(np.mean([pm(PM_BASE[k], y5, i, idx) for i in (1, 2, 3, 4)])), 2) for k in KS])
save_csv('T8_error_analysis', ['grouping', 'group', 'n'] + [f'{NAME[k]} kcal' for k in KS] + [f'{NAME[k]} macro' for k in KS], rows)
err = np.abs(PM_BASE['M'][:, 0] - y5[:, 0]) / y5[:, 0]; worst = np.argsort(-err)[:12]
save_csv('T8b_worst12', ['dish', 'true kcal', 'ANF kcal', 'rel error', 'true mass', 'n_ingredients'], [[tid[j], round(float(y5[j, 0]), 1), round(float(PM_BASE['M'][j, 0]), 1), round(float(err[j]), 3), round(float(y5[j, 1]), 1), int(ni[j])] for j in worst])
# ---------- E8 deletion test ----------
net = KEEP['M']; Xte = XB[torch.tensor(np.where(te5)[0], device=dev)]
with torch.no_grad(): p0, mp0 = net(Xte)
e0 = mp0[..., 1]; DEL = {}
for k in (8, 16, 32, 64):
    top = torch.topk(e0, k, 1).indices; rnd = torch.stack([torch.randperm(Xte.shape[1], device=dev)[:k] for _ in range(len(Xte))])
    out = []
    for ix in (top, rnd):
        Xd = Xte.clone(); Xd[torch.arange(len(Xd), device=dev)[:, None], ix] = 0
        with torch.no_grad(): out.append(float((1 - net(Xd)[0][:, 0] / p0[:, 0]).mean()))
    DEL[k] = dict(top_energy_cells=out[0], random_cells=out[1])
RES['E8'] = DEL; save_csv('T9_deletion', ['k cells removed', 'energy drop: top-energy cells', 'energy drop: random cells'], [[k, round(v['top_energy_cells'], 4), round(v['random_cells'], 4)] for k, v in DEL.items()]); log('E8', DEL)
# ---------- factors table ----------
frows = []
for key, v in FACT.items():
    for kk, lst in (v.items() if isinstance(v, dict) else [('M', v)]):
        for j, fa in enumerate(lst): frows.append([key, NAME.get(kk, kk), j] + [round(x, 3) for x in fa])
for f in LC: frows.append([f'E2_fraction_{f}', NAME['M'], 'mean'] + [round(x, 3) for x in LC[f]['factors']])
save_csv('T4_factors', ['experiment', 'model', 'run', 'fat kcal/g', 'carb kcal/g', 'protein kcal/g'], frows)
# ======================= FIGURES =======================
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
fr = list(LC); fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for k in KS:
    for a_, q in zip(ax, ('kcal', 'macro')):
        mu_ = [np.mean(LC[f][k][q]) for f in fr]; sd_ = [np.std(LC[f][k][q]) for f in fr]
        a_.errorbar([100 * f for f in fr], mu_, yerr=sd_, marker='o', ms=4, capsize=2, color=COL[k], label=NAME[k], lw=2 if k == 'M' else 1.2)
for a_, t in zip(ax, ('Energy (kcal) PMAE %', 'Mean PMAE % (mass, fat, carb, protein)')): a_.set_xscale('log'); a_.set_xlabel('Training dishes used (%)'); a_.set_ylabel(t); a_.set_xticks([5, 10, 25, 50, 100]); a_.set_xticklabels(['5', '10', '25', '50', '100']); a_.grid(alpha=.3)
ax[0].legend(frameon=False, fontsize=8); savefig(fig, 'F_learning_curve')
fig, ax = plt.subplots(1, 5, figsize=(14, 3))
for i, t in enumerate(TG):
    m_ = max(y5[:, i].max(), PM_BASE['M'][:, i].max()); ax[i].scatter(y5[:, i], PM_BASE['R_soft'][:, i], s=4, c=COL['R_soft'], alpha=.4, label=NAME['R_soft']); ax[i].scatter(y5[:, i], PM_BASE['M'][:, i], s=4, c=COL['M'], alpha=.6, label=NAME['M'])
    ax[i].plot([0, m_], [0, m_], 'k--', lw=.8); ax[i].set_title(t); ax[i].set_xlabel('true'); ax[i].set_ylabel('predicted' if i == 0 else '')
ax[0].legend(frameon=False, fontsize=7, markerscale=3); savefig(fig, 'F_scatter_n5k')
afb = np.mean(FACT['E1_base']['M'], 0); fig, ax = plt.subplots(figsize=(4.5, 3))
for k in KS: ax.hist(np.clip(gap(PM_BASE[k], afb), 1e-4, None), bins=np.logspace(-4, 2.5, 40), alpha=.6, color=COL[k], label=NAME[k])
ax.set_xscale('log'); ax.set_xlabel('|energy − Σ factor × macro| / energy (%)'); ax.set_ylabel('test dishes'); ax.legend(frameon=False, fontsize=7); savefig(fig, 'F_consistency')
fig, ax = plt.subplots(figsize=(5, 3)); groups = [('N5k base', FACT['E1_base']['M']), ('N5k large', FACT['E1_large']['M']), ('NV-Real base', FACT['E3_base']), ('NV-Real large', FACT['E3_large'])] + [(f'N5k {int(100 * f)}%', [LC[f]['factors']]) for f in LC]
for j, (lab, lst) in enumerate(groups):
    a_ = np.array(lst)
    for c, (ref, mk) in enumerate(zip((9, 4, 4), ('o', 's', '^'))): ax.scatter(np.full(len(a_), j) + (c - 1) * .2, a_[:, c] / ref, marker=mk, s=14, color=['#c44e52', '#4c72b0', '#55a868'][c], label=['fat (÷9)', 'carb (÷4)', 'protein (÷4)'][c] if j == 0 else None)
ax.axhspan(.85, 1.15, color='grey', alpha=.12); ax.axhline(1, color='k', lw=.8); ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], rotation=45, ha='right'); ax.set_ylabel('learned factor / Atwater'); ax.legend(frameon=False, fontsize=7); savefig(fig, 'F_factors')
fig, ax = plt.subplots(figsize=(4.5, 3)); ks_ = list(DEL); ax.plot(ks_, [100 * DEL[k]['top_energy_cells'] for k in ks_], 'o-', color=COL['M'], label='highest-energy cells'); ax.plot(ks_, [100 * DEL[k]['random_cells'] for k in ks_], 's--', color='grey', label='random cells')
ax.set_xlabel('cells removed (of 256)'); ax.set_ylabel('fall in predicted energy (%)'); ax.legend(frameon=False); ax.grid(alpha=.3); savefig(fig, 'F_deletion')
fig, ax = plt.subplots(figsize=(5, 3)); w_ = .2
for j, k in enumerate(KS): ax.bar(np.arange(3) + (j - 1.5) * w_, [pm(PM_BASE[k], y5, 0, np.where(tert == g_)[0]) for g_ in range(3)], w_, color=COL[k], label=NAME[k])
ax.set_xticks(range(3)); ax.set_xticklabels(['light', 'medium', 'heavy']); ax.set_ylabel('energy PMAE %'); ax.set_xlabel('dish mass tertile'); ax.legend(frameon=False, fontsize=7); savefig(fig, 'F_error_by_mass')
TH5 = {OK5[i]: np.asarray(Image.fromarray(IM5[i]).resize((256, 256))) for i in np.where(te5)[0]}
order = np.argsort(err); pick = list(order[:3]) + list(order[len(order) // 2 - 1:len(order) // 2 + 2]) + list(order[-2:])
fig, ax = plt.subplots(len(pick), 4, figsize=(8, 2.1 * len(pick)))
for r, j in enumerate(pick):
    m_, e_ = MAPS['M'][j, :, 0].reshape(16, 16), MAPS['M'][j, :, 1].reshape(16, 16); fs = MAPS['R_sum'][j, :, 0].reshape(16, 16)
    ax[r, 0].imshow(TH5[tid[j]]); ax[r, 0].set_ylabel(f'true {y5[j, 0]:.0f}\nANF {PM_BASE["M"][j, 0]:.0f} kcal', fontsize=7)
    for c, (mm, ttl, cm) in enumerate(((m_, 'ANF mass (g)', 'viridis'), (e_, 'ANF energy (kcal)', 'magma'), (fs, 'Summed-map energy', 'magma'))):
        ax[r, c + 1].imshow(TH5[tid[j]]); im_ = ax[r, c + 1].imshow(np.kron(mm, np.ones((16, 16))), cmap=cm, alpha=.6); plt.colorbar(im_, ax=ax[r, c + 1], fraction=.046)
        if r == 0: ax[r, c + 1].set_title(ttl, fontsize=8)
    for c in range(4): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
ax[0, 0].set_title('photo', fontsize=8); savefig(fig, 'F_maps_examples')
fig, ax = plt.subplots(figsize=(4.5, 3))
for k in KS: ax.plot(CURVES[f'main_base_{k}_s0'], color=COL[k], label=NAME[k])
ax.set_xlabel('epoch'); ax.set_ylabel('training loss'); ax.set_yscale('log'); ax.legend(frameon=False, fontsize=7); savefig(fig, 'F_training_curves')
# ---------- save ----------
np.savez_compressed(OUT + '/maps_n5k_base.npz', dish=np.array(tid), **{k: v.astype(np.float16) for k, v in MAPS.items()})
np.savez_compressed(OUT + '/thumbs_n5k_test.npz', dish=np.array(tid), rgb=np.stack([TH5[d] for d in tid]))
json.dump(RES, open(OUT + '/results.json', 'w'), indent=1, default=float); json.dump(FACT, open(OUT + '/factors.json', 'w')); json.dump(CURVES, open(OUT + '/curves.json', 'w'))
json.dump(dict(total_min=(time.time() - T0) / 60, **TIMING), open(OUT + '/timing.json', 'w'))
json.dump(dict(python=platform.python_version(), torch=torch.__version__, device=dev, gpu=(torch.cuda.get_device_name(0) if dev == 'cuda' else '')), open(OUT + '/env.json', 'w'))
log('FILES', sorted(os.listdir(OUT + '/tables')), sorted(os.listdir(OUT + '/figures')))
log('DONE')
