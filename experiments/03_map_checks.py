# Map faithfulness: deletion test (empty-plate replacement feature) and border energy share. Kaggle GPU T4.
import os, io, re, glob, time, json, csv, zipfile, platform, concurrent.futures as cf, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, requests
from PIL import Image
DRY = os.environ.get('DRY'); SMOKE = os.environ.get('SMOKE')
OUT = '/kaggle/working/DY' if not DRY else DRY; W = '/kaggle/temp/dy' if not DRY else DRY + '/tmp'; os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True); T0 = time.time()
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
os.makedirs(OUT + '/figures', exist_ok=True)
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
CURVES, TRAIN_S = {}, {}
C = cells(IM5, 'base'); proj = pca_fit(C, ~te5); X = proj(C); Yt = torch.tensor(Yn5, device=dev); scale = Yt[torch.tensor(~te5, device=dev)].mean(0)
itr = np.where(~te5)[0]; ite = torch.tensor(np.where(te5)[0], device=dev); Xte = X[ite]; G = 16
ring = np.zeros((G, G), bool); ring[:2] = ring[-2:] = True; ring[:, :2] = ring[:, -2:] = True; ring = torch.tensor(ring.reshape(-1), device=dev)
OUTR = {}
for kind in ('M', 'R_sum', 'R_soft'):
    for seed in (0, 1, 2):
        net = train(kind, X, Yt, scale, itr, seed, 'dy')
        with torch.no_grad(): p0, mp0 = net(Xte); _, mptr = net(X[torch.tensor(itr, device=dev)])
        e0 = mp0[..., 1] if kind == 'M' else mp0[..., 0]; etr = mptr[..., 1] if kind == 'M' else mptr[..., 0]
        # background feature = mean feature of the 20% training cells with the LOWEST predicted energy (empty plate/table)
        flat = etr.reshape(-1); thr = torch.quantile(flat[torch.randperm(len(flat), device=dev)[:200000]], 0.2); Xtr_c = X[torch.tensor(itr, device=dev)].reshape(-1, X.shape[-1])
        bg = Xtr_c[(flat <= thr)].mean(0)
        border = float((e0[:, ring].sum(1) / e0.sum(1)).mean())
        r = dict(border_energy_share=border)
        for k in (8, 16, 32, 64):
            top = torch.topk(e0, k, 1).indices; g = torch.Generator(device=dev).manual_seed(seed + k); rnd = torch.stack([torch.randperm(G * G, device=dev, generator=g)[:k] for _ in range(len(Xte))])
            for nm, ix in (('top', top), ('random', rnd)):
                Xd = Xte.clone(); Xd[torch.arange(len(Xd), device=dev)[:, None], ix] = bg
                with torch.no_grad(): r[f'{nm}_{k}'] = float((1 - net(Xd)[0][:, 0] / p0[:, 0]).mean())
        OUTR[f'{kind}_s{seed}'] = r; log(kind, seed, {q: round(v, 4) for q, v in r.items()})
S = {}
for kind in ('M', 'R_sum', 'R_soft'):
    rs = [OUTR[f'{kind}_s{s}'] for s in (0, 1, 2)]; S[kind] = {q: [float(np.mean([x[q] for x in rs])), float(np.std([x[q] for x in rs]))] for q in rs[0]}
uniform_border = float(ring.float().mean())
D1 = all(S['M'][f'top_{k}'][0] > S['M'][f'random_{k}'][0] + 0.05 for k in (8, 16, 32)) and all(S['M'][f'top_{a}'][0] <= S['M'][f'top_{b}'][0] + 1e-3 for a, b in ((8, 16), (16, 32), (32, 64)))
D2 = S['M']['border_energy_share'][0] <= 0.5 * min(S['R_sum']['border_energy_share'][0], S['R_soft']['border_energy_share'][0])
RES = dict(summary=S, runs=OUTR, uniform_border_share=uniform_border, D1=bool(D1), D2=bool(D2))
json.dump(RES, open(OUT + '/results.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(8, 3)); ks = [8, 16, 32, 64]
for kind in ('M', 'R_sum', 'R_soft'):
    ax[0].errorbar(ks, [100 * S[kind][f'top_{k}'][0] for k in ks], yerr=[100 * S[kind][f'top_{k}'][1] for k in ks], marker='o', color=COL[kind], label=NAME[kind] + ' (top cells)')
ax[0].errorbar(ks, [100 * S['M'][f'random_{k}'][0] for k in ks], yerr=[100 * S['M'][f'random_{k}'][1] for k in ks], marker='s', ls='--', color='grey', label='ANF (random cells)')
ax[0].set_xlabel('cells replaced by empty-plate feature (of 256)'); ax[0].set_ylabel('fall in predicted energy (%)'); ax[0].legend(frameon=False, fontsize=7); ax[0].grid(alpha=.3)
ax[1].bar(range(3), [100 * S[k]['border_energy_share'][0] for k in ('M', 'R_sum', 'R_soft')], yerr=[100 * S[k]['border_energy_share'][1] for k in ('M', 'R_sum', 'R_soft')], color=[COL[k] for k in ('M', 'R_sum', 'R_soft')])
ax[1].axhline(100 * uniform_border, color='k', ls=':', lw=.8); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([NAME[k] for k in ('M', 'R_sum', 'R_soft')], rotation=15, fontsize=7); ax[1].set_ylabel('energy on outer 2-cell ring (%)')
fig.savefig(OUT + '/figures/F_map_sanity.pdf', bbox_inches='tight'); fig.savefig(OUT + '/figures/F_map_sanity.png', dpi=300, bbox_inches='tight')
log('RESULT', json.dumps(dict(D1=RES['D1'], D2=RES['D2'], uniform_border=round(uniform_border, 3), summary={k: {q: round(v[0], 4) for q, v in S[k].items()} for k in S})))
log('DONE')
