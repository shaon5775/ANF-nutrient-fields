# Cheap partial labels: energy-only / mass-only / energy+mass labels for dishes outside a 10% fully labelled subset. Kaggle GPU T4.
import os, io, re, glob, time, json, csv, zipfile, platform, concurrent.futures as cf, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, requests
from PIL import Image
DRY = os.environ.get('DRY'); SMOKE = os.environ.get('SMOKE')
OUT = '/kaggle/working/EA' if not DRY else DRY; W = '/kaggle/temp/ea' if not DRY else DRY + '/tmp'; os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True); T0 = time.time()
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
def pca_fit(Cf, trm):
    C = Cf.float(); D = C.shape[-1]; Z = C[torch.tensor(trm)].reshape(-1, D); mu = Z.mean(0); Zs = Z[torch.randperm(len(Z))[:60000]] - mu
    _, _, V = torch.linalg.svd(Zs, full_matrices=False); V = V[:128].T; s = ((C[torch.tensor(trm)] - mu) @ V).std()
    return lambda Cg: (((Cg.float() - mu) @ V) / s).to(dev)
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
# ======================= Test EA body =======================
def build(kind, scale): return {'R_pool': lambda: Pool(scale), 'R_sum': lambda: Sum(scale), 'R_soft': lambda: Sum(scale), 'M': lambda: ANF(scale)}[kind]().to(dev)
KS = ('R_pool', 'R_sum', 'R_soft', 'M')
ARMS = {'A0': (0, 0, 0, 0, 0), 'AE': (1, 0, 0, 0, 0), 'AM': (0, 1, 0, 0, 0), 'AEM': (1, 1, 0, 0, 0)}  # which targets cheap dishes keep
def train(kind, X, Yt, Wt, scale, itr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = build(kind, scale)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    for e in range(EP):
        net.train()
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // 64)):
            b = torch.tensor(b, device=dev); p, _ = net(X[b]); loss = (Wt[b] * (p - Yt[b]).abs() / scale).sum(1).mean()
            if kind == 'R_soft': loss = loss + ((p[:, 0] - (p[:, 2:] * Fn.softplus(net.a)).sum(1)).abs() / scale[0]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval(); return net
def pm(p, y, i, idx=slice(None)): return float(100 * np.abs(p[idx, i] - y[idx, i]).sum() / y[idx, i].sum())
def mac(p, y, idx=slice(None)): return float(np.mean([pm(p, y, i, idx) for i in (2, 3, 4)]))
C = cells(IM5, 'base'); X = pca_fit(C, ~te5)(C); del C; torch.cuda.empty_cache()   # PCA on ALL training images
itr_all = np.where(~te5)[0]; ite = torch.tensor(np.where(te5)[0], device=dev); y5 = Yn5[te5]; Yfull = torch.tensor(Yn5, device=dev)
RES = {}; PRED = {}
for f in (0.10, 0.05):
    for dr in (200, 201, 202):
        sub = np.random.default_rng(dr).choice(itr_all, max(20, int(round(f * len(itr_all)))), replace=False); cheap = np.setdiff1d(itr_all, sub)
        for arm, keep in ARMS.items():
            Wn = np.zeros((len(OK5), 5), np.float32); Wn[sub] = 1
            if arm != 'A0': Wn[cheap] = np.array(keep, np.float32)
            itr = sub if arm == 'A0' else itr_all; Wt = torch.tensor(Wn, device=dev); Yt = Yfull * Wt
            scale = torch.tensor([Yn5[Wn[:, i] > 0, i].mean() for i in range(5)], device=dev)
            for k in KS:
                sd = dr - 200; net = train(k, X, Yt, Wt, scale, itr, sd)
                with torch.no_grad(): p = net(X[ite])[0].cpu().numpy()
                key = f'f{f}_{arm}_{k}_d{dr}'; PRED[key] = p
                RES[key] = dict(pmae=[pm(p, y5, i) for i in range(5)], macro=mac(p, y5), factors=(Fn.softplus(net.a).tolist() if k in ('M', 'R_soft') else None), n_full=len(sub), n_cheap=int(len(cheap) * (arm != 'A0')))
                log(key, 'macro', round(RES[key]['macro'], 2), 'kcal', round(RES[key]['pmae'][0], 2), 'mass', round(RES[key]['pmae'][1], 2))
            json.dump(RES, open(OUT + '/ea_results.json', 'w'))
np.savez_compressed(OUT + '/ea_preds.npz', y=y5, **{k: v for k, v in PRED.items()})
log('DONE minutes', round((time.time() - T0) / 60, 1))
