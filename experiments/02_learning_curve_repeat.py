# Independent repetition of the learning curves (new training subsets and seeds) and of NutritionVerse-Real. Kaggle GPU T4.
import os, io, re, glob, time, json, csv, zipfile, platform, concurrent.futures as cf, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, requests
from PIL import Image
DRY = os.environ.get('DRY'); SMOKE = os.environ.get('SMOKE')
OUT = '/kaggle/working/testDX' if not DRY else DRY; W = '/kaggle/temp/dx' if not DRY else DRY + '/tmp'; os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True); T0 = time.time()
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
Y5 = {}
for f in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
    for line in get(f'{B}/metadata/{f}', False).splitlines():
        p = line.strip().split(',')
        if len(p) > 6:
            try: Y5[p[0]] = [float(x) for x in p[1:6]]
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
def pca(Cf, trm):
    C = Cf.float(); D = C.shape[-1]; mu = C[torch.tensor(trm)].reshape(-1, D).mean(0); Zs = C[torch.tensor(trm)].reshape(-1, D); Zs = Zs[torch.randperm(len(Zs))[:60000]] - mu
    _, _, V = torch.linalg.svd(Zs, full_matrices=False); X = (C - mu) @ V[:128].T; return (X / X[torch.tensor(trm)].std()).to(dev)
# ---------------- models ----------------
def make(kind, scale):
    class Pool(nn.Module):
        def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, 5))
        def forward(s, x): return Fn.softplus(s.f(torch.cat([x.mean(1), x.amax(1)], 1))) * scale, None
    class Sum(nn.Module):
        def __init__(s):
            super().__init__(); s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5)); s.a = nn.Parameter(torch.log(torch.expm1(torch.tensor([9., 4., 4.]))))
        def forward(s, x): mp = Fn.softplus(s.f(x)) * scale / 256; return mp.sum(1), mp
    class ANF(nn.Module):
        def __init__(s):
            super().__init__(); s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5)); s.a = nn.Parameter(torch.log(torch.expm1(torch.tensor([9., 4., 4.]))))
            with torch.no_grad(): s.f[-1].bias[1:] = torch.log(torch.tensor([.06, .09, .08, .77]))
        def forward(s, x):
            o = s.f(x); m = Fn.softplus(o[..., 0]) * scale[1] / 256; pi = torch.softmax(o[..., 1:], -1); comp = m[..., None] * pi[..., :3]; e = (comp * Fn.softplus(s.a)).sum(-1)
            return torch.stack([e.sum(1), m.sum(1), comp[..., 0].sum(1), comp[..., 1].sum(1), comp[..., 2].sum(1)], 1), torch.stack([m, e], -1)
    return {'R_pool': Pool, 'R_sum': Sum, 'R_soft': Sum, 'M': ANF}[kind]().to(dev)
def run(X, Yn, trm, tem, tag, keep_maps=False, seeds=SEEDS):
    Yt = torch.tensor(Yn, device=dev); scale = Yt[torch.tensor(trm, device=dev)].mean(0); itr = np.where(trm)[0]; ite = torch.tensor(np.where(tem)[0], device=dev)
    P, MP, FA = {}, {}, {}
    for kind in ('R_pool', 'R_sum', 'R_soft', 'M'):
        ps, mps, fas = [], [], []
        for seed in seeds:
            torch.manual_seed(seed); rng = np.random.default_rng(seed); net = make(kind, scale); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP); cur = []
            for e in range(EP):
                net.train(); tot = []
                for b in np.array_split(rng.permutation(itr), max(1, len(itr) // 64)):
                    b = torch.tensor(b, device=dev); p, _ = net(X[b]); loss = ((p - Yt[b]).abs() / scale).sum(1).mean()
                    if kind == 'R_soft': loss = loss + ((p[:, 0] - (p[:, 2:] * Fn.softplus(net.a)).sum(1)).abs() / scale[0]).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); tot.append(loss.item())
                sch.step(); cur.append(round(float(np.mean(tot)), 5))
            CURVES[f'{tag}_{kind}_s{seed}'] = cur; net.eval()
            with torch.no_grad(): p, mp = net(X[ite])
            ps.append(p.cpu().numpy())
            if keep_maps and mp is not None: mps.append(mp.cpu().numpy())
            if kind in ('R_soft', 'M'): fas.append(Fn.softplus(net.a).tolist())
        P[kind] = np.mean(ps, 0)
        if mps: MP[kind] = np.mean(mps, 0).astype(np.float16)
        if fas: FA[kind] = fas
    return P, MP, FA
def pm(p, y, i, idx=slice(None)): return float(100 * np.abs(p[idx, i] - y[idx, i]).sum() / y[idx, i].sum())
def summarize(P, y, af):
    R = {}
    for k, p in P.items():
        R[k] = {t: pm(p, y, i) for i, t in enumerate(TG)}; R[k]['macro_mean'] = float(np.mean([R[k][t] for t in TG[1:]])); R[k]['kcal_mae'] = float(np.abs(p[:, 0] - y[:, 0]).mean())
        R[k]['consistency_gap_pct'] = float(100 * np.mean(np.abs(p[:, 0] - p[:, 2:] @ af) / np.maximum(p[:, 0], 1)))
    return R
CURVES = {}; RES = {}; FACT = {}
FR = (0.05, 0.10, 0.25, 0.50, 1.0); KS = ('R_pool', 'R_sum', 'R_soft', 'M')
C5 = cells(IM5, 'base'); log('N5k features done'); y = Yn5[te5]; itr_all = np.where(~te5)[0]
PRED = {}
for f in FR:
    draws = (0,) if f == 1.0 else (0, 1, 2); acc = {k: [] for k in KS}
    for dr in draws:
        sub = np.sort(np.random.default_rng(100 + dr).choice(itr_all, max(20, int(round(f * len(itr_all)))), replace=False)) if f < 1 else itr_all
        trm = np.zeros(len(OK5), bool); trm[sub] = True; X5 = pca(C5, trm)
        P, _, FA = run(X5, Yn5, trm, te5, f'n5k_f{f}_d{dr}', seeds=((20 + dr,) if f < 1 else (20, 21, 22)))
        for k in KS: acc[k].append(P[k])
        FACT[f'n5k_f{f}_d{dr}'] = FA['M']
    PRED[f] = {k: np.mean(acc[k], 0) for k in KS}
    af = np.mean([np.mean(v, 0) for kk, v in FACT.items() if kk.startswith(f'n5k_f{f}_')], 0)
    R = summarize(PRED[f], y, af); RES[f'n5k_f{f}'] = R
    log('fraction', f, 'n_train', len(sub), {k: (round(R[k]['kcal'], 2), round(R[k]['macro_mean'], 2)) for k in KS}, 'factors', np.round(af, 2).tolist())
def boot(Pa, Pb, idxs):  # mean PMAE difference (Pb - Pa) over targets idxs, bootstrap over test dishes
    rng = np.random.default_rng(0); n = len(y); out = []
    for _ in range(2000):
        j = rng.integers(0, n, n); out.append(np.mean([pm(Pb, y, i, j) - pm(Pa, y, i, j) for i in idxs]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
S1 = S2 = True; CI = {}
for f in (0.05, 0.10, 0.25):
    R = RES[f'n5k_f{f}']; rk = min(('R_pool', 'R_sum', 'R_soft'), key=lambda k: R[k]['kcal']); rm = min(('R_pool', 'R_sum', 'R_soft'), key=lambda k: R[k]['macro_mean'])
    ck = boot(PRED[f]['M'], PRED[f][rk], [0]); cm = boot(PRED[f]['M'], PRED[f][rm], [1, 2, 3, 4]); CI[str(f)] = dict(best_kcal_rival=rk, ci_kcal=ck, best_macro_rival=rm, ci_macro=cm)
    S1 &= R['M']['kcal'] <= 0.95 * R[rk]['kcal'] and (ck[0] > 0 if f <= 0.10 else True)
    if f <= 0.10: S2 &= R['M']['macro_mean'] <= 0.95 * R[rm]['macro_mean'] and cm[0] > 0
R = RES['n5k_f1.0']; S3 = R['M']['kcal'] <= 1.05 * min(R[k]['kcal'] for k in ('R_pool', 'R_sum', 'R_soft'))
# NV-Real replication with fresh seeds (data already seen in DW: disclosed)
CV = cells(IMV, 'base'); PV = {k: np.zeros((len(dV), 5)) for k in KS}; FV = []
for k_ in range(5):
    trm, tem = foldV != k_, foldV == k_; XV = pca(CV, trm); P, _, FA = run(XV, YnV, trm, tem, f'nvr_f{k_}', seeds=(20, 21, 22))
    for k in KS: PV[k][tem] = P[k]
    FV += FA['M']
RV = summarize(PV, YnV, np.mean(FV, 0)); RES['nvr_base'] = RV; FACT['nvr'] = FV
S4 = RV['M']['kcal'] <= 0.90 * min(RV[k]['kcal'] for k in ('R_pool', 'R_sum', 'R_soft'))
RES.update(CI=CI, S1=bool(S1), S2=bool(S2), S3=bool(S3), S4=bool(S4), verdict='GO' if all([S1, S2, S3, S4]) else 'NO-GO')
json.dump(RES, open(OUT + '/results.json', 'w'), indent=1); json.dump(FACT, open(OUT + '/factors.json', 'w')); json.dump(CURVES, open(OUT + '/curves.json', 'w'))
with open(OUT + '/learning_curve.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['fraction', 'model'] + TG + ['macro_mean', 'consistency_gap_pct'])
    for f in FR:
        for k in KS: w.writerow([f, k] + [round(RES[f'n5k_f{f}'][k][t], 3) for t in TG] + [round(RES[f'n5k_f{f}'][k]['macro_mean'], 3), round(RES[f'n5k_f{f}'][k]['consistency_gap_pct'], 3)])
tid = [OK5[i] for i in np.where(te5)[0]]
for f in FR:
    with open(OUT + f'/preds_n5k_f{f}.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['dish'] + [f'true_{t}' for t in TG] + [f'{k}_{t}' for k in KS for t in TG])
        for j, d in enumerate(tid): w.writerow([d] + [round(float(v), 2) for v in y[j]] + [round(float(PRED[f][k][j, i]), 2) for k in KS for i in range(5)])
json.dump(dict(total_min=(time.time() - T0) / 60), open(OUT + '/timing.json', 'w')); json.dump(dict(python=platform.python_version(), torch=torch.__version__, device=dev), open(OUT + '/env.json', 'w'))
log('CI', json.dumps(CI)); log('NV', json.dumps({k: (round(RV[k]['kcal'], 2), round(RV[k]['macro_mean'], 2)) for k in KS}), 'factors', np.round(np.mean(FV, 0), 3).tolist())
log('RESULT', json.dumps({k: RES[k] for k in ('S1', 'S2', 'S3', 'S4', 'verdict')}))
log('VERDICT', RES['verdict'])
