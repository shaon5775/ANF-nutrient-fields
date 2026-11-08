# Energy factors started away from 9/4/4 (6/6/6, 4/9/9, 3/3/3, 12/12/12), normal and 10x factor learning rate. Kaggle GPU T4.
import os, io, re, glob, time, json, csv, zipfile, platform, concurrent.futures as cf, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, requests
from PIL import Image
DRY = os.environ.get('DRY'); SMOKE = os.environ.get('SMOKE')
OUT = '/kaggle/working/DZ' if not DRY else DRY; W = '/kaggle/temp/dz' if not DRY else DRY + '/tmp'; os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True); T0 = time.time()
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
def pca_fit(Cf, trm):
    C = Cf.float(); D = C.shape[-1]; Z = C[torch.tensor(trm)].reshape(-1, D); mu = Z.mean(0); Zs = Z[torch.randperm(len(Z))[:60000]] - mu
    _, _, V = torch.linalg.svd(Zs, full_matrices=False); V = V[:128].T; s = ((C[torch.tensor(trm)] - mu) @ V).std()
    return lambda Cg: (((Cg.float() - mu) @ V) / s).to(dev)
# ======================= Test DZ body =======================
STARTS = {'S0': (9., 4., 4.), 'S1': (6., 6., 6.), 'S2': (4., 9., 9.), 'S3': (3., 3., 3.), 'S4': (12., 12., 12.)}
LRF = {'R1': 1e-3, 'R10': 1e-2}
class ANF(nn.Module):
    def __init__(s, scale, a0):
        super().__init__(); s.sc = scale; s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5)); s.a = nn.Parameter(torch.log(torch.expm1(torch.tensor(a0))))
        with torch.no_grad(): s.f[-1].bias[1:] = torch.log(torch.tensor([.06, .09, .08, .77]))
    def forward(s, x):
        o = s.f(x); m = Fn.softplus(o[..., 0]) * s.sc[1] / x.shape[1]; pi = torch.softmax(o[..., 1:], -1); comp = m[..., None] * pi[..., :3]; e = (comp * Fn.softplus(s.a)).sum(-1)
        return torch.stack([e.sum(1), m.sum(1), comp[..., 0].sum(1), comp[..., 1].sum(1), comp[..., 2].sum(1)], 1)
def train(X, Yt, scale, itr, seed, a0, lrf):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = ANF(scale, a0).to(dev)
    opt = torch.optim.AdamW([{'params': net.f.parameters(), 'lr': 1e-3}, {'params': [net.a], 'lr': lrf}], 1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP); traj = []
    for e in range(EP):
        net.train()
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // 64)):
            b = torch.tensor(b, device=dev); p = net(X[b]); loss = ((p - Yt[b]).abs() / scale).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); traj.append([round(v, 4) for v in Fn.softplus(net.a).tolist()])
    net.eval(); return net, traj
def pm(p, y, i): return float(100 * np.abs(p[:, i] - y[:, i]).sum() / y[:, i].sum())
RES = {}
# Nutrition5k, base backbone, full training set
C = cells(IM5, 'base'); X = pca_fit(C, ~te5)(C); Yt = torch.tensor(Yn5, device=dev); trm = ~te5; scale = Yt[torch.tensor(trm, device=dev)].mean(0)
itr = np.where(trm)[0]; ite = torch.tensor(np.where(te5)[0], device=dev); y5 = Yn5[te5]
for st, a0 in STARTS.items():
    for rn, lrf in LRF.items():
        for sd in ((10,) if SMOKE else (10, 11, 12)):
            net, traj = train(X, Yt, scale, itr, sd, a0, lrf)
            with torch.no_grad(): p = net(X[ite]).cpu().numpy()
            RES[f'N5k_{st}_{rn}_s{sd}'] = dict(start=a0, lr=lrf, final=traj[-1], traj=traj, pmae=[pm(p, y5, i) for i in range(5)])
            log('N5k', st, rn, sd, 'factors', traj[-1], 'kcal', round(pm(p, y5, 0), 2))
    json.dump(RES, open(OUT + '/dz_results.json', 'w'))
del C, X; torch.cuda.empty_cache()
# NutritionVerse-Real, base backbone, 5 dish-grouped folds, seed 10
CV = cells(IMV, 'base'); YtV = torch.tensor(YnV, device=dev)
for k_ in range(5):
    trm, tem = foldV != k_, foldV == k_; X = pca_fit(CV, trm)(CV); scale = YtV[torch.tensor(trm, device=dev)].mean(0); itr = np.where(trm)[0]; ite = torch.tensor(np.where(tem)[0], device=dev)
    for st, a0 in STARTS.items():
        for rn, lrf in LRF.items():
            net, traj = train(X, YtV, scale, itr, 10, a0, lrf)
            with torch.no_grad(): p = net(X[ite]).cpu().numpy()
            RES[f'NVR_{st}_{rn}_f{k_}'] = dict(start=a0, lr=lrf, final=traj[-1], traj=traj, pmae=[pm(p, YnV[tem], i) for i in range(5)])
            log('NVR fold', k_, st, rn, 'factors', traj[-1], 'kcal', round(pm(p, YnV[tem], 0), 2))
    json.dump(RES, open(OUT + '/dz_results.json', 'w'))
log('DONE minutes', round((time.time() - T0) / 60, 1))
