# IAF vs ANF with 5% and 10% of the training dishes, localisation on the public masks. Kaggle GPU T4.
import os, io, re, glob, time, json, csv, zipfile, platform, concurrent.futures as cf, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, requests
from PIL import Image
DRY = os.environ.get('DRY'); SMOKE = os.environ.get('SMOKE')
OUT = '/kaggle/working/EC' if not DRY else DRY; W = '/kaggle/temp/eb' if not DRY else DRY + '/tmp'; os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True); T0 = time.time()
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
KINDS = ('IAF', 'ANF')
# ======================= Test EB body =======================
GMAP = {"celery": "berry", "olives": "fruit_fatty", "brussels sprouts": "cruciferous", "eggplant": "squash_eggplant", "chicken": "poultry", "cauliflower": "cruciferous", "steak": "meat_red", "cucumbers": "cucumber", "eggs": "egg", "broccoli": "cruciferous", "strawberries": "berry", "raspberries": "berry", "bacon": "bacon", "yam": "tuber", "blackberries": "berry", "cantaloupe": "melon", "deprecated": "fruit_tree", "potatoes": "tuber", "honeydew melons": "melon", "grapes": "fruit_tree", "mixed greens": "leafy_greens", "almonds": "nuts_seeds", "sausage": "sausage", "spinach (raw)": "leafy_greens", "white rice": "rice", "cherry tomatoes": "tomato", "avocado": "fruit_fatty", "mushroom": "mushroom", "squash": "squash_eggplant", "oatmeal": "grain_other", "apple": "fruit_tree", "sweet potato": "tuber", "carrot": "root_vegetable", "garden salad": "leafy_greens", "tomatoes": "tomato", "corn on the cob": "corn", "vinegar": "berry", "walnuts": "berry", "blueberries": "berry", "pepper": "peppers", "mustard": "nuts_seeds", "salt": "herb", "pineapple": "fruit_tropical", "orange": "fruit_tropical", "pears": "fruit_tree", "chickpeas": "legumes_other", "parsley": "herb", "onions": "alium_vegetable", "cranberries": "berry", "olive oil": "leafy_greens", "asparagus": "vegetable_other", "scrambled eggs": "egg", "hash browns": "tuber", "bell peppers": "peppers", "berries": "berry", "chicken apple sausage": "sausage", "mandarin oranges": "fruit_tropical", "caesar salad": "salad_prepared", "pepperoni pizza": "pizza", "brownies": "dessert", "watermelon": "melon", "pork": "meat_red", "croutons": "bread", "fish": "fish", "pizza": "pizza", "brown rice": "rice", "tofu": "tofu", "bok choy": "leafy_greens", "quinoa": "grain_other", "grapefruit juice": "fruit_tropical", "yogurt": "yogurt", "green beans": "legumes_green", "wild rice": "rice", "mustard greens": "leafy_greens", "kale": "leafy_greens", "bulgur": "grain_other", "chicken breast": "poultry", "arugula": "leafy_greens", "parmesan cheese": "cheese", "thyme": "herb", "garlic": "herb", "wheat berry": "grain_other", "tortilla chips": "condiment_dressing", "corn": "corn", "sour cream": "leafy_greens", "black beans": "legumes_other", "hominy": "corn", "chicken thighs": "poultry", "chili": "corn", "cheese pizza": "pizza", "beef": "meat_red", "pasta": "pasta", "salmon": "fish", "shallots": "herb", "lemon juice": "herb", "chard": "berry", "zucchini": "squash_eggplant", "lime": "herb", "soy sauce": "grain_other", "ginger": "grain_other", "green onions": "grain_other", "wheat bread": "bread", "feta cheese": "alium_vegetable", "cream": "berry", "millet": "grain_other", "roasted potatoes": "tuber", "white wine": "salad_prepared", "oregano": "salad_prepared", "rosemary": "herb", "basil": "herb", "lettuce": "leafy_greens", "vinaigrette": "berry", "egg whites": "egg_whites", "cottage cheese": "cheese", "cilantro": "herb", "cabbage": "cruciferous", "mayonnaise": "root_vegetable", "jalapenos": "herb", "goat cheese": "cheese", "pecans": "berry", "bread": "bread", "white beans": "legumes_other", "chilaquiles": "bread", "frozen yogurt": "yogurt", "chive": "herb", "snow peas": "legumes_green", "artichokes": "legumes_green", "lentils": "legumes_other", "country rice": "rice", "pilaf": "rice", "jicama": "leafy_greens", "greek yogurt": "cruciferous", "chia seeds": "nuts_seeds", "granola": "berry", "chicken salad": "salad_prepared", "radishes": "vegetable_other", "pumpkin seeds": "nuts_seeds", "caesar dressing": "cheese", "grilled chicken": "poultry", "figs": "fruit_tree", "cream cheese": "cheese", "bagels": "bread", "ground turkey": "poultry", "fried rice": "rice", "salsa": "condiment_dressing", "sandwiches": "bread", "turkey bacon": "poultry", "ketchup": "condiment_dressing", "waffles": "dessert", "spinach (cooked)": "berry", "chayote squash": "squash_eggplant", "rice noodles": "pasta", "pasta salad": "salad_prepared", "celery root": "grain_other", "cookies": "dessert", "tuna": "fish", "brown sugar": "condiment_dressing", "butter": "salad_prepared", "sun dried tomatoes": "tomato", "kimchi": "vegetable_other", "tuna salad": "salad_prepared", "couscous": "grain_other", "cheese": "cheese", "tatsoi": "squash_eggplant", "succotash": "corn", "wine": "tomato", "orange juice": "squash_eggplant"}
GR = sorted(set(GMAP.values())) + ['other']; G = len(GR); gi = {g: i for i, g in enumerate(GR)}
# per-dish group masses + training food table from Nutrition5k ingredient lines
ING = {}
for f in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
    for line in get(f'{B}/metadata/{f}', False).splitlines():
        p = line.strip().split(',')
        if len(p) > 6:
            try: ING[p[0]] = [(p[6 + 7 * j + 1], float(p[6 + 7 * j + 2]), [float(x) for x in p[6 + 7 * j + 4:6 + 7 * j + 7]]) for j in range((len(p) - 6) // 7)]
            except ValueError: pass
YG = np.zeros((len(OK5), G), np.float32); YGM = np.zeros((len(OK5), G, 3))
for n, d in enumerate(OK5):
    for name, g, mac3 in ING.get(d, []):
        k = gi[GMAP.get(name, 'other')]; YG[n, k] += g
        YGM[n, k] += mac3
def set_table(trm):
    global T, Tt, share
    TAB = YGM[trm].sum(0); TM = YG[trm].sum(0); T = TAB / np.maximum(TM, 1e-6)[:, None]; T[TM == 0] = TAB.sum(0) / TM.sum()
    Tt = torch.tensor(T, dtype=torch.float32, device=dev); share = torch.tensor(np.maximum(YG[trm].sum(0) / YG[trm].sum(), 1e-4), dtype=torch.float32)
set_table(~te5)
log('groups', G, 'table fat/carb/prot per g (first 5)', {GR[k]: np.round(T[k], 3).tolist() for k in range(5)})
class IAF(nn.Module):
    def __init__(s, scale, free=False):
        super().__init__(); s.sc = scale; s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1 + G)); s.a = nn.Parameter(torch.log(torch.expm1(torch.tensor([9., 4., 4.]))))
        s.t = nn.Parameter(torch.log(torch.expm1(Tt.clamp_min(1e-4))), requires_grad=free)
        with torch.no_grad(): s.f[-1].bias[1:] = torch.log(share)
    def forward(s, x):
        o = s.f(x); m = Fn.softplus(o[..., 0]) * s.sc[1] / x.shape[1]; q = torch.softmax(o[..., 1:], -1); gm = m[..., None] * q   # B x cells x G
        Mg = gm.sum(1); mac = Mg @ Fn.softplus(s.t); e = mac @ Fn.softplus(s.a)
        return torch.cat([e[:, None], Mg.sum(1, keepdim=True), mac], 1), Mg, gm, m
class PoolG(nn.Module):
    def __init__(s, scale, gsc): super().__init__(); s.sc = torch.cat([scale, gsc]); s.f = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, 5 + G))
    def forward(s, x): o = Fn.softplus(s.f(torch.cat([x.mean(1), x.amax(1)], 1))) * s.sc; return o[:, :5], o[:, 5:], None, None
class SumG(nn.Module):
    def __init__(s, scale, gsc): super().__init__(); s.sc = torch.cat([scale, gsc]); s.f = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5 + G))
    def forward(s, x): mp = Fn.softplus(s.f(x)) * s.sc / x.shape[1]; o = mp.sum(1); return o[:, :5], o[:, 5:], mp[..., 5:], mp[..., 1]
class ANFw(nn.Module):
    def __init__(s, scale): super().__init__(); s.n = ANF(scale); s.a = s.n.a
    def forward(s, x): p, mp = s.n(x); return p, None, None, mp[..., 0]
def build(kind, scale, gsc):
    return {'IAF': lambda: IAF(scale), 'IAF_free': lambda: IAF(scale, True), 'ANF': lambda: ANFw(scale), 'R_poolG': lambda: PoolG(scale, gsc), 'R_sumG': lambda: SumG(scale, gsc)}[kind]().to(dev)
def train(kind, X, Yt, YGt, scale, gsc, itr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = build(kind, scale, gsc); mm = scale[1]
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    for e in range(EP):
        net.train()
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // 64)):
            b = torch.tensor(b, device=dev); p, Mg, _, _ = net(X[b]); loss = ((p - Yt[b]).abs() / scale).sum(1).mean()
            if Mg is not None: loss = loss + ((Mg - YGt[b]).abs() / mm).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval(); return net
def pm(p, y, i): return float(100 * np.abs(p[:, i] - y[:, i]).sum() / y[:, i].sum())
# ---------- evaluation masks (public, CC BY 4.0) ----------
GH = 'https://raw.githubusercontent.com/liamhand/ingredient-level-nutrition-regression/main/data'
rows = list(csv.DictReader(io.StringIO(get(f'{GH}/ingredient_nutrition.csv', False))))
tpos = {d: j for j, d in enumerate([OK5[i] for i in np.where(te5)[0]])}; LAB = {}
for r in rows:
    if r['dish_id'] in tpos: LAB.setdefault(r['dish_id'], {})[int(r['label_value'])] = r['annotation_name']
def getmask(d):
    a = get(f'{GH}/masks/{d}.png')
    return None if a is None else np.asarray(Image.open(io.BytesIO(a)).resize((448, 448), Image.NEAREST))
MASK = {}
with cf.ThreadPoolExecutor(12) as ex:
    for d, m in zip(list(LAB), ex.map(getmask, list(LAB))):
        if m is not None: MASK[d] = m
REG = []   # (test row j, group index, w[256])
for d, m in MASK.items():
    gs = {}
    for lv, name in LAB[d].items():
        if name in gi: gs.setdefault(gi[name], []).append(lv)
    if len(gs) < 2: continue
    for k, lvs in gs.items():
        w = np.isin(m, lvs).astype(np.float32).reshape(16, 28, 16, 28).mean((1, 3)).reshape(256)
        if w.sum() >= 2: REG.append((tpos[d], k, w))
log('annotated test dishes', len(MASK), 'multi-group regions', len(REG), 'dishes', len({r[0] for r in REG}))
def localise(gm, m):   # gm: test x 256 x G (group mass maps), m: test x 256 mass map
    out = []
    for j, k, w in REG:
        g = gm[j, :, k]; out.append((j, float((w * g).sum() / max(g.sum(), 1e-9)), float((w * m[j]).sum() / max(m[j].sum(), 1e-9)), float(w[int(np.argmax(g))] >= 0.5)))
    return out
def bootd(L, n=2000):
    rng = np.random.default_rng(0); ds = sorted({r[0] for r in L}); by = {d: [r[1] - r[2] for r in L if r[0] == d] for d in ds}; o = []
    for _ in range(n):
        pick = rng.choice(ds, len(ds)); v = [x for d in pick for x in by[d]]; o.append(np.mean(v))
    return [round(float(np.percentile(o, 2.5)), 4), round(float(np.percentile(o, 97.5)), 4)]
# ---------- run (Test EC: learning-curve subsets) ----------
CB = cells(IM5, 'base'); Yt = torch.tensor(Yn5, device=dev); YGt = torch.tensor(YG, device=dev); ite = torch.tensor(np.where(te5)[0], device=dev); y5 = Yn5[te5]; yg = YG[te5]
itr_all = np.where(~te5)[0]; RES = {}
for f in (0.10, 0.05):
    ACC = {k: dict(P=[], GM=[], MM=[], MG=[]) for k in KINDS}
    for dr in (200, 201, 202):
        sub = np.random.default_rng(dr).choice(itr_all, max(20, int(round(f * len(itr_all)))), replace=False); trm = np.zeros(len(OK5), bool); trm[sub] = True
        X = pca_fit(CB, trm)(CB); set_table(trm); scale = Yt[torch.tensor(trm, device=dev)].mean(0); gsc = YGt[torch.tensor(trm, device=dev)].mean(0).clamp_min(1e-3)
        for k in KINDS:
            net = train(k, X, Yt, YGt, scale, gsc, sub, dr - 200)
            with torch.no_grad(): outs = [net(X[ite[i:i + 128]]) for i in range(0, len(ite), 128)]
            p = torch.cat([o[0] for o in outs]).cpu().numpy(); A = ACC[k]; A['P'].append(p)
            r = dict(pmae=[pm(p, y5, i) for i in range(5)], factors=Fn.softplus(net.a).tolist())
            if outs[0][1] is not None:
                mg = torch.cat([o[1] for o in outs]).cpu().numpy(); A['MG'].append(mg); r['group_mae_pct'] = float(100 * np.abs(mg - yg).sum(1).mean() / y5[:, 1].mean())
            if outs[0][2] is not None:
                g_ = torch.cat([o[2] for o in outs]).cpu().numpy(); m_ = torch.cat([o[3] for o in outs]).cpu().numpy(); A['GM'].append(g_); A['MM'].append(m_)
                LL = localise(g_, m_); r['H_minus_H0'] = float(np.mean([a[1] - a[2] for a in LL])); r['pointing'] = float(np.mean([a[3] for a in LL]))
            RES[f'f{f}_{k}_d{dr}'] = r; log(f, k, dr, {a: (np.round(b, 3).tolist() if isinstance(b, list) else round(b, 4)) for a, b in r.items()})
        del X; torch.cuda.empty_cache()
    for k in KINDS:
        A = ACC[k]; agg = dict(pmae=np.mean([RES[f'f{f}_{k}_d{d}']['pmae'] for d in (200, 201, 202)], 0).tolist()); agg['macro3'] = float(np.mean(agg['pmae'][2:]))
        if A['GM']:
            LL = localise(np.mean(A['GM'], 0), np.mean(A['MM'], 0)); agg.update(H_minus_H0=float(np.mean([a[1] - a[2] for a in LL])), pointing=float(np.mean([a[3] for a in LL])), ci=bootd(LL))
        if A['MG']: agg['group_mae_pct'] = float(np.mean([RES[f'f{f}_{k}_d{d}']['group_mae_pct'] for d in (200, 201, 202)]))
        RES[f'f{f}_{k}_agg'] = agg; log('AGG', f, k, agg)
    json.dump(RES, open(OUT + '/ec_results.json', 'w'))
# NutritionVerse-Real metadata columns (for a later decision only)
try:
    import zipfile as _z
    with requests.get('https://www.kaggle.com/api/v1/datasets/download/nutritionverse/nutritionverse-real', stream=True, timeout=600) as r_:
        r_.raise_for_status(); zp = W + '/nvr.zip'
        with open(zp, 'wb') as fh:
            for ch in r_.iter_content(1 << 20): fh.write(ch)
    zz = _z.ZipFile(zp); names = [n for n in zz.namelist() if n.endswith('.csv')]; log('NVR csv files', names)
    for n in names: log('NVR columns', n, zz.open(n).readline().decode('utf-8', 'ignore')[:1500])
except Exception as e_: log('NVR check failed', repr(e_))
log('DONE minutes', round((time.time() - T0) / 60, 1))
