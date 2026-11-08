# Maps Nutrition5k ingredient names -> the 40 groups of the public ingredient-mask set (liamhand, CC BY 4.0)
# by matching per-dish masses. Usage: py -3.10 map_ingredients.py <dir with dish_metadata_cafe*.csv, ingr.csv, *_ids.txt>
import csv, sys, os, collections, json
D = sys.argv[1]; H = os.path.dirname(os.path.abspath(__file__))
N5 = {}
for f in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
    for line in open(os.path.join(D, f)):
        p = line.strip().split(',')
        if len(p) > 6:
            ing = [(p[6 + 7 * j + 1], float(p[6 + 7 * j + 2]), [float(x) for x in p[6 + 7 * j + 3:6 + 7 * j + 7]]) for j in range((len(p) - 6) // 7)]
            N5[p[0]] = ing
A = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, 'ingr.csv'))): A[r['dish_id']].append((r['annotation_name'], float(r['mass'] or 0)))
vote = collections.defaultdict(collections.Counter); matched = tot = 0
for d, anns in A.items():
    if d not in N5: continue
    for name, g, _ in N5[d]:
        tot += 1; best = min(anns, key=lambda a: abs(a[1] - g)) if anns else None
        if best and abs(best[1] - g) <= max(1.0, 0.02 * g): vote[name][best[0]] += 1; matched += 1
MAP = {n: c.most_common(1)[0][0] for n, c in vote.items()}
purity = sum(c.most_common(1)[0][1] for c in vote.values()) / max(1, sum(sum(c.values()) for c in vote.values()))
tr = set(open(os.path.join(D, 'train_ids.txt')).read().split()); te = set(open(os.path.join(D, 'test_ids.txt')).read().split())
allnames = collections.Counter(); covered = collections.Counter()
for d, ing in N5.items():
    if d not in tr: continue
    for n, g, _ in ing: allnames['all'] += g; covered['all'] += g * (n in MAP)
print('ingredient entries in annotated dishes', tot, 'matched by mass', matched, 'names mapped', len(MAP), 'of', len({n for v in N5.values() for n, _, _ in v}), 'vote purity', round(purity, 3))
print('train-split grams covered by mapped names', round(covered['all'] / allnames['all'], 3))
print('annotated dishes in official test split', sum(d in te for d in A), 'train', sum(d in tr for d in A))
json.dump(MAP, open(os.path.join(H, '..', 'data', 'ingredient_group_map.json'), 'w'), indent=0)
