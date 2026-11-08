# Rebuilds the test-row order used by 06_ingredient_field.py (official test ids with labels and an overhead RGB image).
import csv, os, sys, json, concurrent.futures as cf, urllib.request
D = sys.argv[1]; H = os.path.dirname(os.path.abspath(__file__))
B = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset'
Y5 = {}
for f in ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv'):
    for line in open(os.path.join(D, f)):
        p = line.strip().split(',')
        if len(p) > 6:
            try: Y5[p[0]] = [float(x) for x in p[1:6]]
            except ValueError: pass
te = open(os.path.join(D, 'test_ids.txt')).read().split()
cand = [d for d in te if d in Y5 and Y5[d][0] > 0]
def ok(d):
    try: urllib.request.urlopen(urllib.request.Request(f'{B}/imagery/realsense_overhead/{d}/rgb.png', method='HEAD'), timeout=30); return True
    except Exception: return False
with cf.ThreadPoolExecutor(16) as ex: flags = list(ex.map(ok, cand))
order = [d for d, f in zip(cand, flags) if f]
print('candidates', len(cand), 'with image', len(order))
json.dump(order, open(os.path.join(H, '..', 'results', 'IAF', 'test_order.json'), 'w'))
