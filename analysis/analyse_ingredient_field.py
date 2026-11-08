# Summarises 06_ingredient_field.py output. Usage: python analyse_ingredient_field.py eb_results.json
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); R = json.load(open(sys.argv[1]))
I, A = R['IAF_agg'], R['ANF_agg']
h1 = I['H_minus_H0'] >= 0.15 and I['ci'][0] > 0
h2 = I['pmae'][0] <= A['pmae'][0] + 1.0 and I['macro3'] <= A['macro3'] + 1.0
out = dict(H1=dict(H_minus_H0=I['H_minus_H0'], ci=I['ci'], PASS=h1), H2=dict(iaf_kcal=I['pmae'][0], anf_kcal=A['pmae'][0], iaf_macro3=I['macro3'], anf_macro3=A['macro3'], PASS=h2),
           TEST_PASS=bool(h1 and h2), models={k[:-4]: v for k, v in R.items() if k.endswith('_agg')})
json.dump(out, open(os.path.join(H, '..', 'results', 'ingredient_field_summary.json'), 'w'), indent=1)
for k, v in out['models'].items():
    print(f"{k:9s} kcal {v['pmae'][0]:.2f} mass {v['pmae'][1]:.2f} fat {v['pmae'][2]:.2f} carb {v['pmae'][3]:.2f} prot {v['pmae'][4]:.2f} macro3 {v['macro3']:.2f}",
          *(f"H-H0 {v['H_minus_H0']:.3f} ci {v['ci']} pointing {v['pointing']:.3f}",) if 'H' in v else (), *(f"groupMAE {v['group_mae_pct']:.1f}%",) if 'group_mae_pct' in v else ())
print('H1', h1, 'H2', h2, 'TEST', 'PASS' if out['TEST_PASS'] else 'FAIL')
