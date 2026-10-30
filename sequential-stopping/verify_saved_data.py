from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
assert (ROOT/'CORE_SOURCE_MANIFEST.csv').exists()
assert len(pd.read_csv(ROOT/'CORE_SOURCE_MANIFEST.csv'))==8
r=pd.read_csv(ROOT/'04_robustness/results/Figure_13_robustness_all12_source_data.csv')
assert len(r)==312,(len(r),'expected 312 design-condition-threshold observations')
assert set(r['criterion_index'])==set(range(12))
assert not r.duplicated(['code','distribution','criterion_index']).any()
assert (r.groupby(['code','distribution']).size()==12).all()
a=pd.read_csv(ROOT/'02_one_way_ANOVAs/results/ANOVA_all_persistence_and_FPR_source.csv')
assert set(a['design'])=={'ind','rm'}
f=pd.read_csv(ROOT/'03_factorial_interactions/results/Factorial_ALL_validated_stopping_rules_with_partial_eta_squared.csv')
assert len(f)>200
print('Saved data structure checks passed. This does not verify full original calibration code, which is not included.')
