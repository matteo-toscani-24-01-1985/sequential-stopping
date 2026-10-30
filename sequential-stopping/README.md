# Sequential stopping: blind-review simulation evidence package

**Scope:** Independent and paired *t*-tests; three-level independent and repeated-measures one-way ANOVAs; prespecified 2 × 2 and 2 × 2 × 2 interactions in independent and repeated-measures designs; and the assumption-violation checks presented in the manuscript.

**Credit requested by the author:** Simulation work written by **MT with ChatGPT support**. This reviewer package was **polished and organised for blind review with ChatGPT assistance**. Names, affiliations, email addresses, and manuscript drafts are deliberately omitted. The initials “MT” are retained at the author's request; remove this line if the journal requires fully anonymous materials.

## Start here

1. The `01_t_tests/`, `02_one_way_ANOVAs/`, and `03_factorial_interactions/` folders contain the reported figure images and available underlying numerical tables from the eight principal designs. Results include all 12 evidence thresholds and, where applicable, the 80% and 95% persistence targets.
2. The `04_robustness/` folder contains the reported small-sample assumption-violation figure, underlying results for all 12 evidence thresholds, and the Python simulation scripts that generated the targeted checks.
3. `CORE_SOURCE_MANIFEST.csv` maps each of the eight principal designs to its original MATLAB simulation program and full results directory, so the corresponding original simulation sources can be inserted without ambiguity.
4. `SHA256SUMS.txt` enables checking that the reviewer files have not changed.

## Methodological conventions

- Evidence criteria: p < .05, .01, .001, .0001; BF10 >= 3, 10, 30, 100; and four design-specific BF10 thresholds numerically matched to the p thresholds at n=50.
- Calibration targets: 80% and 95% persistence over the subsequent 1,000 observations. (In independent designs, N is per group/cell; in repeated designs, N is participants.)
- Minimum-N stopping rules were derived from training/calibration simulations and assessed using separate independent null simulations. For full observed-effect rules, the effect-size bin is recomputed at each eligible look. Its overall false-positive rate pertains to the full adaptive rule, not a single effect-size bin.
- The prespecified false-positive criterion is a **one-sided 95% Wilson upper confidence bound < 5%**, not simply a raw estimate below 5%.
- Core null validation in the reported results involved up to 800,000 trajectories for t-tests/one-way ANOVAs and 200,000 for factorial interaction studies.
- MATLAB R2026a is the manuscript's recorded software version. Python, NumPy, SciPy, pandas, and matplotlib are used for the targeted robustness checks and figure assembly.

## Important status of this copy

**This ZIP is a curated reviewer evidence bundle, not yet a fully self-contained raw-simulation reproduction archive.** The original eight MATLAB calibration engines and their complete source checkpoint MAT files are available in the project workspace, but they were not present as local source files for this archive build. Their **exact filenames and result folder names** are documented in `CORE_SOURCE_MANIFEST.csv`, and must be copied in before the package is described as a complete independent rerun archive. Saved paper-level results and figure data are included.

The `04_robustness` null checks used **24,000 trajectories per condition** in the current manuscript figure and should be identified as exploratory relative to the higher-replication main studies. Their saved false-positive estimates do **not** establish that 95% evidence persistence itself holds under all violated assumptions. Core MATLAB source and larger-simulation verification should be added before publication.

No human participant data, draft manuscripts, correspondence, unused study designs, missing-data or clustering scenarios, or unequal-group-size explorations are included.

## Reproduce the targeted robustness figure

From the unpacked archive:

```bash
python 04_robustness/reproduce_figure_from_results.py
```

To rerun the *focused null simulation* with 24,000 trajectories per design/distribution (a substantially more computationally expensive operation):

```bash
cd 04_robustness/simulation_code
python focused_all_thresholds_simulation.py 24000
python focused_extra_violations.py
```

The simulation scripts use fixed random seeds. They need Python packages `numpy`, `pandas`, and `scipy` and read the archived original minimum-N tables. The precomputed Figure 13 source-data file remains the authoritative dataset for the displayed figure, because rerunning random simulations can produce small sampling differences if code/library versions change.

## Blind-review safeguards

This archive excludes manuscript documents, Dropbox links, directory IDs and identifying file metadata. Reviewers should verify the design-specific minimum-N rules against the saved CSVs. The paper's wording should distinguish persistence calibration, sequential null error control, and sensitivity analyses.
