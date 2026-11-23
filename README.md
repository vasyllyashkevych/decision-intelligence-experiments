# Decision-intelligence experiments (chapter companion code)

Companion code for the chapter *From Software Analytics to Business Value* (Lyashkevych V., Lyashkevych M., Zomchak L.).
Tests three analytical-layer claims of the framework on open multi-release defect data. **No LLM is used and no business-value claim is made.**

```
python3 -I src/get_data.py          # downloads 40 PROMISE/Jureczko release files (not redistributed), writes SHA-256 manifest
python3 -I src/run_experiments.py   # E1-E3 -> results/*.csv, results/summary.json
python3 -I src/make_figures.py      # figures/*.png
```
Design, hypotheses and the "not claimed" list are in `PREREGISTRATION.md` (written before the first run). One analysis,
`EXPLORATORY_observable_proxy`, was added after seeing E1 and is labelled as exploratory.
Requirements: Python 3.10+, numpy, pandas, scipy, scikit-learn, matplotlib. Seed 20261123.

Data: Jureczko & Madeyski (2010), PROMISE repository, via https://github.com/klainfo/DefectData (check the licence/terms of the original repository before redistributing).
Caveats: pairs from one project share classes and are not independent; label = bug>0 on Java classes of 11 Apache/other projects (2000s releases);
a single training release per pair; default hyper-parameters; costs in E1 are an abstract ratio b:c, not money.
