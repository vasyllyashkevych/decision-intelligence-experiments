# Pre-registered design (written before the first run)

Data: PROMISE/Jureczko multi-release Java defect data (11 projects, 40 releases, 29 consecutive release pairs r -> r+1).
Unit: class. Label: bug > 0. Features: 20 CK/OO metrics (wmc ... avg_cc), log1p-transformed, standardised on the training release.
All models are trained on release r only and evaluated on release r+1 of the same project (a release-planning situation: evidence
from the current release, outcome observed in the next one). Seed 20261123. No result is removed after the first run.

## E1. Do evidence-layer metrics transfer to decision-layer value?
Models: logistic regression (LR), random forest (RF), LOC-rank baseline. Evidence layer: AUC, Brier score, ECE (10 bins).
Decision layer: an action "inspect class" costs c = 1 and finds a defect with value b (b in {2, 3, 5, 10, 20}).
A calibrated model should inspect when p >= c/(b+c). Net benefit per class NB = (b*TP - n_inspected) / N.
Reference policies: inspect-all, inspect-none, oracle threshold (best threshold chosen on the test release, an unattainable bound).
Hypotheses (directional, stated in advance):
 H1a. Prevalence shift between r and r+1 degrades calibration (ECE rises with |prev_{r+1} - prev_r|).
 H1b. In a non-trivial share of (pair, b) cases the thresholded model policy does not beat the best of the two trivial policies.
 H1c. In a non-trivial share of cases the ordering of LR vs RF by AUC disagrees with their ordering by NB.

## E2. Compensatory vs non-compensatory aggregation (package level)
Class risk for release r: 5-fold out-of-fold LR probabilities within release r. Package = class name without last segment.
Aggregators: mean (compensatory), max, mean of the top-3 classes, share of classes with p >= 0.5.
Outcome: share of defective classes of the same package in r+1 (packages with >= 3 classes in both releases; >= 8 packages per pair).
Metric: Spearman rho per pair; paired comparison across pairs (Wilcoxon signed-rank, bootstrap CI of mean difference).
Hypothesis H2: an aggregator that does not average away the worst classes (max/top-3) ranks packages at least as well as the mean.
A null or reversed result will be reported as such.

## E3. Stability of the weighted utility rubric
Actions: inspect the top q of classes by LR risk, q in {0.05, 0.10, 0.20, 0.30, 0.50, 0.75}. Attributes: defect recall,
effort saved (1 - share of LOC inspected), precision. U = sum w_i v_i with w ~ Dirichlet(1,1,1), 2000 draws per pair.
Measures: probability that the winning action differs from the winner under equal weights; mean Kendall tau of the ranking vs the equal-weight ranking;
same measures with a hard constraint (recall >= 0.5) applied before ranking.
Hypothesis H3: the winner is weight-sensitive in a majority of pairs, and a hard constraint reduces the set of admissible winners.

## Not claimed
No LLM is used. These experiments test the analytical/decision layers of the framework on software-analytics evidence only.
They are not evidence about GenAI productivity or business value.
