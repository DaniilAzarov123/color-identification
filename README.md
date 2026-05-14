# Visual Working Memory: Model Fitting

Computational modeling of the complete-identification task from **Nosofsky & Gold (2018)**.
Subjects studied a memory set of colored items and identified which color appeared in the set
from a 16-item response wheel. Four positions were designated high-payoff (+10 pts correct vs +1 pt for others).

---

## Table of Contents

- [Models](#models)
  - [SCM](#similarity-choice-model-scm)
  - [SCM-mix](#scm-mixture-scm-mix)
  - [TCC](#target-confusability-competition-tcc)
  - [TCC-mix](#tcc-mixture-tcc-mix)
- [MDS Coordinates](#mds-coordinates)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results-all-8-subjects)
- [Summary](#summary)
- [References](#references)

---

## Models

### Similarity-Choice Model (SCM)

Choice probability is proportional to the similarity between a studied item and each response option, weighted by response biases:

$$
p_{ij} = \frac{b_j \cdot s_{ij}}{\sum_k b_k \cdot s_{ik}}, \quad s_{ij} = e^{-c \cdot d_{ij}}, \quad b_j^{(HP)} = B \cdot b_j
$$

Parameters: 15 free biases + 1 determined, boost $B$ for high-payoff items, $c$ per set size — **19 total**.

### SCM-Mixture (SCM-mix)

Adds a discrete guessing state. When memory fails, subjects guess according to a sharpened bias distribution:

$$
P(R_j | S_i) = p_{mem} \cdot p_{ij} + (1 - p_{mem}) \cdot g_j, \quad g_j = \frac{b_j^\gamma}{\sum_k b_k^\gamma}
$$

Parameters: SCM params + $p_{mem}$ per set size + $\gamma$ — **23 total**.

### Target Confusability Competition (TCC)

The main idea is taken from **Schurgin et al. (2020)**. Each response item $j$ has its own familiarity distribution with mean:

$$
\mu_{ij} = d'^{(ss)} \cdot s_{ij} + b_j, \quad \mu_{ij}^{(HP)} = \mu_{ij} + B, \quad d'^{(ss)} = t^{(ss)} \cdot d'^{(ss=2)}
$$

The probability of choosing $j$ given studied item $i$ is:

$$
p_{ij} = \int_{-\infty}^{+\infty} \varphi(x - \mu_{ij}) \cdot \prod_{k \neq j} \Phi(x - \mu_{ik}) \cdot dx
$$

Parameters: $d'$ base + 2 scales, $c$, 16 biases, high-payoff boost (*B*) — **21 total**.

### TCC-Mixture (TCC-mix)

Combines TCC memory state with a softmax-based guessing state:

$$
P(R_j | S_i) = p_{mem} \cdot p_{ij}^{TCC} + (1 - p_{mem}) \cdot g_j, \quad g_j = \text{softmax}(b)^\gamma
$$

where the bias for the high-payoff color is $b_j^{(HP)} = b_j + B$.

Parameters: TCC params + $p_{mem}$ per set size + $\gamma$ — **25 total**.

---

## MDS Coordinates

Distances between stimuli $d_{ij}$ used in the similarity function can be handled in two ways, controlled by `USE_SINGLE_MDS_SOLUTION` in the config:

- **Fixed** (`True`): all subjects share the same distances derived from a unit circle (16 stimuli equally spaced) — faster, fewer parameters
- **Free** (`False`): each subject's MDS coordinates are fitted as free parameters
  (32 extra params per subject) alongside the model parameters — slower, more flexible

---

## Repo Structure

```
├── models.py          # Model functions: SCM, SCM-mix, TCC, TCC-mix
│                      # MDS utilities: coords_to_distances, init_coords_circle, get_distances
├── utils.py           # Data loading, fitting, metrics, plotting
│   ├── Data loading:  load_data, extract_observed, prepare_data
│   ├── Fitting:       validity_checks, fit_model
│   ├── Metrics:       compute_aic_bic, print_model_comparison, compute_correlations
│   └── Plotting:      plot_predicted_vs_observed, plot_predicted_vs_observed_comparison,
│                      plot_aic_bic, plot_mds_solutions
├── analysis.ipynb     # Main analysis notebook
├── Data/              # Raw subject data
├── results/           # Saved fits, predictions, plots
│   ├── Fixed_MDS/     # Results with fixed unit circle MDS
│   └── Free_MDS/      # Results with subject-specific MDS coordinates
└── README.md
```

---

## Setup

```bash
pip install numpy scipy matplotlib pandas colormath
```

---

## Usage

Open `analysis.ipynb`. At the top of the notebook, adjust the config:

```python
FIT_SUBJECTS            = None    # None = all 8 subjects, or e.g. [2, 4]
USE_SINGLE_MDS_SOLUTION = True    # True = fixed circle, False = free per-subject MDS
SAVE_RESULTS            = False   # True = save pickle + text + plots to results/
READ_PARAMS             = True   # True = load previously saved fit instead of refitting
N_POINTS                = 100     # TCC integration points (100 is fast, 500 is precise)
X_RANGE                 = (-3, 9) # TCC integration range
```

The notebook then:

1. Loads raw data from `Data/`
2. Fits all four models via two-stage L-BFGS-B (warmup + final)
3. Computes AIC/BIC and per-condition correlations
4. Generates predicted vs observed plots and MDS solution plots
5. Optionally saves everything to `results/Fixed_MDS/` or `results/Free_MDS/`

---

## Results

### Fixed MDS (unit circle)

| Model   | n free params | NLL       | AIC                 | BIC                 |
| ------- | ------------- | --------- | ------------------- | ------------------- |
| SCM     | 19            | 43,162.77 | 86,363.54           | 86,521.13           |
| SCM-mix | 23            | 40,805.75 | **81,657.51** | **81,848.28** |
| TCC     | 21            | 41,129.66 | 82,301.32           | 82,475.50           |
| TCC-mix | 25            | 41,120.27 | 82,290.55           | 82,497.91           |

SCM-mix wins by both AIC and BIC. TCC-mix demonstrates similar results

### Free MDS (subject-specific coordinates, 32 extra params per subject)

| Model   | n free params | NLL       | AIC                 | BIC                 |
| ------- | ------------- | --------- | ------------------- | ------------------- |
| SCM     | 275           | 38,840.66 | 78,231.32           | 80,512.29           |
| SCM-mix | 279           | 38,388.52 | 77,335.03           | **79,649.18** |
| TCC     | 277           | 38,574.04 | 77,702.08           | 79,999.64           |
| TCC-mix | 281           | 38,384.07 | **77,330.14** | 79,660.88           |

TCC-mix wins by AIC, SCM-mix wins by BIC. The difference between them is negligible
(ΔAIC: 4.89, ΔBIC: 11.70).

Fitting subject-specific MDS coordinates **improves AIC and BIC for all models** despite
the large parameter penalty (256 extra parameters across 8 subjects). For example,
SCM-mix improves from AIC=81,657 → 77,335 and BIC=81,848 → 79,649 — reductions of ~4,300
and ~2,200 points respectively. This confirms that individual differences in perceptual
similarity structure are real and worth capturing, even after penalizing for complexity.

---

## Summary

Across both MDS conditions, **mixture models substantially outperform simple models**,
meaning a guessing state is necessary to explain the data. The gap between SCM-mix and
TCC-mix is small and changes depending on the complexity penalty (AIC vs BIC), so the
key finding is the mixture structure itself rather than which specific memory mechanism
is used. Fitting individual MDS solutions further improves all models, suggesting
meaningful subject-level variation in perceptual similarity.

However, these results should be interpreted cautiously. The payoff structure of the
experiment — high-payoff correct responses earning 10× more than low-payoff — creates
a strongly rational incentive to guess the high-payoff item whenever memory is at least
a little uncertain. The guessing signature the mixture models capture may therefore
reflect **decision strategy under asymmetric payoffs** rather than a fundamental
property of memory. A more balanced payoff scheme would be needed to disentangle these
accounts. TCC-mix is particularly well positioned to test this: $d'$ should remain
stable across payoff conditions while $p_{mem}$ and the guessing bias should respond
to the manipulation.

---

## References

Nosofsky, R. M., & Gold, J. M. (2018). Biased guessing in a complete-identification
visual-working-memory task: Further evidence for mixed-state models.
*Journal of Experimental Psychology: Human Perception and Performance, 44*(4), 603.

Schurgin, M. W., Wixted, J. T., & Brady, T. F. (2020). Psychophysical scaling reveals
a unified theory of visual memory strength. *Nature Human Behaviour, 4*(11), 1156–1172.
