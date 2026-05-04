# Visual Working Memory: Model Fitting

Computational modeling of the complete-identification task from  **Nosofsky & Gold (2018)** .
Subjects studied a memory set of colored items and identified which color appeared in the set from a 16-item response wheel. Four positions were designated high-payoff (+10 pts correct vs +1 pt for others).

---

## Models

### Similarity-Choice Model (SCM)

Choice probability is proportional to the similarity between a studied item and each response option, weighted by response biases:

$$
p_{ij} = \frac{b_j \cdot s_{ij}}{\sum_k b_k \cdot s_{ik}}, \quad s_{ij} = e^{-c \cdot d_{ij}}, \quad b_j^{(HP)} = B \cdot b_j
$$

Parameters: 15 free biases + 1 determined, boost $B$ for high-payoff items, $c$ per set size —  **19 total** .

### SCM-Mixture (SCM-mix)

Adds a discrete guessing state. When memory fails, subjects guess according to a sharpened bias distribution:

$$
P(R_j | S_i) = p_{mem} \cdot p_{ij} + (1 - p_{mem}) \cdot g_j, \quad g_j = \frac{b_j^\gamma}{\sum_k b_k^\gamma}
$$

Parameters: SCM params + $p_{mem}$ per set size + $\gamma$ —  **23 total** .

### Target Confusability Competition (TCC)

The main idea for the model is taken from **Schurgin et al. (2020)**. Each response item $j$ has its own familiarity distribution with mean

$$
\mu_{ij} = d'^{(ss)} \cdot s_{ij} + b_j, \quad \mu_{ij}^{(HP)} = \mu_{ij} + B, \quad d'^{(ss)} = t^{(ss)} \cdot d'^{(ss=2)}
$$

 The probability of choosing $j$ given studied item $i$ is:

$$
p_{ij} = \int_{-\infty}^{+\infty} \varphi(x - \mu_{ij}) \cdot \prod_{k = 1, k \neq j}^J \Phi(x - \mu_{ik}) \cdot dx
$$

Parameters: $d'$ base + 2 scales *t*, $c$, 16 biases, HP boost —  **21 total** .

### TCC-Mixture (TCC-mix)

Combines TCC memory state with a softmax-based guessing state:

$$
P(R_j | S_i) = p_{mem} \cdot p_{ij} + (1 - p_{mem}) \cdot g_j, \quad g_j = softmax(b_j)^\gamma
$$

note that the bias for the high-payoff color is $b_j^{(HP)} = b_j + B$.

Parameters: TCC params + $p_{mem}$ per set size + $\gamma$ —  **25 total** .

---

## Results (Subject 4)

| Model   | n params | NLL  | AIC            | BIC            |
| ------- | -------- | ---- | -------------- | -------------- |
| SCM     | 19       | 5121 | 10281          | 10399          |
| SCM-mix | 23       | 4634 | **9314** | **9458** |
| TCC     | 21       | 4752 | 9546           | 9677           |
| TCC-mix | 25       | 4639 | 9327           | 9484           |

SCM-mix wins by both AIC and BIC. The mixture component is critical — adding a guessing state substantially improves fit for both model families.

---

## Repo Structure

```
├── models.py        # SCM, SCM-mix, TCC, TCC-mix model functions
├── utils.py         # Data loading, fitting, metrics, plotting
├── analysis.ipynb   # Main analysis notebook
├── Data/            # Raw subject data
├── MDS_results/     # Individual MDS coordinates
└── README.md
```

---

## Setup

```bash
pip install numpy scipy matplotlib pandas colormath
```

---

## Usage

Open `analysis.ipynb`. The notebook:

1. Loads raw data from `Data/`
2. Attaches MDS distances (perfect circle or subject-specific)
3. Fits all four models via L-BFGS-B
4. Computes AIC/BIC and per-condition correlations
5. Generates predicted vs observed plots

To fit all subjects, set `FIT_SUBJECTS = None` in the config cell. To use subject-specific MDS solutions, set `USE_SINGLE_MDS_SOLUTION = False` (requires files in `MDS_results/`).

---

## Reference

Nosofsky, R. M., & Gold, J. M. (2018). Biased guessing in a complete-identification visual-working-memory task: Further evidence for mixed-state models.  *Journal of Experimental Psychology: Human Perception and Performance, 44* (4), 603.

Schurgin, M. W., Wixted, J. T., & Brady, T. F. (2020). Psychophysical scaling reveals a unified theory of visual memory strength.  *Nature human behaviour*, *4* (11), 1156-1172.
