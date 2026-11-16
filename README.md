### Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run this from the repo root:

```bash
uv sync
```

Leave `RUN = False` to inspect the saved results. No GPU or model downloads are needed. Stuff is saved in `artifacts` dir.

### Overview

* Is the Jacobian as a model relevant or can identity and random matrices give a similar concept–coherence trade-off? See [notebook 1](01_qwen3_model_ablation.ipynb).
* Can we keep more coherence at a given concept rate by canceling drift orthogonal to the concept direction? See [notebook 2](02_drift_lqr_ablation.ipynb).

### Dynamics models

LQR needs a model of how a small change in activations passes through each transformer block. We try three choices:

* **Mean Jacobian:** Measure how small changes at a block's input affect its output, then average across prompts. This is the original A-LQR choice.
* **Identity:** Assume the change passes through the block unchanged.
* **Random:** Start with identity and add random noise, scaled to match the mean Jacobian's distance from identity.

### Measuring concept and coherence

We give [Granite Guardian 4.1-8B](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b) the prompt and completion, then ask two separate yes/no questions:

* **Concept:** Does the completion meaningfully include the topic? Paraphrases count; mentioning it in passing does not.
* **Coherence:** Is it readable and easy to follow? Minor grammar mistakes and cut-off endings are fine. We ask the judge to ignore factual accuracy and whether the text is on topic.

Each rate is the fraction of completions marked “yes”. We plot concept rate on the x-axis and coherence rate on the y-axis. Error bars show ±1 standard error (stdev/sqrt(n))

The judging code is in [`extensions/judge.py`](extensions/judge.py): `concept_criterion` and `COHERENCE` contain the exact criteria, and `build_judge` handles batched scoring. In [`extensions/experiments.py`](extensions/experiments.py), `judge_run` scores the saved completions and `summarize` calculates the rates.

### Notebook 1

In [notebook 1](01_qwen3_model_ablation.ipynb), we swap the dynamics matrix and compare how well A-LQR steers. We use the same concept direction for all three operators, steer every block.

There are three saved runs:
* football on Qwen3-4B,
* dog on Qwen3-4B,
* dog on Qwen3-8B.

Other details: 

* `R = 0.1, 1, 10`, with `Q = QF = 0.1`. 
* Steering strength `λ = (0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.)`.
* 100 separately sampled continuations of “Once upon a time” per setting, plus an unsteered baseline.

The first plot shows concept versus coherence for each saved run (rows) and each `R` (columns). The heatmaps show how hard the controller pushes at each layer, with one figure per run.

Each point is the average over 100 completions. The whiskers are ±1 standard error.

**Takeaways**

* Swapping the operator doesn't really change the concept–coherence trade-off.
* Mean Jacobian usually gets there with the least push energy, and the gap widens as `R` goes up.

### Notebook 2

In [notebook 2](02_drift_lqr_ablation.ipynb), we add drift cancellation and check whether it keeps coherence higher at the same concept rate.

#### What is drift cancellation?

Original A-LQR steers by adding pushes toward the concept. Drift cancellation, at each steered block, also measures how far the activation has moved from where it would be without steering. To do so, it runs a second, unsteered copy of the model and computes the difference between the steered and unsteered activations. Then it removes the part along the concept direction. What remains is the "drift". The controller adds a correction pushing against the drift, using the same precomputed gain. `w` sets how strongly it pushes: `w = 0` is original A-LQR, and it is compared with `w = 2`.

At block $\ell$, with $x_\ell$ the last token's activation entering the block, $v_\ell = E_\ell / \|E_\ell\|$ the unit concept direction, and $K_\ell$ the LQR gain, original A-LQR adds

$$u_\ell = K_\ell\,\alpha_\ell v_\ell, \qquad \alpha_\ell = \lambda\|E_\ell\| - v_\ell^\top x_\ell.$$

With drift cancellation, $x_\ell^{\text{unsteered}}$ is the same activation in the unsteered copy, and

$$d_\ell = x_\ell - x_\ell^{\text{unsteered}}, \qquad d_\ell^{\perp} = d_\ell - (v_\ell^\top d_\ell)\,v_\ell, \qquad u_\ell = K_\ell\left(\alpha_\ell v_\ell - w\,d_\ell^{\perp}\right).$$

There is one saved run: football on Qwen3-4B.

Other details:

* Identity, Random, and Mean Jacobian operators.
* Drift cancellation off (`w = 0`, original A-LQR) or on (`w = 2`, with drift cancellation).
* Steering all blocks, the last 8, or the last 4.
* `R = 1, 10`, with `Q = QF = 0.1`.
* Steering strength `λ = (1, 2, 3, 4, 5, 6, 7, 8)`.
* 1,000 separately sampled continuations of “Once upon a time” per setting, plus an unsteered baseline.

The first set of plots shows concept versus coherence with cancellation off and on, for each choice of layers (rows) and each operator (columns). There is one figure per `R`.

The second set of plots shows where the steering change ends up. At the last layer (just before the final normalization), let $d = x - x^{\text{unsteered}}$ and split it into a part along the concept direction $v$ and the rest, $d^{\perp}$, as above. The x-axis is the size of the change, $\|d\| / \|x^{\text{unsteered}}\|$. The y-axis is the off-concept share, $\|d^{\perp}\|^2 / \|d\|^2$: 0% means all the change is along $v$, 100% means none of it is. Both are averaged over generated tokens. Dot color shows coherence and dot size shows concept rate.

Each point is the average over 1,000 completions. The whiskers (not really visible) are ±1 standard error.

**Takeaways**

* At the same `λ`, drift cancellation raises coherence but can lower concept, mostly when steering all layers. So `λ` values aren't comparable between `w = 0` and `w = 2`.
* At the same concept rate (75% or 90%), cancellation either helps or makes no difference. The gain is large when steering all layers (up to +46 points of coherence) and negligible when steering the last 8 or 4.