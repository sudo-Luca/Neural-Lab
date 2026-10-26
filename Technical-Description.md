# Neural Lab v13

> **A self-contained, pedagogical deep learning simulator built entirely in a single HTML file.**

Neural Lab v13 is an interactive browser-based tool for visualising, training, and understanding multilayer perceptrons (MLPs), with four companion labs for convolutional networks, recurrent networks, Transformers and reinforcement learning. It requires no server, no framework, and no installation — just open the HTML file in any modern browser. Every forward pass, backpropagation step, gradient computation, and weight update is performed in vanilla JavaScript (optionally on the GPU through WebGPU) and rendered in real time on an HTML5 Canvas, making it ideal for learning, teaching, or experimenting with neural network fundamentals.

---

## Table of Contents

1. [Overview](#overview)
2. [Interface Layout](#interface-layout)
3. [Architecture & Global State](#architecture--global-state)
4. [Activation Functions](#activation-functions)
5. [Loss Functions](#loss-functions)
6. [Weight Initialisation](#weight-initialisation)
7. [Core Neural Network Functions](#core-neural-network-functions)
8. [Optimiser Engine](#optimiser-engine)
9. [Training Techniques (Batch Norm, Augmentation, Early Stopping)](#training-techniques-batch-norm-augmentation-early-stopping)
10. [Training Control Functions](#training-control-functions)
11. [GPU Training & Performance (WebGPU)](#gpu-training--performance-webgpu)
12. [Visualisation & Canvas Rendering](#visualisation--canvas-rendering)
13. [Dataset Management](#dataset-management)
14. [Right-Panel Tabs & Inspection Tools](#right-panel-tabs--inspection-tools)
15. [Calculation Mode (Individual ⇄ Matrix)](#calculation-mode-individual--matrix)
16. [Manual Calculation Mode (« Manuel »)](#manual-calculation-mode--manuel-)
17. [Quiz Tab](#quiz-tab)
18. [Step-by-Step Explainer](#step-by-step-explainer)
19. [Extension Features (v12 block)](#extension-features-v12-block)
20. [CNN Lab (Convolutional Networks)](#cnn-lab-convolutional-networks)
21. [RNN Lab (Recurrent Networks)](#rnn-lab-recurrent-networks)
22. [Transformer Lab](#transformer-lab)
23. [RL Lab](#rl-lab)
24. [Code & Data Export](#code--data-export)
25. [Automatic FR → EN Translation Layer](#automatic-fr--en-translation-layer)
26. [UI Utility Functions](#ui-utility-functions)
27. [Keyboard Shortcuts](#keyboard-shortcuts)
28. [Mobile Support](#mobile-support)
29. [Local Dev](#local-dev)
30. [Colour Palette Reference](#colour-palette-reference)
31. [Dependencies](#dependencies)

---

## Overview

Neural Lab v13 is structured as a **single HTML document** (~30,000 lines) combining:

- **HTML** — three resizable panels (left config, centre canvas, right details) plus several modal dialogs (Wizard, Advanced Builder, decision boundary, visual dataset editor, Save/Load slots, optimiser race, network comparison, CNN Lab, RNN Lab, Transformer Lab, RL Lab, node/line detail popovers).
- **CSS** — a custom dark-mode design system using CSS variables, with several selectable colour themes, an additive "Aurora" visual layer, and a dedicated mobile readability layer (`@media (max-width: 900px)` only).
- **Vanilla JavaScript** — all neural network logic, rendering, and interactivity, split across **eleven inline `<script>` blocks**:
  1. **Main core** (≈ lines 8291–19741) — architecture, activations, losses, optimisers, Batch Norm / augmentation / early stopping, the WebGPU training engine, training loop, canvas rendering, quiz, individual/matrix calculation engine, Test tab, dataset tools and visual dataset editor, formula library, manual-calculation mode, code export, performance settings.
  2. **Mobile helpers** (≈ 19762–19839) — mobile navigation and stat-bar sync.
  3. **Extension block** (≈ 21964–27060) — decision boundary, optimiser race, network comparison, mutation, save/load, LR scheduler, gradient/layer/benchmark tools, effects (Matrix rain, sound, confetti, data flow, heatmap), **CNN Lab** and **RNN Lab**.
  4. **Translation layer** (≈ 27062–27490) — automatic FR ↔ EN interface translation.
  5. **RL Lab** (≈ 27493–28282) — bandits, calculator, grid world (Q-learning/SARSA), dynamic programming, policy gradient, DQN on CartPole.
  6. **Tab indicator** (≈ 28289–28414) — the sliding underline of tab bars.
  7. **Micro-interactions** (≈ 28421–28640) — ripples, magnetic spotlight, particle bursts, animated counters, modal entrance animations.
  8. **Slider value bubble** (≈ 28762–28832) — floating value above a dragged slider.
  9. **Transformer Lab** (≈ 28833–29032).
  10. **Section folding & Aurora pack** (≈ 29187–29255) — collapsible left-panel sections and log, animated signal flow on the network canvas.
  11. **Step-by-step explainer** (≈ 29665–30016) — the 🎓 « Pas à pas » mode.

There are no bundled external JavaScript dependencies except the **KaTeX** library (loaded via CDN) for rendering mathematical notation. The automatic translation layer additionally calls a public translation API (Google Translate's unofficial endpoint, with a MyMemory fallback) over the network at runtime — it is not a bundled dependency and only activates when the user enables translation. GPU training uses the browser's built-in **WebGPU** API when available and is entirely optional.

### Key capabilities

| Category | Details |
|---|---|
| Architectures | Fully-connected MLP, arbitrary depth and width, per-layer activations |
| Activations | Sigmoid, ReLU, Tanh, Leaky ReLU, ELU, Swish, GELU, SELU, Softsign, Linear, Softmax |
| Loss functions | MSE, MAE, Log Loss (BCE), Huber, Hinge |
| Optimisers | SGD, Momentum, RMSProp, Adam, AdamW, Nesterov AG |
| Weight init | Xavier/Glorot, He/Kaiming, Uniform, Normal, Small, Zero (debug) |
| LR schedulers | Cosine annealing, Step decay, Exponential decay, Warmup-Cosine |
| Training modes | Single step, single epoch, N epochs, auto-run, train-to-target, early stopping |
| Regularisation & techniques | L2 weight decay, global gradient-norm clipping, dropout, **Batch Normalisation**, **Gaussian-noise data augmentation**, **early stopping**, per-100-epoch LR decay |
| Compute | CPU (default) or opt-in **WebGPU** training, verified against the CPU path |
| Calculation views | One global switch: **individual** (neuron by neuron) or **matrix** (real `W·a + b` products) in Test, Details, Formulas, Manual, logs and code export |
| Interactivity | Ctrl+click a neuron to pin its activation, Shift+click a neuron/weight to freeze it, Alt+click to force a pattern (0 / 1 / sine / random) |
| Inspection | Step-by-step log, 🎓 step-by-step explainer, gradient analysis, layer inspector, weight heatmap, zoomed loss chart, node/connection detail popovers |
| Datasets | 8 logic/circuit presets, row editor, JSON import/export, **visual 2D classification/regression editor** |
| Export | PNG canvas, clipboard copy, LaTeX log, Markdown log, JSON model, save/load slots, **Python / JavaScript code generation** (loops or NumPy), dataset JSON import/export |
| Extras | Decision boundary visualiser, optimiser race, **two-network comparison**, network mutation, confetti, Matrix rain, data flow animation, sound feedback, colour themes |
| Learning tools | Formula library (~110 entries, 12 categories), **manual "do the math" mode**, **140-question quiz** |
| CNN Lab | Convolution playground, kernel gallery, multi-layer pipeline, kernel training, complete trainable image classifier with bitmap editor up to 1536×1536 |
| RNN Lab | Recurrent-cell step-through, sequence unrolling, vanishing/exploding-gradient demo, trainable character-level RNN |
| Transformer Lab | Theory walkthrough, live multi-head attention, positional encodings, decoding strategies, parameter/FLOP/memory estimates |
| RL Lab | Bandits, RL calculator, Q-learning / SARSA / Expected SARSA, Bellman DP, policy gradient, DQN on CartPole |
| Localisation | One-click automatic FR → EN interface translation (external API, DOM-based) |

---

## Interface Layout

The UI is divided into three resizable panels separated by drag handles (`.rsz-col`) and a vertically resizable log panel (`.rsz-row`).

```
┌──────────────┬────────────────────────────┬───────────────────┐
│  LEFT PANEL  │     CENTER — Canvas        │   RIGHT PANEL     │
│  (260 px)    │   Neural network diagram   │   (370 px)        │
│              │   (+ 🎓 step-by-step panel)│                   │
│  Architecture│                            │  Tabs (9):        │
│  Training    │                            │  - Détails        │
│  Mode de     │                            │  - Test           │
│    calcul    │                            │  - Dataset        │
│  Display     │                            │  - Formules       │
│              ├────────────────────────────┤  - Options        │
│              │   LOG PANEL (190 px)       │  - Guide          │
│              │   Structured backprop log  │  - 🧠 Quiz        │
│              │                            │  - ✎ Manuel       │
│              │                            │  - 🛠 Outils       │
└──────────────┴────────────────────────────┴───────────────────┘
```

The left panel is made of four collapsible sections (**Architecture**, **Entraînement**, **Mode de calcul**, **Affichage**); clicking a section title folds it, and the log panel can be collapsed by clicking its title.

Modal overlays sit above this layout: the **Wizard** and **Advanced Builder**, the **decision boundary**, the **visual dataset editor** (opened from the Dataset tab), **Save/Load slots**, the **optimiser race** and the **network comparison** (from the Outils tab), and four self-contained multi-tab labs — **CNN Lab**, **RNN Lab**, **Transformer Lab** and **RL Lab** — each documented in its own section below.

On viewports ≤ 900 px the layout switches to a **mobile mode**: a fixed top bar, a bottom navigation bar, and each panel occupies the full screen as a switchable tab.

---

## Architecture & Global State

### Global variables

| Variable | Type | Description |
|---|---|---|
| `NET` | `{L, W, B}` or `null` | The active network. `L` = layer sizes array, `W` = weight tensors (3-D), `B` = bias vectors (2-D) |
| `LOSS_H` | `number[]` | Loss history per epoch |
| `EPOCH` | `number` | Current epoch counter |
| `autoRunning` | `boolean` | Whether the auto-training loop is active |
| `LAST_ACT` | `number[][]` | Activations from the most recent forward pass (per layer) |
| `LAST_Z` | `number[][]` | Pre-activations (z values) from the most recent forward pass |
| `LAST_DELTAS` | `number[][]` | Delta values from the most recent backward pass |
| `VEL` | `{w, b}` | Velocity accumulators for Momentum / NAG |
| `MA` | `{w, b}` | First-moment estimates for Adam/AdamW |
| `MV` | `{w, b}` | Second-moment estimates for Adam/AdamW/RMSProp |
| `ADAM_T` | `number` | Adam step counter (for bias correction) |
| `DISP` | `{W,B,A,I,G,Grid}` | Boolean display toggles for canvas overlays |
| `DS` | `{inputs, outputs, inN, outN}` | Active dataset |
| `POS` | `{x,y,l,j,av}[]` | Computed neuron positions for canvas hit-testing |
| `LINES` | `object[]` | Computed connection lines for canvas hit-testing |
| `LAYER_ACTS` | `string[]` | Per-layer activation function overrides (used by Advanced Builder) |
| `BN` | `{gamma, beta, runMean, runVar}` | Batch-Normalisation parameters and running statistics, one value per hidden neuron (see [Training Techniques](#training-techniques-batch-norm-augmentation-early-stopping)) |
| `BN_ZHAT` / `BN_GRAD` | `number[][]` / `{gamma, beta}` | Normalised pre-activations of the last forward pass / γ,β gradients of the last backward pass |
| `EARLY` | `{bestLoss, badEpochs}` | Early-stopping tracker |
| `PERF` | `{fastTrainLite, _savedGlow}` | Performance preferences (lighter rendering during Auto training) |
| `GPU` | `{supported, ready, device, pipelines, adapterInfo, verified, busy, main}` | WebGPU training state (see [GPU Training](#gpu-training--performance-webgpu)) |
| `CALC_MODE` | `"indiv"` \| `"matrix"` | Global calculation display mode (see [Calculation Mode](#calculation-mode-individual--matrix)) |
| `VERSION` / `REPO` | `string` | `"v13"` and the GitHub repository URL, injected into every `.version` element |
| `RAF_ID` | `number` | `requestAnimationFrame` handle for the auto-training loop |
| `SELECTED` | `object` | Currently selected neuron (for tooltip) |
| `ε` | `1e-15` | Numerical stability constant |

### Helper constants / micro-functions

| Name | Signature | Description |
|---|---|---|
| `h2r` | `(hex) → {r,g,b}` | Converts a hex colour string to an RGB object |
| `clamp` | `(v, a, b) → number` | Clamps `v` between `a` and `b` |
| `gauss` | `() → number` | Box-Muller Gaussian sample (mean=0, σ=1) |
| `getLR` | `() → number` | Reads the learning rate from the `lrN` input |
| `getAct` | `() → string` | Reads the hidden-layer activation key from `iAct` |
| `getActOut` | `() → string` | Reads the output-layer activation key (`iActOut`), falling back to `getAct()` if set to `"same"` |
| `getLoss` | `() → string` | Reads the loss function key from `iLoss` |
| `getOpt` | `() → string` | Reads the optimiser key from `iOpt` |
| `getV` | `(id) → number` | Reads a numeric value from an input element by ID |
| `bnOn` / `bnMomentum` | `() → boolean / number` | Batch-Norm switch (`cBN`) and running-statistics momentum (`cBNMom`, default 0.9) |
| `augOn` / `augNoise` | `() → boolean / number` | Data-augmentation switch (`cAug`) and noise σ (`cAugNoise`) |
| `earlyOn` / `earlyPatience` / `earlyMinDelta` | `() → boolean / number` | Early-stopping switch (`cES`), patience (`cESPat`, default 30) and minimum improvement (`cESDelta`) |
| `ensureBN` | `() → void` | Re-creates the `BN` arrays so they match the current network shape |
| `fmt` | `(v) → string` | Formats a number to 5 decimal places, or `"NaN"` |
| `syncLR` | `(v) → void` | Synchronises the LR slider, number input, and stats display |

---

## Activation Functions

### `act(z, fn) → number`

Computes the activation output for a single pre-activation value `z` using the function identified by `fn`.

| `fn` | Formula | Notes |
|---|---|---|
| `sigmoid` | `1/(1+e^{-z})` | Clamped to `[-500, 500]` for numerical stability |
| `relu` | `max(0, z)` | Standard rectified linear unit |
| `tanh` | `tanh(z)` | Hyperbolic tangent via `Math.tanh` |
| `leakyrelu` | `z > 0 ? z : 0.01·z` | Fixed leak slope 0.01 |
| `elu` | `z ≥ 0 ? z : e^z − 1` | Exponent clamped to `[-50, 0]` |
| `swish` | `z · sigmoid(z)` | Self-gated activation (Google Brain) |
| `gelu` | `0.5·z·(1 + tanh(√(2/π)·(z + 0.044715·z³)))` | Tanh approximation, used in BERT/GPT |
| `selu` | `scale · (z ≥ 0 ? z : α·(e^z − 1))` | α=1.6732632, scale=1.0507009 |
| `softsign` | `z / (1 + |z|)` | Range (−1, 1), less saturating than tanh |
| `linear` | `z` | Identity, used for regression outputs |
| `softmax` | pass-through (handled externally by `softmax()`) | See below |

### `actD(a, z, fn) → number`

Computes the derivative of the activation for use in backpropagation. Takes both the output activation `a` and the pre-activation `z` because some derivatives are more efficiently expressed in terms of `a`.

| `fn` | Derivative |
|---|---|
| `sigmoid` | `a·(1−a)` |
| `relu` | `z > 0 ? 1 : 0` |
| `tanh` | `1 − a²` |
| `leakyrelu` | `z > 0 ? 1 : 0.01` |
| `elu` | `z ≥ 0 ? 1 : a + 1` |
| `swish` | `s + z·s·(1−s)` where `s = sigmoid(z)` |
| `gelu` | Full analytical derivative via tanh approximation |
| `selu` | `z ≥ 0 ? scale : scale·α·e^z` |
| `softsign` | `1 / (1 + |z|)²` |
| `linear` | `1` |

### `softmax(arr) → number[]`

Applies the softmax function to a vector `arr`. Uses the **max-subtraction trick** (`eᵛ⁻ᵐᵃˣ`) for numerical stability. Returns a normalised probability distribution summing to 1.

### `getActForLayer(l) → string`

Returns the activation function name for layer index `l` (0-indexed). If `LAYER_ACTS` has been populated by the Advanced Builder, it returns `LAYER_ACTS[l]`; otherwise it returns the output activation for the last layer and the global hidden activation for all others.

---

## Loss Functions

### `computeLoss(pred, y) → number`

Computes the scalar loss between prediction array `pred` and target array `y`.

| Key | Formula | Notes |
|---|---|---|
| `mse` | `(1/n)·Σ(p−y)²` | Mean Squared Error |
| `mae` | `(1/n)·Σ|p−y|` | Mean Absolute Error |
| `logloss` | `−(1/n)·Σ[y·log(p̂)+(1−y)·log(1−p̂)]` | Binary Cross-Entropy; prediction clamped to `[ε, 1−ε]` |
| `huber` | `½e² if |e|≤δ, else δ(|e|−½δ)` | Hybrid MSE/MAE; δ read from input `cHD` |
| `hinge` | `(1/n)·Σ max(0, 1−t·p)` | SVM-style; label `y=0` maps to `t=−1` |

### `computeLossGrad(pred, y) → number[]`

Returns the per-output gradient of the loss `∂L/∂p` for each output neuron. Uses the same formula keys as `computeLoss`. This gradient seeds the backpropagation chain rule at the output layer.

---

## Weight Initialisation

### `initW(nIn, nOut, method) → number`

Generates a single initial weight value given the fan-in `nIn` and fan-out `nOut`.

| `method` | Distribution | Best for |
|---|---|---|
| `xavier` | `U(−√(6/(nIn+nOut)), +√(6/(nIn+nOut)))` | Sigmoid, Tanh |
| `he` | `N(0, √(2/nIn))` — Box-Muller | ReLU, Leaky ReLU |
| `uniform` | `U(−1, 1)` | General purpose |
| `normal` | `N(0, 0.1)` | Small random |
| `small` | `U(−0.1, 0.1)` | Avoiding saturation |
| `zero` | `0` | Debugging |
| *(default)* | `U(−0.8, 0.8)` | Fallback |

The `method` defaults to the value of the `cInit` selector in the UI.

---

## Core Neural Network Functions

### `buildNet(customLayers?, customActs?, customInitMethod?) → void`

Constructs a new neural network from scratch, replacing any existing one.

**Parameters:**
- `customLayers` — array of layer sizes (e.g. `[2,4,3,1]`); defaults to the value of `iLayers` in the UI.
- `customActs` — array of per-layer activation names (populates `LAYER_ACTS`); used by the Advanced Builder.
- `customInitMethod` — weight initialisation method string; falls back to the `cInit` selector.

**Behaviour:**
1. Parses and validates the layer size array (minimum two layers).
2. Synchronises `DS.inN`/`DS.outN` with the new architecture if they differ.
3. Initialises `NET.W` (weight matrices) and `NET.B` (bias vectors) using `initW`.
4. Resets all optimiser state (`VEL`, `MA`, `MV`, `ADAM_T`).
5. Resets epoch counter, loss history, and cached pass data.
6. Triggers a full canvas redraw and logs a creation summary.
7. Calls `gpuOnNetChanged()` so that GPU buffers are re-allocated and re-verified when WebGPU training is active (`resetW()` does the same).

### `resetW() → void`

Re-randomises all weights and biases of the existing network using the current initialisation method. Resets epoch, loss history, and optimiser state but preserves the network topology.

### `setBias(l, j, val) → void`

Manually sets the bias of neuron `j` in layer `l` to a parsed float value. Redraws and logs the change. Used from click-to-edit interactions in the detail panel.

### `forward(inputs, isTrain?) → {activations, zs}`

Performs a **forward pass** through the entire network.

**Returns:**
- `activations` — 2-D array of shape `[layers+1][neurons]`; `activations[0]` is the input layer.
- `zs` — 2-D array of shape `[layers][neurons]`; the pre-activation value that actually enters each non-input neuron's activation function (after Batch Norm when it is enabled).

**Algorithm:**
1. Seeds the activation array with the input vector.
2. For each layer `l`, computes the dot product `z[j] = Σ(W[l][j][k] · a[l][k]) + B[l][j]`.
3. If **Batch Normalisation** is on and the layer is hidden: when `isTrain` is true, updates the running mean/variance of `z`; then replaces `z` by `γ·(z − mean)/√(var + 1e-5) + β` (see [Training Techniques](#training-techniques-batch-norm-augmentation-early-stopping)).
4. Applies the layer's activation function (via `getActForLayer`), then any **node pattern** forced by Alt+click (`0`, `1`, `random`, or `sin(EPOCH·0.1 + j)`).
5. For the last layer, applies **softmax** if the output activation is `"softmax"`.
6. For hidden layers, applies **dropout** (each activation zeroed with probability `cDrop`) when the rate is > 0.
7. Stores results in `LAST_ACT` and `LAST_Z` (done by the callers) for inspection and logging; `BN_ZHAT` keeps the normalised values for the backward pass.

### `backward(activations, zs, y) → deltas`

Performs **backpropagation** and returns the delta matrix without updating weights.

**Returns:**
- `deltas` — 2-D array of shape `[layers][neurons]`, one delta per neuron (`∂L/∂z` of the raw weighted sum, which is what the weight gradient `δ·a` needs).

**Algorithm:**
1. Computes the output layer deltas: `δ[j] = (∂L/∂a) · f'(z)` using `computeLossGrad` and `actD`.
2. For softmax outputs, approximates `f'(z) = a·(1−a)` (diagonal of the Jacobian).
3. Propagates error backwards through hidden layers: `err = Σⱼ δ[l+1][j] · W[l+1][j][k]`, then `dz_eff = err · f'(z[l][k])`.
4. With Batch Norm on, the gradient is chained through the affine step: `δ[l][k] = dz_eff · γ/√(var + 1e-5)`, and `BN_GRAD.gamma[l][k] = dz_eff · ẑ`, `BN_GRAD.beta[l][k] = dz_eff`. Without it, `δ[l][k] = dz_eff`.
5. Deltas are stored in `LAST_DELTAS` by the caller for logging.

### `trainSample(inputs, y, verbose?) → number`

Runs a complete **forward + backward + weight update** cycle for a single training example. If data augmentation is on, Gaussian noise is added to the input used for this step (the stored dataset is untouched).

**Returns:** the loss for this sample.

**Optimiser update rules (implemented in `updateParam`, applied per weight `w`):**

| Optimiser | Update rule |
|---|---|
| `sgd` | `w ← w − lr · g` |
| `momentum` | `w ← w − (mom·v + lr·g)` |
| `nag` | same rule as momentum (look-ahead handled via the stored velocity) |
| `rmsprop` | `vₙ ← β₂·v + (1−β₂)·g²`, then `w ← w − lr·g / (√vₙ+ε)` |
| `adam` | bias-corrected `m`/`v` (`β₁`, `β₂`), `w ← w − lr·m̂/(√v̂+ε)` |
| `adamw` | Adam update **plus** a decoupled weight-decay term `w ← w − lr·0.01·w` |

**Regularisation applied inside `applyGrad()` before the update loop:**

- **L2 weight decay** — if the "L2" hyper-parameter (`cL2`) is > 0, it is added directly to each weight's gradient: `g = dj·activation + l2·w` (weights only, not biases).
- **Global gradient-norm clipping** — if the "Grad Clip" hyper-parameter (`cGC`) is > 0, the L2 norm of all delta values is computed and, if it exceeds the threshold, every delta is rescaled by `threshold / norm` before weights are updated.
- **Batch-Norm parameters** — when Batch Norm is on, γ and β are updated by plain SGD with the effective learning rate after the weights.
- **Learning-rate decay** — the effective learning rate is `lr · lrDecay^⌊epoch/100⌋`, i.e. the configured "LR Decay" factor (`cLRD`) is applied automatically every 100 epochs.
- **Weight / node freezing** — before updating a weight `NET.W[l][j][k]`, `applyGrad` checks `NET.frozenWeights["l-j-k"]`; before updating a bias, it checks `NET.frozenNodes["(l+1)-j"]`. Frozen parameters are skipped entirely (no gradient applied). Weights and neurons are frozen/unfrozen interactively by **Shift+clicking** them on the canvas (see [UI Utility Functions](#ui-utility-functions)).

If `verbose` mode is active (the "Complet" log tab is shown), `trainSample` also generates the detailed step-by-step HTML log covering all four phases: Forward Pass, Loss, Backpropagation, and Weight Update (`logFull`).

---

## Optimiser Engine

The optimiser state is stored globally:

- `VEL.w[l][j][k]` / `VEL.b[l][j]` — velocity (Momentum/NAG)
- `MA.w[l][j][k]` / `MA.b[l][j]` — first moment (Adam/AdamW)
- `MV.w[l][j][k]` / `MV.b[l][j]` — second moment (Adam/RMSProp/AdamW)
- `ADAM_T` — global step counter for Adam bias correction

Hyperparameters default to conventional values but are all user-editable in the **Options** tab's "Algorithme avancé" section:
- Momentum β = 0.9 (`cMom`)
- RMSProp β = 0.9 (shares `cMom`)
- Adam β₁ = 0.9 (`cB1`), β₂ = 0.999 (`cB2`), ε = 10^`cEps` (default exponent −8)
- AdamW λ (weight decay) = 0.01 (fixed)
- L2 weight decay coefficient (`cL2`, default 0)
- Global gradient-clip threshold (`cGC`, default 0 = disabled)
- Learning-rate decay factor applied every 100 epochs (`cLRD`, default 1 = disabled)
- Dropout rate and Huber-loss delta are also configurable here (`cDrop`, `cHD`)

Batch Norm, data augmentation and early stopping live in their own **Techniques d'entraînement** section — see [Training Techniques](#training-techniques-batch-norm-augmentation-early-stopping).

---

## Training Techniques (Batch Norm, Augmentation, Early Stopping)

Configured in **Options → Techniques d'entraînement**. All three are optional and off by default. Dropout (configured in *Algorithme avancé*) is documented here too because it is applied in the same place.

### Batch Normalisation (`cBN`, `cBNMom`)

State: `BN = {gamma, beta, runMean, runVar}` (one value per hidden neuron), `BN_ZHAT` (normalised pre-activations of the last forward pass) and `BN_GRAD` (γ/β gradients of the last backward pass). `ensureBN()` re-creates the arrays (γ = 1, β = 0, mean = 0, var = 1) whenever the network shape changes, so BN survives `buildNet`, the wizard, imports and loads.

- **Scope** — hidden layers only, never the output layer.
- **Forward** — the layer's pre-activation is normalised with *running* statistics: when `isTrain` is true, `mean ← m·mean + (1−m)·z` and `var ← m·var + (1−m)·(z−mean)²` (momentum `m` from `cBNMom`, default 0.9), then `ẑ = (z − mean)/√(var + 1e-5)` and the activation receives `γ·ẑ + β`. These are per-neuron running statistics over the samples seen, not statistics over a mini-batch.
- **Backward** — the gradient is chained through the affine `γ/β` step; mean and variance are treated as constants (detached running statistics). The delta stored for the layer is `dL/dz_eff · γ/√(var + 1e-5)`; `BN_GRAD.gamma = dL/dz_eff · ẑ`, `BN_GRAD.beta = dL/dz_eff`.
- **Update** — γ and β are updated by plain SGD with the effective learning rate, independently of the chosen optimiser.

### Data augmentation (`cAug`, `cAugNoise`)

When enabled, `trainSample` adds Gaussian noise `N(0, σ)` (σ = `cAugNoise`, default 0.05) to the input of each **training** sample. The stored dataset `DS.inputs` is never modified. Test, accuracy and manual mode use the clean inputs.

### Early stopping (`cES`, `cESPat`, `cESDelta`)

`EARLY = {bestLoss, badEpochs}`. `earlyStopUpdate(avgLoss)` is called after each epoch: if `bestLoss − avgLoss > minDelta` (default 0.0005) the best loss is updated and the counter reset, otherwise the counter increases. When it reaches the patience (default 30 epochs) the function returns `true`, and `doEpoch`/`doN` stop training (`stopAutoWithMessage` also stops Auto mode and prints an « ⏸ Early stopping » message in the log). Single steps (`doStep`) do not feed the tracker.

### Dropout (`cDrop`)

In `forward()`, when the dropout rate is > 0, every hidden activation is set to 0 with that probability. No `1/(1−p)` rescaling is applied, and the mask is drawn on every forward call (including inference-time forward passes), so keep the rate at 0 for deterministic tests.

---

## Training Control Functions

### `doStep() → Promise<void>`

Runs one training sample chosen **at random** from the dataset, updates the weights, increments `EPOCH` by one, pushes the loss to `LOSS_H`, updates stats, redraws the canvas and plays a flash animation. Uses the GPU path when `gpuEligible()` is true, otherwise the CPU `trainSample(…, verbose = true)` (which also writes the detailed log).

### `doEpoch() → Promise<void>`

Iterates over **every sample in the dataset once** (in dataset order), then increments `EPOCH`, pushes the average loss to `LOSS_H`, feeds `earlyStopUpdate`, updates stats, redraws, and logs the epoch summary with trend arrows (`logEpoch`, or a « ⚡ GPU » line on the GPU path). If early stopping triggers it stops Auto mode.

### `doN(n) → Promise<void>`

Runs **`n` full epochs** in a loop (on the GPU path, `gpuTrainEpochs(n, earlyOn())` queues all of them and reads the losses back once, or after each epoch when early stopping is on). Logs a « ×n » summary and an early-stopping message if applicable.

### `toggleAuto() / runAuto() → void`

Toggles the auto-training loop. `runAuto` awaits `doEpoch()` repeatedly, scheduling itself with `setTimeout` when the `autoSpeed` slider is > 0 ms (up to 500 ms) and `requestAnimationFrame` otherwise. While Auto is running, `body.auto-mode` is set and — if the « Effets réduits pendant l'entraînement rapide » option (`PERF.fastTrainLite`) is on — the glow overlay is switched off, then restored on stop.

### `trainToTarget() → void`

Trains the network epoch by epoch until either the average loss drops below the target value from `lossTarget` or a maximum of 10 000 epochs is reached. Updates the `target-status` element with live progress. Plays a success tone on convergence if sound is enabled.

### `stopTargetTrain() → void`

Clears the `targetTrainTimer` timeout used by `trainToTarget()`.

### `acc() → number`

Computes the current classification accuracy (in %) over the full dataset. A prediction is considered correct if `Math.round(p) === y` for all outputs. Returns 0 if the dataset is empty.

### `updateStats(loss) → void`

Updates the four stat cards in the left panel (`sEpoch`, `sLoss`, `sAcc`, `sLR`), appends a ↘/↗ trend arrow (loss compared with the mean of the last 10 values), redraws the mini loss chart and flashes the cards.

### `drawLossChart() → void`

Renders the small loss curve in the left panel (`#lossChart`) using an HTML5 Canvas. Draws the raw loss curve in `--red` and, when more than 10 data points are available, a 10-point moving average in `--ylw` (dashed).

---

## GPU Training & Performance (WebGPU)

A browser cannot call CUDA/cuDNN/TensorRT; **WebGPU** is the only GPU compute API available to a web page. It is vendor-neutral and runs on the machine's GPU (NVIDIA, AMD, Intel, Apple) through the browser driver. Neural Lab uses it as an **opt-in training backend** (Options → *Performance & Accélération matérielle* → *Calcul d'entraînement sur GPU (WebGPU)*, `cGPUTrain`).

### Engine

- `GPU_WGSL` — a WGSL shader module implementing the forward pass, the loss gradient, backpropagation and the optimiser update. Numeric codes map the UI choices: `ACT_CODE` (11 activations incl. softmax), `LOSS_CODE` (5 losses), `OPT_CODE` (6 optimisers). A uniform `StepParams` block carries lr, L2, gradient clip, momentum, β₁/β₂, ε, Adam step counter, Huber δ, and codes.
- `GPU = {supported, ready, device, pipelines, adapterInfo, verified, busy, main}` — runtime state; `initWebGPU()` requests a high-performance adapter and device (and logs a message if the device is lost); `buildGPUPipelines()` compiles the pipelines.
- Buffers: `gpuBuildLayerSet()` / `gpuAllocateBuffers()` create per-layer weight, bias, activation, delta and optimiser-state buffers for the current topology (`netSig` = layer sizes); `gpuSyncWeightsToDevice()` uploads the CPU weights, `gpuEncodeSampleOn()` records one sample's forward/backward/update dispatches, `gpuCommitResults()` / `gpuReadWeightsFromDevice()` read weights back into `NET`.
- `gpuTrainOneSample(x, y)` (used by `doStep`) and `gpuTrainEpochs(n, checkEarlyStop)` (used by `doEpoch`, `doN`, Auto) run the training, accumulate per-epoch losses on the GPU (`lossAcc`) and honour augmentation and early stopping.

### When the GPU is used — `gpuEligible()`

All must hold: WebGPU ready and verified, no GPU call in flight, buffers match the current architecture, a non-empty dataset, the *GPU training* option enabled (persisted in `localStorage["nl_gpu_train"]`), **Batch Norm off**, **dropout = 0**, **no forced neuron patterns**, **no frozen weights/neurons**. Otherwise the CPU path is used transparently.

### Verification

`verifyGPUvsCPU()` runs one training step on the first sample with fully isolated GPU buffers and compares the resulting weights with `cpuReferenceStep()`; it succeeds when the largest deviation is < 1e-3. `setGPUTrain(true)` runs it on activation, `manualVerifyGPU()` re-runs it on demand (« 🔍 Vérifier GPU vs CPU »), and `gpuOnNetChanged()` (called from `buildNet`/`resetW`) re-allocates buffers and re-verifies silently. A deviation does not disable the GPU — the choice stays with the user — but a warning is shown in the status line (`gpuTrainStatus`) and the log.

### Trade-offs

- On small networks the CPU can be faster; the option is meant for experimentation.
- In GPU mode the detailed step-by-step formula logs are **not** produced during Step / Epoch / ×N / Auto, and the **Manuel** tab always stays on the CPU.

### Rendering performance

`detectGPU()` reports WebGL/GPU availability in `#gpuInfoBox`; `setGPUAccel`, `setCanvasRenderMode` (Auto / Force GPU / Force CPU for the 2D canvas — a page reload applies it fully) and `canvasCtxAttrs()` / `applyGPUCompositingHints()` tune the canvas contexts; `setFastTrainLite` toggles the reduced-effects mode used by Auto training; `initPerfSettings()` restores the saved choices.

---

## Visualisation & Canvas Rendering

### `redraw() → void`

The main canvas rendering function. Clears and repaints the entire neural network diagram. Called after every training step, weight change, or resize event.

**Steps:**
1. Clears the canvas and fills the background (from `cBG` colour picker).
2. Optionally draws a **grid** (cross or dot pattern) from `cGridMode`.
3. Computes neuron positions (`POS`) based on layer widths and spacing factors `cSV`/`cSH`.
4. Draws **connections** (weight lines) with:
   - Colour from `cWP`/`cWN` pickers for positive/negative weights.
   - Opacity and thickness proportional to `|w|`.
   - Optional Bézier curves (`cCurve`) or straight lines.
   - Optional arrowheads (`cArrow`) at the midpoints.
   - Weight value labels (if `DISP.W` and network is narrow enough).
   - **Frozen weight indicators** (❄ icon) if `NET.frozenWeights` is set.
5. Draws **neurons** with:
   - Radial glow halos proportional to activation (if `DISP.G`).
   - Fill colour based on activation magnitude (if `cColorNode` is checked).
   - Input / hidden / output layer colour scheme from `cNI`/`cNH`/`cNO` pickers.
   - Neuron index labels (if `DISP.I`).
   - Activation values (if `DISP.A`).
   - Bias values (if `DISP.B`).

### `resizeCvs() → void`

Sets the canvas `width` and `height` to the pixel dimensions of its wrapper div `canvasWrap`. Called on window resize and panel resize.

### `getR() → number`

Returns the neuron radius in pixels from the `cR` input, defaulting to 18.

### `tog(key) → void`

Toggles a display flag in the `DISP` object and updates the corresponding button's `on` class. Keys: `'W'`, `'B'`, `'A'`, `'I'`, `'G'`, `'Grid'`.

---

## Dataset Management

The dataset object `DS` holds `{inputs, outputs, inN, outN}`.

### Presets and row editor

#### `loadPreset() → void`
Loads the built-in preset selected in `dsPreset` into `DS` and rebuilds the editor and preview. Available presets: **XOR**, **AND**, **OR**, **NAND**, **XNOR** (2→1), **Half Adder** (2→2: sum and carry), **4bit Identity** (4→4), and **Personnalisé** (leaves the dataset untouched for manual entry). Circle, spiral and sine datasets are no longer presets: they come from the [visual dataset editor](#visual-dataset-editor).

#### `rebuildDSEditor() / buildDSEditorFromCurrent() → void`
`rebuildDSEditor()` reads the input/output counts (`dsIn`, `dsOut`) and regenerates the editable rows; `buildDSEditorFromCurrent()` regenerates the rows of `#dsEditor` from the current `DS.inputs` and `DS.outputs`.

#### `addDSRow() / delDSRow(i) → void`
Add a blank sample row, or remove the row at index `i`.

#### `applyDS() → void`
Validates the editor's rows and commits them into `DS`; the network's input/output layer sizes are adapted if needed.

#### `renderDSPreview() → void`
Renders a compact read-only preview table of the current dataset.

#### `exportDSJSON() / importDSJSON(input) → void`
Download / load the dataset as JSON (a quick-import box also accepts an array of `{i: [], o: []}`).

### Visual dataset editor

Opened from the Dataset tab with `openVDS()` (modal `#vds-modal-bg`, state `VDS`, closed with `closeVDS()`, Escape or a click on the backdrop). It writes straight into `DS` — and therefore into the network of the central canvas — from points placed with the mouse, and refreshes a live preview every 400 ms (`vdsStartLive`/`vdsStopLive`). Two tabs (`vdsShowTab`), whose datasets cannot coexist: showing a tab reloads its points into `DS`.

- **Classification** (`vdsRenderClass`, `vdsClassSync`) — 2 or more named classes (colour chips; `vdsClassAddClass`, `vdsClassRemoveClass`, `vdsClassSelect`); left-click adds a point of the active class in `[-1, 1]²`, an *erase* mode (or right-click, `vdsClassCanvasContext`) removes the nearest point (`vdsClassRemoveNearest`); `vdsClassGenerate(kind)` creates demos *circles*, *spiral*, *checker* or *blobs*. With ≥ 2 classes the dataset becomes `inN = 2`, `outN = nClasses`, outputs one-hot. When the network architecture is compatible, the background shows the decision regions it currently predicts (`vdsClassRenderBg`).
- **Régression** (`vdsRenderReg`, `vdsRegSync`) — points `(x, y)` with the same add/erase tools; `vdsRegGenerate(kind)` creates *linear*, *quad* or *sine* clouds (24 noisy points). The dataset becomes `1 → 1` and the learned curve is drawn over the points.
- **Architecture helper** — `vdsArchMismatchMsg(nIn, nOut)` reports whether the current network fits the dataset and offers one-click `vdsApplyArch(nIn, nOut)`, which rewrites the first/last layer sizes, selects *softmax + log loss* for several outputs or *linear + MSE* for one output, rebuilds the network and redraws.

---

## Right-Panel Tabs & Inspection Tools

The right panel uses a tab system with **nine tabs**: **Détails**, **Test**, **Dataset**, **Formules**, **Options**, **Guide**, **🧠 Quiz**, **✎ Manuel**, **🛠 Outils**. The **Détails** tab is the default view and hosts the node/connection detail popovers described under [Canvas interaction](#ui-utility-functions); the **Quiz** tab and the **Manuel** tab are covered in their own sections below. The Détails, Test, Formules and Manuel tabs each contain a calculation-mode switch (see [Calculation Mode](#calculation-mode-individual--matrix)).

### Test tab

#### `runTest() → void`
Runs a single inference on the values entered in the test input fields (`tIn0`, `tIn1`, …) and builds the full pedagogical breakdown for it (`showTestInputs()` pre-fills the fields from the selected dataset sample). The explanation is made of four collapsible phases (`togglePhase(id)`), rendered in the current calculation mode:
- **Phase 1 – Forward Pass**: for each neuron (individual mode) or each layer (matrix mode: `Z = W·A + b`), the weighted sum, bias addition, activation formula, and f'(z) for backprop.
- **Phase 2 – Loss**: the loss term per output, with the formula and numerical substitution.
- **Phase 3 – Backpropagation**: the chain rule, output-layer deltas, and hidden-layer delta propagation.
- **Phase 4 – Weight Update**: the gradient `∂L/∂w = δ·a` and new weight values under the selected optimiser.

#### `runAllTest() → void`
Tests every sample in the dataset, displays predictions vs. targets, and reports overall accuracy.

#### `togglePhase(id) → void`
Shows or hides a collapsible phase section in the test steps view.

### Formules tab (Formula Library)

#### `buildFLib() → void`
Initialises the formula library UI: extracts unique tags from `FLIB` (the formula data array), renders tag filter pills, and calls `renderFTree`.

The library holds about **110 entries in 12 categories**: **Bases** (14 formulas in five groups — forward propagation, loss & metrics, backpropagation, learning, architecture: `z`, `a`, the layer-wise forward pass, batch propagation, error, cost, accuracy, `δ` at the output and hidden layers, batch gradient, update rule, LR decay, gradient clipping, parameter count), **Activations**, **Fonctions de perte**, **Optimiseurs**, **Rétropropagation**, **Régularisation**, **Initialisation**, **Architectures**, **CNN**, **RNN**, **Transformer** and **RL**. Each entry has an equation, a KaTeX source (`FLIB_TEX`), and — depending on the calculation mode — an indexed form (`FLIB_IND`) or a matrix/vector form (`FLIB_MAT`) shown under the equation. A collapsible **📖 Légende des symboles** (`renderGlobalLegend`) lists the symbols of the library, and each formula card has its own legend (`renderFormulaLegend`).

#### `setFTag(tag, el) → void`
Filters the library to show only entries with the selected tag. Updates active state on the tag pills.

#### `filterF(query) → void`
Text-search filter applied to formula names, equations, descriptions, and tags. Triggers a re-render.

#### `getFilt() → object[]`
Returns the filtered subset of `FLIB` matching the current tag and text search.

#### `renderFTree(list) → void`
Renders the formula tree as nested collapsible categories and subcategories. Each entry is rendered as a card with its equation, description, pros/cons, and a small activation curve graph if applicable.

#### `showFDetail(id) → void`
Expands the detail view for a specific formula from `FLIB` (function name is `showFDetail`, not `showFormula`). Renders mathematical notation via `renderEqAuto`, shows the individual/matrix form (`cmFormulaExtraHTML`) and a symbol legend (`getFormulaSymbols`, resolved through the v2 glossary `FSYM`: each symbol has a notation, name, explanation, kind — variable, learned parameter, hyperparameter, function/operator, dimension/index or constant — range and a matrix variant; `resolveFormulaSymbols`, `renderFormulaLegend`), and draws an activation curve in a mini canvas via `drawFGraph`.

#### `renderEqAuto(eq) → string`
The primary equation renderer. If a KaTeX-compatible LaTeX source is available for the formula (looked up in the `FLIB_TEX` dictionary) **and** the KaTeX library has loaded, it renders true typeset math via `katex.renderToString`. Otherwise it falls back to `renderMathEq(eq)`.

#### `renderMathEq(eq) → string`
Fallback converter from a plain-text mathematical expression into styled HTML using `<span>` elements (used when KaTeX/`FLIB_TEX` isn't available for a given formula). Handles:
- Fractions `A/B` using `.math-frac` layout.
- Large sigma `Σ` with `.math-sigma` styling.
- Colour coding for `lr`, Greek letters, `ŷ`, `←`/`→` symbols.

#### `togFCat(id) / togFSub(id) → void`
Expand/collapse a top-level category or a subcategory in the formula tree rendered by `renderFTree`.

#### `drawFGraph(domId, fnId) → void`
Draws the activation curve for function `fnId` into a canvas element at `fgcvs_<domId>`. Plots values over z ∈ [−4, +4] using the `act()` function.

#### `getFormulaSymbols(f) → object[]`
Returns the relevant symbol definitions for a formula entry based on its tags and equation. Used to populate the legend block under each formula's detail view.

### Outils (Tools) tab

#### `analyzeGradients() → void`
Runs a full forward + backward pass on every dataset sample, averages the absolute delta and weight-gradient values per layer, and renders a visual report with:
- Min/avg/max of `|δ|` (error signal) per layer.
- Min/avg/max of `|∂L/∂w|` (actual weight gradient) per layer.
- Log-scale bar charts for each metric.
- Automatic detection and labelling of **vanishing gradients** (max < 1e-4) or **exploding gradients** (max > 50).
- A global summary ratio `max_δ / min_δ` to detect inter-layer imbalance.

#### `inspectLayerStats() → void`
Reads the selected layer index from `inspectLayer` and displays a statistics card with:
- Number of neurons and total weights.
- Min/max/mean/std of all weights.
- Min/max of biases.
- Count of dead neurons (activation < 1e-6 on last pass).

#### `populateLayerInspect() → void`
Repopulates the `<select id="inspectLayer">` dropdown with one option per weight layer, after building a new network.

#### `drawZoomedLoss() → void`
Renders a zoomed loss chart in `#zoomedLossChart` showing the last `n` epochs (controlled by `lossZoom` slider). Draws a filled gradient area, the raw loss curve in red, and a 10-point moving average in yellow dashed.

#### `runBenchmark() → void`
Measures training throughput by running 500 consecutive forward + backward passes and reporting samples/second, total time, and milliseconds per sample.

The Outils tab also hosts the UI entry points for several other features documented in their own sections below: the **decision boundary** (`showBoundary()`, `showBoundaryAnim()` for the live view), the **📈 LR Scheduler** controls, the **🏁 Competition Mode** (optimiser race) and **⚔ Comparer 2 réseaux** launchers, **🧬 mutation**, **💾 save slots** and JSON export, the **✨ Effects & Sound** toggles (Matrix rain, sound, data flow, confetti), **🧪 Data Perturbation**, **📷 Export image** (PNG / clipboard), **🗓 Weight Heatmap**, and — as full standalone modal workspaces — the **CNN, RNN, Transformer and RL Lab** launch buttons (`openCNNLab()`, `openRNNLab()`, `openTFLab()`, `openRLLab()`; see their dedicated sections below).

### Dataset tab

A dedicated tab for full dataset editing and I/O: the preset selector, the row editor, `applyDS()`, `renderDSPreview()`, JSON import/export, and the button that opens the 🎯 **visual regression / classification editor**. All functions are described in [Dataset Management](#dataset-management).

### Guide tab

A static, in-app reference tab (no computation) with written explanations of the interface, workflow tips, and a summary of keyboard shortcuts — a quick-start help page for new users.

### Options tab (formerly "Settings")

Beyond the visual customisation controls (neuron radius, font size, line width, colours, grid mode, curve/arrow toggles — all applied immediately via `redraw()`), the Options tab is organised into collapsible sections:

- **Langue / Language** — the FR → EN auto-translation toggle (see [Automatic FR → EN Translation Layer](#automatic-fr--en-translation-layer)).
- **Thème & Couleurs** — colour theme presets, applied via `applyTheme(name)`.
- **Géométrie** — canvas geometry controls (neuron radius, spacing, line width, curve style).
- **Effets visuels** — toggles for animation/visual effects on the canvas.
- **Performance & Accélération matérielle** — GPU detection, hardware-accelerated canvas, canvas render mode, reduced effects during fast training, and the **WebGPU training** switch and verification button (see [GPU Training](#gpu-training--performance-webgpu)).
- **Algorithme avancé** — the optimiser/regularisation hyper-parameters described in [Optimiser Engine](#optimiser-engine): weight-init method, momentum, Adam β₁/β₂/ε, L2 coefficient, dropout rate, Huber delta, gradient-clip threshold, and per-100-epoch LR decay factor.
- **Techniques d'entraînement** — Batch Normalisation, data augmentation and early stopping (see [Training Techniques](#training-techniques-batch-norm-augmentation-early-stopping)).
- **Sauvegarde / Export** — JSON import/export, Python / JavaScript code export, and a shortcut to the Manuel tab.

---

## Calculation Mode (Individual ⇄ Matrix)

A single global setting, stored in `CALC_MODE` (`"indiv"` or `"matrix"`, persisted in `localStorage["nl_calc_mode"]`), controls how every calculation is displayed. The switch (« ● Individuels » / « ▦ Matriciels ») is rendered by `calcSwitchHTML()` in every `.calc-sw-slot` (left panel, Details, Test, Manuel…); `setCalcMode(m)` changes the mode, toggles `body.matrix-mode`, refreshes the switches and re-renders what is already on screen via `calcRerenderAll()` — which **never changes the active tab** (it neutralises `showTab()` through `window.__calcNoTab` while views such as `showNodeDetail` are re-rendered).

Matrix mode is not cosmetic: the products `W·a`, `Wᵀ·δ`, `δ·aᵀ`… are really computed with a small matrix algebra (`cmCol`, `cmFlat`, `cmT`, `cmMul`, `cmAdd`, `cmSub`, `cmScale`, `cmHad`, `cmMap`, `cmOuter`) and then compared with the neuron-by-neuron engine (`cmMaxDiff`/`cmCheck`).

| Area | Functions |
|---|---|
| Test tab — one sample | `cmFwdLayer`, `cmFwdAll` (`Z = W·A + b`, `A = f(Z)`), `cmLossHTML`, `cmBackAll` (`δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ f′(Z⁽ˡ⁾)`, with `cmDeltaWarn`), `cmUpdAll` (`∇W = δ·aᵀ`) |
| Test tab — « Tester tout » | `cmBatchHTML()` stacks the whole dataset as the columns of `X` and propagates it in one pass: `Z = W·A + b·1ᵀ` |
| Details tab | `cmNeuronDetailHTML`, `cmLineDetailHTML`, `cmNeuronGradHTML`, `cmSwapCard`/`cmPostNode`/`cmPostLine` (a neuron shows its row of `W`) |
| Manuel tab | `cmInputGrid`/`cmInputCell`, `cmManFwdFields`, `cmManDeltaFields`, `cmManUpdFields` (matrix input grids that reuse the individual-mode element ids, so the checks are unchanged), `cmManResultGrid`, `cmManFwdResult`, `cmManDeltaResult`, `cmManUpdResult` |
| Journal | `cmLogForward` and friends; each log block keeps both renderings in `blk._bodies = {ind, mat}` and `calcRefreshLogBlocks()` swaps them without recomputing |
| Formules tab | `FLIB_MAT` (matrix/vector form, e.g. softmax Jacobian, `∇W = δaᵀ`) and `FLIB_IND` (indexed form) with `cmFormulaMatrixHTML`/`cmFormulaIndivHTML`; activations share a generic `_act` entry |
| Code export | `cmExportBody(lang)` — explicit loops (individual) or NumPy / array code (matrix) |
| Labs | `cmCnnCalcHTML` (window-by-window convolution vs `im2col` product), `cmRnnCellHTML`, `cmTfDetHTML` (attention), refreshed by `calcRerenderLabs()` |

### Matrix-notation box — `matrixNotesHTML()`

A collapsible `<details class="matrix-notes">` legend prepended to each phase of the detailed logs (`logFull`, `logManualFull`) in matrix mode — while the Formules tab shows its global « 📖 Légende des symboles », whose entries switch to their matrix variants (with dimensions) in matrix mode: `X = A⁽⁰⁾ ∈ ℝ^{n₀×m}`, `W⁽ˡ⁾ ∈ ℝ^{n_l×n_{l−1}}`, `b⁽ˡ⁾ ∈ ℝ^{n_l×1}` (broadcast over columns), `Z⁽ˡ⁾, A⁽ˡ⁾, δ⁽ˡ⁾ ∈ ℝ^{n_l×m}`, `Y, Ŷ ∈ ℝ^{n_L×m}`, `∇W, ∇b` (same shapes as `W`, `b`), plus the core equation `Z⁽ˡ⁾ = W⁽ˡ⁾A⁽ˡ⁻¹⁾ + b⁽ˡ⁾ → A⁽ˡ⁾ = f(Z⁽ˡ⁾)`. Convention: rows = neurons/features, columns = examples, `m` = batch size, `L` = number of layers excluding the input, while the script-L `𝓛` denotes the loss.

---

## Manual Calculation Mode (« Manuel »)

The **Manuel** tab turns the tool into a self-quiz: instead of watching the engine compute a forward/backward pass, the user does the arithmetic by hand and the app checks the answer against the real engine's numbers (tolerance ±0.001).

- `manPopulateSamples() → void` — fills the sample selector with every dataset row (`Exemple i: X=[…] → Y=[…]`), called whenever the Manuel tab is opened or the dataset changes.
- `manLoadSample() → void` — runs a real `forward()` + `backward()` pass on the selected sample (stored in `MAN_SAMPLE`), displays `X`/`Y`, shows the current loss function's formula, and builds three sets of blank input fields via `manBuildFwdFields()`, `manBuildDeltaFields()`, and `manBuildUpdFields()` — one numeric input per neuron/weight the user must fill in (weighted sum `z`, activation `a`, loss, error term `δ`, and updated weight value). It also writes the fully-worked solution to the main log via `logManualFull()` for later comparison.
- `manCheckFwd() → void` — compares the user's entered `z`/`a` values for every neuron against the real `forward()` output, marking each with ✓/✗ and a green "Parfait !" banner if everything matches.
- `manCheckLoss() → void` — compares the user's entered loss value against `computeLoss()`.
- `manCheckDeltas() → void` — compares the user's entered error terms (`δ`) against the real `backward()` output, layer by layer.
- `manCheckUpdates() → void` — compares the user's entered post-update weight values against what `applyGrad()`/`updateParam()` would actually produce for the current optimiser.
- `manRevealAll() → void` — fills every field with the correct reference value, for users who want to see the worked solution instead of self-checking.

**v13 additions:**
- The tab contains the calculation-mode switch: in **matrix mode** the answer fields become matrix input grids (`cmManFwdFields`, `cmManDeltaFields`, `cmManUpdFields`) that reuse the same element ids, so every `manCheck*` function works unchanged, and the verification results are shown as matrices (`cmManFwdResult`, `cmManDeltaResult`, `cmManUpdResult`). Typed values are preserved when the mode is switched.
- The manual mode always runs on the **CPU**, even when WebGPU training is enabled.
- With Batch Norm on, the reference values include the normalisation step, since they come from the real `forward()`/`backward()`.

---

## Quiz Tab

A self-assessment tab with a bank of **140 questions** (`QUIZ_BANK`, ids 1–140), each with an explanation shown after answering.

- **Chapters** (`QUIZ_CATS`): Bases (18), Forward (18), Loss & Backprop (18), Optimisation (24), CNN (24), RNN (24), RL (10), Transformer (4).
- **Types** (`QUIZ_TYPES`): QCM — multiple choice (52), Vrai / Faux — true/false (45), Ouvertes — open answer (43). Open answers are checked by `quizAnswerOpen()` against a list of accepted strings (`q.accept`) after `quizNorm()` normalisation (lower-case, accents and punctuation removed, `includes` match).
- **Modes**: `quizStart(cat, type)` plays a chapter/type filter in shuffled order; `quizStartRandom(n, cat, type)` draws `n` random questions from the filtered pool (`quizStartRandomFromInput` reads `#quizRandomN`). Filter chips (`quizSetCat`, `quizSetType`, `quizRenderFilters`) show live pool sizes.
- **Flow**: `quizRenderCard` → `quizAnswerChoice(val)` / `quizAnswerOpen()` → `quizResolve` (feedback + explanation, running score) → `quizNext` → `quizFinish` / `quizRenderEnd`.
- **End screen**: percentage score, **personal record** per chapter or random-set size in `localStorage` (`nlquiz_best_cat_<cat>`, `nlquiz_best_random_<n>`), the list of missed questions, a replay button and **« Revoir mes erreurs »** (`quizRestartWrong`).

---

## Step-by-Step Explainer

An overlay (the last inline script, panel `#exPanel` inserted under the canvas) opened by the **🎓 Pas à pas** button next to **▶ Step**. It replays one dataset sample through the current network as a guided, animated lesson — **without touching the real network until the final step**.

- **Computation** — `build()` computes the full forward pass, loss, deltas, gradients and the updated weights for the chosen sample (index in `#exSample`) without modifying `NET`; `makeSteps()` turns them into an ordered list of steps with numeric explanations and KaTeX formulas: ① input → each neuron's `z` and `a` → loss/error → output and hidden deltas → weight updates. Each step carries the neurons, connections and tags to highlight.
- **Canvas highlighting** — the current element is drawn in yellow (`#ffd600`), its sources in cyan, the next targets in purple, with value tags; elements are revealed progressively (`applyState()` shows the network as it stands at the current step).
- **Controls** — ⏮ first, ◀ previous, ▶ play/pause, next ▶, a slider and a counter; play speed *Lent / Normal / Rapide* (2800 / 1800 / 900 ms); ⚙ sample and speed options; ▾ to fold the detail text and see more of the network; ← / → / Space shortcuts while the mode is open.
- **Safety** — the mode warns that **Batch Norm and Dropout are ignored**; closing it before the last step discards the preview (updates are kept only when you are on the final step); calling `doStep`, `doEpoch`, `doN` or `toggleAuto` closes the mode first.

---

## Extension Features (v12 block)

All features below are defined in the extension `<script>` block (≈ lines 21964–27060), which follows the modal markup.

### 1. Net Stats Bar — `updateNetStatsBar() → void`
Refreshes the `#net-stats-bar` strip at the top of the canvas area, displaying architecture, total parameter count, epoch, loss, accuracy, LR, and a keyboard shortcut hint.

### 2. Decision Boundary — `renderBoundary() → void`
Renders a 2-D classification boundary in a modal canvas (`#boundary-cvs`). Sweeps a grid of `res×res` pixels over the input space `[−0.2, 1.2]²`, runs a forward pass for each point, and colours pixels cyan (class 0) or purple (class 1) proportionally to the output confidence. Overlays data points as coloured dots.

- `showBoundary() → void` — opens the boundary modal and calls `renderBoundary`.
- `showBoundaryAnim() → void` — opens the modal and starts the live mode (the **▶ Live** button).
- `startBoundaryLive() / stopBoundaryLive() → void` — starts/stops a 350 ms refresh interval for live updates during training.

### 3. Network Mutation — `mutateNet(evalAfter?) → void`
Randomly perturbs 40% of weights and 30% of biases by ±`str` (from `mutStr` input). If `evalAfter` is true, evaluates the new loss; if the loss has worsened by more than 50%, the mutation is reverted using a backup. Plays a visual animation on the canvas.

### 4. Save / Load — localStorage-backed model persistence
Five slots (`NUM_SLOTS = 5`) stored in `localStorage` under the key `neurallab_v12_saves` (the key kept its v12 name so earlier saves remain loadable). Opened from the Outils tab via `showSaveLoad()`, which shows the **Save/Load modal** and calls `renderSaveSlots()` to list each slot's architecture summary, timestamp, and last loss (or an empty-slot placeholder with a 💾 save button).

- `saveToSlot(slot) → void` — serialises `NET.W`, `NET.B`, `NET.L`, `EPOCH`, the last 50 loss values, and `LAYER_ACTS` to JSON and stores them in the given slot.
- `saveNet() → void` — quick-save shortcut that finds the first empty slot (or overwrites slot 0 if all are full) and calls `saveToSlot`.
- `loadSlot(slot) → void` — restores a network from a slot, redraws, and refreshes the stats bar.
- `deleteSlot(slot) → void` — clears a specific slot.
- `exportNetJSON() → void` — downloads the full network (layers, weights, biases, epoch, layer activations, optimiser) as a `.json` file.
- `importNetJSON(input) → void` — reads a JSON file from a file input and restores the network.

### 5. LR Scheduler — `applyScheduler() → void`
Applies a learning rate schedule based on the current epoch. Called after every training step.

| Schedule | Formula |
|---|---|
| `cosine` | `lr_min + 0.5·(lr_max−lr_min)·(1+cos(π·(t mod T)/T))` |
| `step` | `lr_max · 0.5^⌊t/T⌋` |
| `exp` | `lr_max · 0.99^t` |
| `warmup` | Linear warmup for first 10% of period, then cosine annealing |

`updateSchedUI()` and `updateSchedInfo()` keep the UI controls and info label in sync.

### 6. Optimiser Race — `startRace() / stopRace() → void`
Runs a head-to-head training competition between four optimisers (SGD, Momentum, Adam, RMSProp) on the current dataset, each starting from the same weight initialisation. Displays live progress bars, loss values, accuracy, a shared loss chart, and announces the winner.

- `initRaceNets()` — creates four independent network copies with identical topology and weights but separate optimiser states.
- `raceStep()` — advances all four networks by one epoch and updates the UI.

### 6bis. Network Comparison — `openCompare() → void` (shortcut `C`)
A second head-to-head mode, in a modal (`#compare-modal-bg`), where — unlike the race — **each of the two networks has its own configuration**: layers (hidden layers free, input/output sizes forced to the dataset by `cmpSanitizeLayers`), hidden activation, output activation, optimiser (any of the six) and learning rate. Both train on the current dataset with the global loss function, for a configurable number of epochs (default 300).

- `cmpReadCfg(prefix)` / `cmpBuildNet` / `initCompareNets()` — read the two cards (`a`, `b`) and build independent networks and optimiser states.
- `cmpForward`, `cmpBackward`, `cmpUpdate` — an engine derived from the race engine, generalised to per-network settings; `startCompare()` / `stopCompare()` / `resetCompare()` run the epochs and `endCompare()` announces the result.
- Live displays: loss and accuracy of each network (`cmpAccuracy`), a shared loss chart (`drawCompareChart`) and each network's decision boundary (`cmpRenderBoundaries`, `cmpRenderOneBoundary`).
- `cmpCopyMain(prefix)` pre-fills a card from the main panel (« ↙ Copier config actuelle »); `cmpApplyToMain(prefix)` (« → Utiliser ce réseau ») loads a card's architecture, activations, optimiser and LR into the main panel and rebuilds `NET`.

### 7. Confetti — `triggerConfetti() → void`
Launches a particle animation on `#confetti-canvas` when the network reaches 100% accuracy (`checkCelebrate()`). Particles are coloured rectangles with random velocities, fading out via `animateConfetti()`.

### 8. Matrix Rain — `toggleMatrix() → void`
Overlays a Matrix-style falling characters animation on `#matrix-canvas` using `animateMatrix()`. Characters include digits, katakana, and mathematical symbols.

### 9. Sound Feedback — `toggleSound() / playTone(freq, dur, type?, vol?) → void`
Uses the Web Audio API (`AudioContext`) to play brief synthesised tones. A pitch proportional to training progress is played on each epoch step. A two-note success chord plays when the target loss is reached.

### 10. Data Flow Animation — `toggleFlow() → void`
Animates small coloured dots travelling along weight connections on the canvas via `animateFlow()`. Dots are spawned on random connections and travel from source to destination neuron. Positive-weight connections produce cyan dots; negative-weight connections produce red dots.

### 11. Weight Heatmap — `renderHeatmap() → void`
Renders a grid of small coloured squares in `#weightHeatmap`, one cell per weight. Positive weights are shown in shades of green-cyan; negative weights in shades of red-purple. The intensity is proportional to the weight's magnitude relative to the layer maximum. Cells show the exact weight value on hover.

### 12. Data Perturbation — `addNoiseToDataset() / resetDatasetNoise() → void`
Adds uniform noise `U(−σ, +σ)` (σ from `noiseLvl` input) to all input features. The original dataset is backed up to `DS_BACKUP`. `resetDatasetNoise()` restores the clean backup.

### 13. Export Canvas — `exportCanvasPNG() / copyCanvasClipboard() → void`
- `exportCanvasPNG()` — triggers a PNG download of the current canvas state, named `neural_net_e<EPOCH>.png`.
- `copyCanvasClipboard()` — copies the canvas PNG to the clipboard via the Clipboard API.

### 14. Log Export — `exportLogLatex() / exportLogMarkdown() → void`
- `exportLogLatex()` — converts the log panel's text content to a LaTeX document and triggers a `.tex` download.
- `exportLogMarkdown()` — converts the log to a Markdown document and triggers a `.md` download.

### 15. Wizard — `showWizard() / applyWizard() / closeModal() → void`
A quick-start modal that auto-configures the network (layer sizes, activation, loss, optimiser) based on the selected problem type (binary classification, multiclass, regression), number of inputs, and number of outputs.

### 16. Advanced Builder — `showAdvBuilder() / applyAdvBuilder() / closeAdvBuilder() → void`
A per-layer network configuration modal. Each layer row has independent controls for neuron count, activation function, dropout rate, and L2 regularisation. Includes quick presets (XOR, Deep, Autoencoder, 3-class, Regression, Wide). A live preview canvas shows the architecture before building.

- `abldAddLayer()` — adds a new hidden layer row.
- `abldDelLayer(i)` / `abldMoveLayer(i, dir)` — remove a layer row by index, or move it up/down.
- `abldSetN(i, n)` / `abldSetAct(i, fn)` — set a layer's neuron count or activation.
- `abldPreset(name)` — applies a named preset configuration.
- `drawAbldPreview()` — redraws the architecture preview canvas.

---

## CNN Lab (Convolutional Networks)

Opened from the Outils tab via `openCNNLab()`, the **CNN Lab** is a self-contained modal workspace (its own state object `CNN`, independent of the main MLP `NET`) dedicated to convolutional-network concepts. It has 5 sub-tabs, switched with `cnnShowTab(name)`.

### Conv — convolution playground
An interactive single-kernel convolution demo on a drawable `n×n` grid (6×6 to 12×12):
- `cnnSetGridSize(n)` / `cnnRenderInputGrid()` / `cnnToggleCell(y, x)` — resize and draw a black/white pixel grid by clicking cells.
- `cnnPreset(name)` — loads a kernel from `CNN_KERNELS`: **Identity, Blur, Sharpen, Sobel X, Sobel Y, Laplacian, Emboss**, or **Custom** (each with a short French explanation of what the kernel detects).
- `cnnKernelPresetChange()` / `cnnRenderKernelGrid()` / `cnnKernelEdit(y, x, val)` — display and hand-edit the 3×3 kernel values.
- `cnnPadInput(input, kH, kW, pad)` — zero-pads the input matrix.
- `cnnConvRaw(padded, kernel, stride)` / `cnnConv2d(input, kernel, stride, pad)` — the raw convolution math (sliding-window dot product).
- `cnnRelu(m)` — applies ReLU element-wise to a matrix.
- `cnnMaxPool2(m)` / `cnnAvgPool2(m)` — 2×2 max/average pooling.
- `cnnCompute()` — runs the full pipeline (convolution → optional ReLU → optional pooling) and renders each stage matrix via `cnnRenderMatrix(m, cellSize)` with a colour legend (`cnnColorLegend`). In matrix calculation mode it also shows the equivalent **im2col** product (`cmCnnCalcHTML`).
- `cnnRenderProgressiveMatrix(m, useRelu, cell)`, `cnnToggleAnim()` / `cnnStopAnim()` — an animated "sliding kernel window" visualisation that steps the convolution across the input one position at a time.

### Gallery — kernel gallery
`cnnGalleryRender()` applies the 7 predefined kernels to a sample image side by side, so the effect of each classic kernel (blur, sharpen, edge detectors, …) can be compared at a glance.

### Pipe — multi-layer pipeline
Chain up to four convolution → ReLU → pooling layers and see the shrinking feature-map sizes end-to-end:
- `cnnPipeInitDefault()`, `cnnPipeAddLayer()`, `cnnPipeRemoveLayer(i)`, `cnnPipeUpdateLayer(i, field, val)` — build/edit an ordered list of pipeline layers (kernel choice, pooling, stride).
- `cnnPipeRender()` — draws the layer chain.
- `cnnPipeCompute()` — executes the full pipeline on the current input grid, showing every intermediate feature map.

### Train — kernel training (gradient descent on a single kernel)
A minimal supervised-learning demo where a 3×3 kernel is learned from scratch towards a target filter's output:
- `cnnMSE(pred, target)` — mean-squared error between the conv output and a target feature map.
- `cnnTrainGrad(padded, pred, target)` — computes the gradient of the MSE loss with respect to every kernel weight and updates the kernel.
- `cnnLossSparkline(hist, w, h, color)` — draws a small loss-history sparkline.
- `cnnTrainSetup()`, `cnnTrainStep()`, `cnnTrainRender()`, `cnnTrainToggle()` (auto-run), `cnnTrainStop()`, `cnnTrainReset()` — the training-loop controls.

### Model — end-to-end image classifier
A genuine (small) trainable CNN (`CNN_MODEL`), independent of the main `NET`/`applyGrad` engine, for classifying hand-drawn images into user-defined classes. Sub-tabs (`cnnModelShowSubtab`): configuration, dataset, run, test.
- **Image size** — `cnnModelSetSize(n)` accepts 8×8 up to **1536×1536** (default 10×10); `cnnModelUpdateSizeWarn()` warns when the size makes training heavy.
- **Drawing surfaces** — up to 32×32 (`CNN_PIXEL_THRESHOLD`) images are edited as a grid of cells (`cnnModelRenderEditableGrid`, `cnnModelToggleDraftCell`); above that, a real **bitmap canvas editor** takes over (`cnnEditorCreate`, `CNN_EDITORS`) with brush, pencil, eraser, line, rectangle, ellipse and flood fill tools, brush size, filled/outline shapes, black/white colour, undo, clear and invert (`cnnEditorSetTool/SetColor/BrushChange/FillToggle/Undo/Clear/Invert`, `cnnEditorLoadGrid`); `cnnGridToImageData` / `cnnImageDataToGrid` / `cnnGridThumbURL` convert between canvas pixels and the model's grids.
- **Dataset** — `cnnModelRenderClasses()`, `cnnModelAddClass()`, `cnnModelRemoveClass(i)` (colour swatches via `cnnModelClassColor`); `cnnModelDraftPreset(name)` and `cnnModelImportFromConv()` fill the draft; `cnnModelAugment(grid)` generates augmented variants (flips/shifts/noise) when adding a sample with `cnnModelAddSample(withVariations)`; `cnnModelDeleteSample(idx)`/`cnnModelClearDataset()`; `cnnModelRenderDsStats()`/`cnnModelRenderDatasetGrid()`; `cnnModelGenerateDemo()`; `cnnModelExportJSON()`/`cnnModelImportJSON()`.
- **Model** — `cnnModelBuild()` assembles **1 or 2 blocks of Conv (3×3, "same" padding) + ReLU [+ MaxPool 2×2]**, then Flatten → optional Dense(ReLU) → Dense + Softmax (`cnnModelShape`, `cnnModelConfigChanged`, `cnnModelBlockForward`, `cnnModelMaxPool2WithArgmax`); `cnnModelResetWeights()`; `cnnModelRenderArch()` draws the diagram.
- **Engine** — `cnnModelForward(grid)` → class probabilities; `cnnModelBackward(cache, labelIdx)` — full backpropagation through the dense layers, unpooling (stored argmax indices) and every convolution; `cnnModelApplyGrad(grads, lr)` — SGD update (default lr 0.03; one step = one example, one epoch = the whole shuffled dataset).
- **Run** — `cnnModelEpoch()` / `cnnModelEpochs(n)`, `cnnModelTrainToggle()` / `cnnModelTrainStop()`, `cnnModelRenderTrain()` (loss/accuracy curves).
- **Test** — `cnnModelRenderTestGrid()`/`cnnModelToggleTestCell(y,x)`/`cnnModelTestPreset(name)`/`cnnModelTestImportFromConv()`; `cnnModelPredict()` (predicted class with per-class confidence); `cnnModelTestAll()` (accuracy over the dataset).

---

## RNN Lab (Recurrent Networks)

Opened from the Outils tab via `openRNNLab()`, the **RNN Lab** is a second self-contained modal workspace dedicated to recurrent networks, with 4 sub-tabs switched with `rnnShowTab(name)` (function names use the `rnn` prefix).

### Cellule — single recurrent-cell step
Visualises one RNN cell's computation, `h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)`, for a single time step: hidden size 2, 3 or 4; editable `x`, `h(t−1)`, `W_x`, `W_h`, `b` (`rnnCellEditWx/Wh/B/HPrev`, random weights, reset of `h`); `rnnCellCompute()` / `rnnCellRenderResults()` show the weighted sums, `tanh` and the new state numerically and with a cell diagram (`rnnCellDiagramSVG`); `rnnCellAdvance()` copies `h(t)` into `h(t−1)` for the next step. In matrix mode the same step is shown as block products (`cmRnnCellHTML`).

### Déroulement — sequence unrolling
Feeds a whole input sequence (presets *Impulsion, Alternance, Rampe, Bruit* or a typed list, `rnnSeqParse`) through the same cell with the weights of the Cellule tab (`rnnSeqRun`), rendering the hidden-state heat map (`rnnSeqRenderHeatmap`), per-step strips (`rnnSeqRenderStrip`) and the trace of a chosen neuron (`rnnSeqRenderChart`); `rnnSeqToggleAnim` animates the unrolling step by step.

### Mémoire & gradients — vanishing/exploding gradients (BPTT)
A demonstrator of the central RNN training problem using a simplified model: the gradient travelling from `h(T)` back to `h(1)` crosses a product of `T` terms, each approximated by `w × s` (effective magnitude of `W_hh` times the mean `tanh′`). Two sliders (magnitude of `W_hh`, mean saturation of `tanh′`) show the resulting magnitude after `T` steps (`rnnGradRender`), with an explanation of how the additive LSTM cell state `C(t) = f⊙C(t−1) + i⊙C̃` mitigates it.

### Entraîner — trainable character-level RNN
A genuine mini character-level language model trained with **real BPTT and Adagrad** (`rnnTrainStep`, `adaUpd`), with four sub-tabs (`rnnTrainShowSub`):
1. **Dataset** — a text corpus (presets *Un mot, Phrase, Petit poème, Alphabet, Nombres*, or your own) validated with `rnnDatasetApply`, with vocabulary statistics.
2. **Modèle** — hidden size H ∈ {8, 16, 32, 64}, BPTT window ∈ {8, 16, 25, 40}, learning rate ∈ {0.03 … 0.3}, gradient clipping ∈ {±1, ±5, ±10}; `rnnModelBuild()` initialises `Wxh, Whh, Why`.
3. **Entraînement** — each step draws a random window from the corpus, computes the loss through BPTT and updates the weights; buttons « Entraîner » (auto), « Un pas », « +50 pas »; live loss curve and weight heat maps (`rnnWeightHeatmapHTML`, `rnnTrainRender`).
4. **Test** — type a primer text (« amorce »): `rnnTestPredictNext()` shows the next-character distribution, `rnnTestGenerate()` samples a continuation.

---

## Transformer Lab

Opened from the Outils tab via `openTFLab()` (modal `#tf-modal-bg`, state `TF`, tabs switched with `tfShow(name)`; each tab is built lazily on first display). It explains the architecture behind GPT, Llama or Claude and shows every calculation; it does **not** train a Transformer.

| Tab | Content |
|---|---|
| **Théorie** (`tfBuildTheory`, `TF_STEPS`) | A step-by-step walk from text to next token: ① tokens (BPE / SentencePiece, vocabulary ≈ 32k–256k) → ② embeddings `X₀ = E[ids] + PE` → ③ attention → … , with the headline formula `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V` |
| **Attention** (`tfBuildAttn`, `tfAttn`, `tfCompute`) | Type a sentence: each word gets a deterministic pseudo-random embedding plus position and goes through **real multi-head attention** (d = 16, heads h ∈ {1, 2, 4}, adjustable weight gain). Weights are random (untrained) — the goal is to see the mechanics. A heat map of the attention matrix, with a click on a row detailing that token's `Q`, `K`, `V`, scores, softmax and output |
| **Positions** (`tfBuildPos`, `tfPos`) | Why positions are needed (attention is permutation-invariant) and the four answers: sinusoidal `PE(p,2i) = sin(p/10000^{2i/d})`, learned (GPT-2), **RoPE** (rotation of Q and K, Llama/Mistral/Qwen) and **ALiBi** (linear bias on scores) |
| **Décodage** (`tfBuildDec`, `tfDec`, `tfDraw`, `TF_V`) | From logits to a token on a toy vocabulary (« Le chat dort sur le … »): greedy, temperature `softmax(z/T)`, top-k, top-p (nucleus), repetition penalty, with sampling counters to see the distribution |
| **Taille & coût** (`tfBuildSize`, `tfSize`, `tfPreset`, `TF_PRE`) | Parameter count, FLOPs and memory (including the KV-cache) from the configuration (`d`, layers, heads, KV heads, FFN width, vocabulary, context…), with presets **GPT-2 small**, **Llama-3 8B**, **Llama-3 70B** and **Mini (jouet)** |

In matrix calculation mode, the attention tab also shows the matrix form `A = softmax_rows(QKᵀ/√d_k)`, `Out = AV` (`cmTfDetHTML`).

---

## RL Lab

Opened from the Outils tab via `openRLLab()` (modal `#rl-modal-bg`, tabs switched with `rlShowTab(name)`, helper `$rl(id)`, seeded random generator `rlMulberry`, shared UI helpers `rlSlider`, `rlPlot`, `rlTheory`, `rlLegend`). Each tab contains the theory, interactive experiments and the detail of every calculation. `Escape` or closing the modal stops all running loops (`rlStopAll`).

| Tab | Content |
|---|---|
| **Concepts** (`rlConceptsBuild`) | Agent/environment loop, return and discount γ (interactive γ explorer) |
| **Bandits** (`RLB`, `rlbBuild`) | A K-armed bandit (Bernoulli or Gaussian rewards, default K = 5, seeded). Algorithms run in parallel: **greedy, ε-greedy, UCB1, softmax, Thompson sampling**, plus manual play (« Vous »); step / run / auto controls, regret and arm-estimate plots, option to reveal the true arm values |
| **Calculateur** (`rlcBuild`, `rlcAll`) | Six live calculators with editable inputs: ① discounted return `G_t`, ② TD(0) error and `V` update, ③ Q-learning vs SARSA vs Expected SARSA on one transition, ④ exploration policies from `Q` (ε-greedy, softmax with τ, UCB with c), ⑤ GAE (γ, λ), ⑥ one softmax policy-gradient step |
| **Q-Learning / SARSA** (`RLG`, `rlgBuild`) | A grid world with presets *Simple 5×5, Falaise (cliff), Labyrinthe, Terrain ouvert 5×6, Glissant (stochastic)*, editable cells (walls, trap, goal, start), rewards (goal +10, trap −10, step −0.1 by default), slip probability, ε-greedy exploration with decay. Algorithms **Q-learning, SARSA, Expected SARSA**; views *Q per action*, *V = max Q*, *policy*; single steps, episodes, auto-run, greedy-path display and comparison with the DP solution (`rlgCompareDP`) |
| **Bellman (DP)** (`RLD`, `rldBuild`) | Dynamic programming on the same grid: **value iteration** and **policy evaluation / improvement**, sweep-by-sweep with the Bellman backup of a selected state, convergence curve of `Δ` |
| **Policy Gradient** (`RLR`, `rlrBuild`) | Tabular softmax policy `θ`: **REINFORCE (no baseline)**, **REINFORCE + mean baseline**, **Actor-Critic (online TD)** with a value critic; per-episode probabilities and returns |
| **DQN (CartPole)** (`RLCP`, `rlqBuild`) | The CartPole physics (g = 9.8, cart 1.0, pole 0.1, length 0.5, force ±10, τ = 0.02, limits ±12° and ±2.4) with a small MLP Q-network written from scratch (`rlNetInit/Forward/Backward/Adam/Copy`): **replay buffer** (10 000 transitions), **target network** (sync period configurable), ε decay down to 0.05, optional **Double DQN**, lr/γ/sync/decay sliders, 1× or turbo (no rendering) speed, live cart drawing and return curve, test run |

---

## Code & Data Export

Beyond PNG/clipboard/LaTeX/Markdown log export and the save-slot JSON export already covered above, two further export paths exist:

### `exportCode(lang) → void`
Generates a complete, runnable standalone implementation of the **current network** (architecture, weights, biases, activation functions) in either **Python** (NumPy) or **JavaScript**, including a forward-pass function, so the trained model can be used outside the browser. The body follows the current calculation mode (`cmExportBody`): explicit loops neuron by neuron in individual mode, matrix products (NumPy `W @ a + b` in Python) in matrix mode. The generated source is shown in a code-preview modal with a copy-to-clipboard button.

### Dataset JSON import/export
`exportDSJSON()` downloads the current dataset (`DS.inputs`/`DS.outputs`) as a `.json` file; `importDSJSON(input)` reads a JSON file back in and rebuilds the dataset editor (`rebuildDSEditor()`) and preview (`renderDSPreview()`), calling `applyDS()` to commit it as the live dataset. The CNN Lab's image dataset has its own JSON import/export (`cnnModelExportJSON` / `cnnModelImportJSON`).

---

## Automatic FR → EN Translation Layer

A self-contained script block (its own inline `<script>` block) adds a one-click interface translator, toggled from the **Options → Langue** section via `toggleTranslation()`.

- It walks the visible DOM text nodes (skipping elements marked `data-no-translate`, `<script>`/`<style>` contents, and form-input values where translation would break functionality) and batches their text for translation.
- Translation requests are sent to a public, keyless translation endpoint (Google Translate's unofficial `translate_a/single` endpoint), with a fallback to the MyMemory translation API if the primary request fails.
- Translated strings are cached so repeated toggles or re-renders don't re-request the same text.
- Because it works by rewriting DOM text after the fact, it is a purely cosmetic UI localisation layer — all underlying variable names, stored data, and exported code/log content remain in French.

---

## UI Utility Functions

### Logging

#### `log(html) → void`
Appends a raw HTML string to the `#log` element and auto-scrolls to the bottom.

#### `clearLog() → void`
Empties the log panel.

#### `logBlock(cls, icon, title, body, startOpen?) → void`
Appends a **collapsible structured block** to the log with a coloured header bar (forward=cyan, loss=red, backprop=orange, update=green). Used during verbose training. Each block can keep two renderings (`blk._bodies = {ind, mat}`) so that the calculation-mode switch re-renders existing log entries without recomputing.

#### `toggleLogBlock(id) → void`
Toggles visibility of a log block body identified by `id`.

#### `logEpoch(avg) → void`
Appends a formatted epoch summary line with loss (colour-coded), trend arrow (% change from previous epoch), accuracy, and optimiser info.

#### `copyLog() → void`
Copies the log panel's full plain-text content (`innerText`) to the clipboard.

#### `logFull(x, y, activations, zs, deltas, ll) → void`
Generates the verbose, fully-annotated backprop log entry for one training sample (called from `trainSample`), with the level of detail controlled by the `#logLevel` select (`ll`). It renders true mathematical notation (fractions, subscripts/superscripts, Σ summation bars, per-symbol legends) using a small set of formatting helpers rather than plain text:

- `mFrac(num, den)` — renders a stacked fraction span.
- `mSub(base, sub)` / `mSup(base, sup)` — renders a subscript/superscript.
- `mSigma(from, to)` — renders a Σ summation with limits.
- `actDeriv(fn)` — returns the textual derivative formula for a given activation function (e.g. `σ(z)·(1−σ(z))` for sigmoid), used to annotate the backprop phase.
- `legendRow(sym, symCls, desc)` — renders one row of a symbol legend (e.g. `δ = error term for this neuron`).
- `matrixNotesHTML()` — in matrix mode, the collapsible « Dimensions des matrices » box prepended to each of the four phases (see [Calculation Mode](#calculation-mode-individual--matrix)).

### Canvas interaction

#### Canvas click handler
Detects clicks on neurons (within radius `r`, via `cpToCanvas()` coordinate mapping) and connections (within a few px of the line, via `lpd()` point-to-segment distance). Plain clicks open a **detail popover**:

- `showNodeDetail(l, j)` — shows a neuron's layer/index, activation function, bias, last pre/post-activation values (`z`, `a`), and gradient (`δ`) when available.
- `showLineDetail(l, j, k)` — shows a connection's source/target neuron, current weight, last gradient, and the accumulated optimiser state (velocity/moment) for that weight.
- `copyDetailJSON()` — copies the currently-open node/line detail popover's data to the clipboard as JSON.

**Modifier-key interactions** on neurons/connections:

| Combo | Effect |
|---|---|
| `Ctrl+Click` on a neuron | Pins ("forces") its activation to a user-specified value for subsequent forward passes |
| `Shift+Click` on a neuron | Freezes/unfreezes its **bias** (toggles `NET.frozenNodes["(l)-(j)"]`) — frozen biases are skipped during `applyGrad()` and shown with a ❄ icon |
| `Shift+Click` on a connection | Freezes/unfreezes that **weight** (toggles `NET.frozenWeights["l-j-k"]`) — frozen weights are skipped during `applyGrad()` |
| `Alt+Click` on a neuron | Opens a small menu to force the neuron's activation to follow a **pattern** each forward pass: constant `0`, constant `1`, a `sine` wave, `random` noise, or `none` (clears the pattern) |

#### `showTestInputs() → void`
Populates the **Test** tab's input fields from the currently selected dataset sample, so the user can immediately run a forward pass on a real example.

### Resize system

#### `initColResize(handleId, targetId, side) → void`
Attaches mouse drag listeners to a column resize handle, allowing the left or right panel to be resized within their CSS `min-width`/`max-width` bounds. Double-click resets to the default width (260 px for left, 370 px for right).

#### `initRowResize(handleId, targetId) → void`
Attaches mouse drag listeners to a row resize handle for the log panel. Double-click resets to 190 px height.

### Tooltip

#### `showTip(html, x, y) → void`
Positions and displays the floating tooltip `#tip` near canvas coordinates `(x, y)`.

#### `hideTip() → void`
Hides the tooltip.

### Progress bar

The `#pb` element is updated during `doN()` to show training progress as a horizontal fill.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | `doStep()` — single training step |
| `E` | `doEpoch()` — one full epoch |
| `A` | `toggleAuto()` — toggle auto-training |
| `M` | `mutateNet(false)` — mutate weights |
| `B` | `showBoundary()` — decision boundary |
| `R` | `showRace()` — optimiser race |
| `C` | `openCompare()` — compare two networks |
| `T` | `showTab('tools')` — switch to Tools tab |
| `G` | `analyzeGradients()` — gradient analysis |
| `H` | `renderHeatmap()` — weight heatmap |
| `Escape` | Close all modals (boundary, save/load, race, compare, CNN, RNN, RL, Transformer, visual dataset editor) and stop live animations and training loops started from them |
| `←` / `→` / `Space` | While the 🎓 step-by-step panel is open: previous step / next step / play-pause (it takes precedence over the global shortcuts) |

Shortcuts are disabled when focus is inside `<input>`, `<textarea>`, or `<select>` elements.

---

## Mobile Support

On viewports ≤ 900 px the layout switches entirely via CSS media queries:

- The three panels stack and are shown one at a time, each taking the full viewport.
- A **top bar** (`#mobile-topbar`) replaces the left panel header, showing the app title plus a live epoch/loss pill.
- A **quick-train bar** (`#mobile-trainbar`) below the top bar provides one-tap access to Step, Epoch, Auto, and ×100 buttons.
- A **bottom navigation bar** (`#mobile-nav`) switches between Config, Network, and Details panels, with a sliding indicator.
- Resize handles are hidden on mobile.
- Modals slide up from the bottom as sheets; the labs open full-width with a sticky header.
- The log panel has a fixed 160 px height.

A dedicated stylesheet (`#mobile-readability-v13`, entirely inside `@media (max-width: 900px)` so the desktop UI is unchanged) improves legibility and touch comfort:

1. a monotone mobile **type scale** (small desktop font sizes are raised, hierarchy preserved);
2. a fix for tiny **inline font sizes**, including those generated by JavaScript templates;
3. larger **touch targets** for buttons, button rows and small controls;
4. a three-line header for the « Calculs détaillés » log blocks;
5. the individual/matrix **switch** restyled for touch wherever it appears;
6. wrapping button rows and touch responsiveness;
7. bigger sliders and checkboxes;
8. the bottom training bar keeps its buttons inside its 34 px height.

#### `mobNav(panel) → void`
Activates the named panel (`'left'`, `'center'`, or `'right'`) by toggling the `mob-active` class and the matching button. On switching to the canvas panel, a 60 ms timeout ensures `resizeCvs()` and `redraw()` fire after the layout is applied. `slideMobNavIndicator(panel)` moves the sliding indicator (and re-positions on resize / rotation).

#### `syncMobStats() → void`
Called every 400 ms via `setInterval`. Copies the epoch and loss text content from the left panel stats display to the top bar pill elements.

---

## Local Dev

Tip to have **hotreload** (*nodejs* must be installed)

```bash
npx live-server .
```

---

## Colour Palette Reference

All colours are defined as CSS custom properties on `:root`:

| Variable | Value | Usage |
|---|---|---|
| `--bg` | `#05060c` | Main background |
| `--p1` | `#0a0c18` | Panel background |
| `--p2` | `#0d1020` | Section/header background |
| `--brd` | `#192040` | Border colour |
| `--a1` | `#00e5ff` | Primary accent (cyan) |
| `--a2` | `#bf5fff` | Secondary accent (purple) |
| `--grn` | `#00ff9d` | Success / positive |
| `--flashy` | `#9dff00` | Highlight (lime) |
| `--ylw` | `#ffd600` | Warning / highlight |
| `--red` | `#ff3d5a` | Error / loss |
| `--org` | `#ff8c2a` | Backprop / alert |
| `--text` | `#b8cce8` | Primary text |
| `--dim` | `#80e1ff` | Secondary / muted text |
| `--glow1`, `--glow2` | cyan / purple glow shadows | Glow effects |

This is the default theme. The additive **Aurora** stylesheet slightly re-tints the background and panel variables (`--bg: #04050b`, `--p1: #090b1a`, `--p2: #0d1126`, `--brd: #1b2447`) and adds a gradient accent line `--line`. The Options tab's "Thème & Couleurs" section offers six canvas colour presets — Dark, Neon, Ocean, Fire, Matrix, Pastel — applied via `applyTheme(name)`, which sets the canvas colour pickers (neuron, weight and background colours) and redraws.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| KaTeX | 0.16.9 | Mathematical formula rendering (formula library, calculation views, labs) |
| JetBrains Mono | Google Fonts | Monospace UI font |
| Syne | Google Fonts | Display / heading font |

All other functionality is implemented in vanilla JavaScript with no bundled runtime dependencies. Two optional features use the network or newer browser APIs: the **auto-translation layer**, which makes live calls to a public translation API when enabled by the user (see [Automatic FR → EN Translation Layer](#automatic-fr--en-translation-layer)), and **WebGPU training**, which relies on the browser's `navigator.gpu`. KaTeX and the fonts are loaded from CDNs; apart from that, the app is fully offline-capable.

