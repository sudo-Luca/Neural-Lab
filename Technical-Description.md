# Neural Lab v13

> **A self-contained, pedagogical deep learning simulator built entirely in a single HTML file.**

Neural Lab v13 is an interactive browser-based tool for visualising, training, and understanding multilayer perceptrons (MLPs). It requires no server, no framework, and no installation — just open the HTML file in any modern browser. Every forward pass, backpropagation step, gradient computation, and weight update is performed in vanilla JavaScript and rendered in real time on an HTML5 Canvas, making it ideal for learning, teaching, or experimenting with neural network fundamentals.

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
9. [Training Control Functions](#training-control-functions)
10. [Visualisation & Canvas Rendering](#visualisation--canvas-rendering)
11. [Dataset Management](#dataset-management)
12. [Right-Panel Tabs & Inspection Tools](#right-panel-tabs--inspection-tools)
13. [Manual Calculation Mode (« Manuel »)](#manual-calculation-mode--manuel-)
14. [v12 Extension Features](#v12-extension-features)
15. [CNN Lab (Convolutional Networks)](#cnn-lab-convolutional-networks)
16. [RNN Lab (Recurrent Networks)](#rnn-lab-recurrent-networks)
17. [Code & Data Export](#code--data-export)
18. [Automatic FR → EN Translation Layer](#automatic-fr--en-translation-layer)
19. [UI Utility Functions](#ui-utility-functions)
20. [Keyboard Shortcuts](#keyboard-shortcuts)
21. [Mobile Support](#mobile-support)
22. [Local Dev](#local-dev)
23. [v13 Additions](#v13-additions) — [WebGPU engine](#webgpu-training-engine), [Training techniques](#training-techniques), [Visual dataset editor](#visual-dataset-editor), [Comparison mode](#comparison-mode), [Matrix calculation mode](#matrix-calculation-mode), [Step-by-step mode](#step-by-step-explanation-mode), [Quiz](#quiz), [Transformer Lab](#transformer-lab), [RL Lab](#rl-lab), [Visual layer](#visual-layer--performance)

---

## Overview

Neural Lab v13 is structured as a **single HTML document** (~30,000 lines) combining:

- **HTML** — three resizable panels (left config, centre canvas, right details) plus several modal dialogs (CNN Lab, RNN Lab, Save/Load slots, node/line detail popovers).
- **CSS** — a custom dark-mode design system using CSS variables for a consistent colour palette, with several selectable colour themes.
- **Vanilla JavaScript** — all neural network logic, rendering, and interactivity, split across **eleven inline `<script>` blocks** (plus the KaTeX `<script src>`):
  1. **Main core** (≈ lines 8291–19739) — architecture, activations, losses, optimisers, Batch Norm / augmentation / early stopping, **WebGPU training engine**, training loop, canvas rendering, dataset + visual dataset editor, quiz, individual/matrix calculation mode, formula library, manual-calculation mode, code export.
  2. **Mobile helpers** (≈ lines 19762–19839) — mobile navigation and stat-bar sync.
  3. **Extension block** (≈ lines 21964–27060; modals HTML at 19841–21963) — decision boundary, optimiser race, **comparison mode**, mutation, save/load, LR scheduler, gradient/layer/benchmark tools, effects, **CNN Lab**, **RNN Lab**, keyboard shortcuts.
  4. **Translation layer** (≈ lines 27062–27490) — automatic FR ↔ EN interface translation.
  5. **RL Lab** (≈ lines 27493–28282) — bandits, calculators, grid-world Q-learning/SARSA, Bellman DP, policy gradient, DQN on CartPole.
  6. **Sliding tab indicator** (≈ 28289–28414), 7. **Satisfaction pack** (ripples, spotlight, sparks, counters; ≈ 28421–28640), 8. **Satisfaction pack 2** (form-control polish, slider value bubble; ≈ 28762–28832) — purely cosmetic, skipped under `prefers-reduced-motion`.
  9. **Transformer Lab** (≈ lines 28833–29032).
  10. **Aurora / compaction / mobile-readability styles + collapsible sections script** (≈ lines 29034–29255).
  11. **Step-by-step explanation mode** (≈ lines 29665–30016).

There are no bundled external JavaScript dependencies except the **KaTeX** library (loaded via CDN) for rendering mathematical notation in the formula library. The automatic translation layer additionally calls a public translation API (Google Translate's unofficial endpoint, with a MyMemory fallback) over the network at runtime — it is not a bundled dependency and only activates when the user enables translation.

### Key capabilities

| Category | Details |
|---|---|
| Architectures | Fully-connected MLP, arbitrary depth and width, per-layer activations |
| Activations | Sigmoid, ReLU, Tanh, Leaky ReLU, ELU, Swish, GELU, SELU, Softsign, Linear, Softmax |
| Loss functions | MSE, MAE, Log Loss (BCE), Huber, Hinge |
| Optimisers | SGD, Momentum, RMSProp, Adam, AdamW, Nesterov AG |
| Weight init | Xavier/Glorot, He/Kaiming, Uniform, Normal, Small |
| LR schedulers | Cosine annealing, Step decay, Exponential decay, Warmup-Cosine |
| Training modes | Single step, single epoch, N epochs, auto-run, train-to-target, optional WebGPU batch training |
| Regularisation / techniques | L2 weight decay, global gradient-norm clipping, dropout, per-100-epoch LR decay, **Batch Normalization**, **input-noise data augmentation**, **early stopping** |
| Interactivity | Ctrl+click a neuron to pin its activation, Shift+click a neuron/weight to freeze it, Alt+click to force a pattern (0 / 1 / sine / random) |
| Inspection | Step-by-step log, gradient analysis, layer inspector, weight heatmap, zoomed loss chart, node/connection detail popovers |
| Export | PNG canvas, clipboard copy, LaTeX log, Markdown log, JSON model, save/load slots, **Python / JavaScript code generation**, dataset JSON import/export |
| v12 extras | Decision boundary visualiser, optimiser race, network mutation, confetti, Matrix rain, data flow animation, sound feedback, save/load slots, colour themes |
| Manual mode | Step-by-step "do the math yourself" quiz (forward pass, loss, deltas, weight updates) checked against the real engine |
| CNN Lab | Standalone convolution playground, kernel gallery, multi-layer pipeline visualiser, kernel training, and an image-classifier modal |
| RNN Lab | Standalone recurrent-cell step-through, sequence unrolling, vanishing/exploding-gradient (BPTT) demo, and a trainable character-level RNN |
| Transformer Lab | Pipeline, live multi-head attention, positional encodings (sinusoidal / RoPE), decoding (temperature / top-k / top-p), params / FLOPs / KV-cache calculator |
| RL Lab | Bandits, return/TD/GAE calculator, Q-learning/SARSA, Bellman DP, REINFORCE/Actor-Critic, DQN CartPole |
| Quiz | 126 questions (QCM / true-false / open) in 6 chapters |
| Step-by-step mode | Animated input → z → a → error → gradient → update walkthrough with canvas highlighting |
| Calculation display | Individual (per-neuron) or matrix notation, everywhere |
| Compare mode | Two independently configured networks trained in parallel |
| Localisation | One-click automatic FR → EN interface translation (external API, DOM-based) |

---

## Interface Layout

The UI is divided into three resizable panels separated by drag handles (`.rsz-col`) and a vertically resizable log panel (`.rsz-row`).

```
┌──────────────┬────────────────────────────┬───────────────────┐
│  LEFT PANEL  │     CENTER — Canvas        │   RIGHT PANEL     │
│  (260 px)    │   Neural network diagram   │   (370 px)        │
│              │                            │                   │
│  Architecture│                            │  Tabs (9):        │
│  Training    │                            │  - Détails        │
│  Display     │                            │  - Test           │
│              ├────────────────────────────┤  - Dataset        │
│              │   LOG PANEL (190 px)       │  - Formules       │
│              │   Structured backprop log  │  - Options        │
│              │                            │  - Guide           │
│              │                            │  - Quiz            │
│              │                            │  - Manuel          │
│              │                            │  - Outils          │
└──────────────┴────────────────────────────┴───────────────────┘
```

Modal overlays sit above this layout: the **CNN Lab**, **RNN Lab**, **Transformer Lab** and **RL Lab** (opened from **Outils**), the **visual dataset editor** (from **Dataset**), plus boundary / race / compare / save-load modals. Each lab is a self-contained multi-tab workspace (see their respective sections below).

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
| `RAF_ID` | `number` | `requestAnimationFrame` handle for the auto-training loop |
| `SELECTED` | `object` | Currently selected neuron (for tooltip) |
| `BN` | `{gamma,beta,runMean,runVar}` | Batch-Norm parameters/statistics per hidden layer (kept consistent by `ensureBN()`) |
| `BN_ZHAT`, `BN_GRAD` | arrays | Normalised pre-activations of the last forward / γ,β gradients of the last backward |
| `EARLY` | `{bestLoss, badEpochs}` | Early-stopping tracker |
| `PERF` | `{fastTrainLite,_savedGlow}` | Performance prefs (glow is lightened during auto-train) |
| `GPU` | object | WebGPU state: `supported, ready, device, pipelines, verified, busy, main` |
| `CALC_MODE` | `"indiv" \| "matrix"` | Calculation display mode (persisted in `localStorage` `nl_calc_mode`) |
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

### `resetW() → void`

Re-randomises all weights and biases of the existing network using the current initialisation method. Resets epoch, loss history, and optimiser state but preserves the network topology.

### `setBias(l, j, val) → void`

Manually sets the bias of neuron `j` in layer `l` to a parsed float value. Redraws and logs the change. Used from click-to-edit interactions in the detail panel.

### `forward(inputs, isTrain?) → {activations, zs}`

Performs a **forward pass** through the entire network.

**Returns:**
- `activations` — 2-D array of shape `[layers+1][neurons]`; `activations[0]` is the input layer.
- `zs` — 2-D array of shape `[layers][neurons]`; the pre-activation value at each non-input neuron.

**Algorithm:**
1. Seeds the activation array with the input vector.
2. For each layer `l`, computes the dot product `z[j] = Σ(W[l][j][k] · a[l][k]) + B[l][j]`.
3. Applies the layer's activation function (via `getActForLayer`).
3b. On hidden layers with **Batch Normalization** on, `z` is normalised with running mean/variance then scaled/shifted: `z_eff = γ·ẑ + β` (stats updated with the BN momentum when `isTrain`).
3c. Node patterns (`0`, `1`, `sin`, `random`) override the activation; **dropout** zeroes hidden activations at the configured rate when > 0.
4. For the last layer, applies **softmax** if the output activation is `"softmax"`.
5. Stores results in `LAST_ACT` and `LAST_Z` for inspection and logging.

### `backward(activations, zs, y) → deltas`

Performs **backpropagation** and returns the delta matrix without updating weights.

**Returns:**
- `deltas` — 2-D array of shape `[layers][neurons]`, one delta per neuron.

**Algorithm:**
1. Computes the output layer deltas: `δ[j] = (∂L/∂a) · f'(z)` using `computeLossGrad` and `actD`.
2. For softmax outputs, approximates `f'(z) = a·(1−a)` (diagonal of Jacobian).
3. Propagates error backwards through hidden layers: `δ[l][k] = (Σⱼ δ[l+1][j] · W[l+1][j][k]) · f'(z[l][k])`.
4. Stores deltas in `LAST_DELTAS` for logging.

### `trainSample(inputs, y, verbose?) → number`

Runs a complete **forward + backward + weight update** cycle for a single training example. If **data augmentation** is on, Gaussian noise `σ·N(0,1)` is added to the input for training only (`DS.inputs` is never modified).

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

- **L2 weight decay** — if the "L2" hyper-parameter (`cL2`) is > 0, it is added directly to each weight's gradient: `g = dj·activation + l2·w`.
- **Global gradient-norm clipping** — if the "Grad Clip" hyper-parameter (`cGC`) is > 0, the L2 norm of all delta values is computed and, if it exceeds the threshold, every delta is rescaled by `threshold / norm` before weights are updated.
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
- Dropout rate and Huber-loss delta are also configurable here (`cDrop`, `cHuber`)

---

## Training Control Functions

> **v13:** `doStep`, `doEpoch`, `doN` and `runAuto` are `async`. When `gpuEligible()` is true they run on the GPU (see [WebGPU engine](#webgpu-training-engine)); `doEpoch`/`doN` call `earlyStopUpdate(avgLoss)` and halt auto-training via `stopAutoWithMessage()` when patience is exhausted. `toggleAuto` temporarily disables glow when `PERF.fastTrainLite` is set.

### `doStep() → void`

Runs one training sample (the next in the dataset in round-robin order). Updates stats, redraws the canvas, and draws the loss chart. In v12, also hooks into `updateNetStatsBar()`, `checkCelebrate()`, and `applyScheduler()`.

### `doEpoch() → void`

Iterates over **every sample in the dataset once**, accumulating the total loss, then increments `EPOCH`, pushes the average loss to `LOSS_H`, updates stats, redraws, and logs the epoch summary with trend arrows.

### `doN(n) → void`

Runs `n` training steps in a synchronous loop (using `requestAnimationFrame` batching for large `n` to avoid blocking the UI). Progress is shown in the progress bar (`#pb`).

### `toggleAuto() → void`

Toggles the auto-training loop. When active, calls `doStep()` repeatedly using `requestAnimationFrame`, with a configurable delay from the `autoSpeed` slider (0–500 ms per batch). The `btnAuto` button visually changes to `pulse` animation.

### `trainToTarget() → void` (v12 §16)

Trains the network epoch by epoch until either the average loss drops below the target value from `lossTarget` or a maximum of 10 000 epochs is reached. Updates the `target-status` element with live progress. Plays a success tone on convergence if sound is enabled.

### `stopTargetTrain() → void`

Clears the `targetTrainTimer` timeout used by `trainToTarget()`.

### `acc() → number`

Computes the current classification accuracy (in %) over the full dataset. A prediction is considered correct if `Math.round(p) === y` for all outputs. Returns 0 if the dataset is empty.

### `updateStats(loss) → void`

Updates the four stat cards in the left panel (`sEpoch`, `sLoss`, `sAcc`, `sLR`) and redraws the mini loss chart.

### `drawLossChart() → void`

Renders the small loss curve in the left panel (`#lossChart`) using an HTML5 Canvas. Draws the raw loss curve in `--red` and, when more than 10 data points are available, a 10-point moving average in `--ylw` (dashed).

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

### `loadPreset(name?) → void`

Loads a built-in dataset preset from the `PRESETS` table (selector `dsPreset`):
- `xor`, `and`, `or`, `nand`, `xnor` — logic gates (2→1)
- `half_adder` — half adder (2→2)
- `4bit` — 4-bit identity (4→4)
- `custom` — keeps/clears the dataset for manual entry

Richer 2-D problems (circles, spiral, checkerboard, blobs; linear / quadratic / sine regression) are produced by the [visual dataset editor](#visual-dataset-editor).

Resets epoch and loss history. Rebuilds the dataset editor UI.

### `buildDSEditorFromCurrent() → void`

Regenerates the dataset editor rows in `#dsEditor` from the current `DS.inputs` and `DS.outputs` arrays.

### `readDSFromEditor() → void`

Reads all editable cells from the dataset table and repopulates `DS.inputs` and `DS.outputs`.

### `addDSRow() → void`

Adds a blank sample row to the dataset editor.

### `removeDSRow(i) → void`

Removes the dataset row at index `i` from both the editor and `DS`.

---

## Right-Panel Tabs & Inspection Tools

The right panel uses a tab system (`#tabs`) with **nine tabs**: **Détails**, **Test**, **Dataset**, **Formules**, **Options**, **Guide**, **🧠 Quiz**, **✎ Manuel**, **🛠 Outils**. The **Détails** tab is the default view and hosts the node/connection detail popovers described under [Canvas interaction](#ui-utility-functions); the **Manuel** tab is a full pedagogical quiz mode covered in its own section below.

### Test tab

#### `runTest() → void`
Runs a single inference on the values entered in the test input fields (`tIn0`, `tIn1`, …), then calls `showTestSteps()` to build the step-by-step explanation.

#### `showTestSteps(inputs, y_target) → void`
Generates the full pedagogical breakdown HTML for a single inference:
- **Phase 1 – Forward Pass**: for each neuron, shows the weighted sum computation, bias addition, activation formula, and f'(z) for backprop.
- **Phase 2 – Loss**: computes and displays the loss term per output, with the formula and numerical substitution.
- **Phase 3 – Backpropagation**: shows the chain rule, output-layer deltas, and hidden-layer delta propagation.
- **Phase 4 – Weight Update**: shows the gradient `∂L/∂w = δ·a` and new weight values under the selected optimiser.

All four phases are rendered as collapsible sections (`togglePhase(id)`).

#### `runAllTest() → void`
Tests every sample in the dataset, displays predictions vs. targets, and reports overall accuracy.

#### `togglePhase(id) → void`
Shows or hides a collapsible phase section in the test steps view.

### Formules tab (Formula Library)

#### `buildFLib() → void`
Initialises the formula library UI: extracts unique tags from `FLIB` (the formula data array), renders tag filter pills, and calls `renderFTree`.

#### `setFTag(tag, el) → void`
Filters the library to show only entries with the selected tag. Updates active state on the tag pills.

#### `filterF(query) → void`
Text-search filter applied to formula names, equations, descriptions, and tags. Triggers a re-render.

#### `getFilt() → object[]`
Returns the filtered subset of `FLIB` matching the current tag and text search.

#### `renderFTree(list) → void`
Renders the formula tree as nested collapsible categories and subcategories. Each entry is rendered as a card with its equation, description, pros/cons, and a small activation curve graph if applicable.

#### `showFDetail(id) → void`
Expands the detail view for a specific formula from `FLIB` (function name is `showFDetail`, not `showFormula`). Renders mathematical notation via `renderEqAuto`, shows a symbol legend from `getFormulaSymbols`, and draws an activation curve in a mini canvas via `drawFGraph`.

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
Returns up to 6 relevant symbol definitions for a formula entry based on its tags. Used to populate the legend block under each formula's detail view.

### Outils (Tools) tab

#### `analyzeGradients() → void` (v12 §12)
Runs a full forward + backward pass on every dataset sample, averages the absolute delta and weight-gradient values per layer, and renders a visual report with:
- Min/avg/max of `|δ|` (error signal) per layer.
- Min/avg/max of `|∂L/∂w|` (actual weight gradient) per layer.
- Log-scale bar charts for each metric.
- Automatic detection and labelling of **vanishing gradients** (max < 1e-4) or **exploding gradients** (max > 50).
- A global summary ratio `max_δ / min_δ` to detect inter-layer imbalance.

#### `inspectLayerStats() → void` (v12 §14)
Reads the selected layer index from `inspectLayer` and displays a statistics card with:
- Number of neurons and total weights.
- Min/max/mean/std of all weights.
- Min/max of biases.
- Count of dead neurons (activation < 1e-6 on last pass).

#### `populateLayerInspect() → void`
Repopulates the `<select id="inspectLayer">` dropdown with one option per weight layer, after building a new network.

#### `drawZoomedLoss() → void` (v12 §13)
Renders a zoomed loss chart in `#zoomedLossChart` showing the last `n` epochs (controlled by `lossZoom` slider). Draws a filled gradient area, the raw loss curve in red, and a 10-point moving average in yellow dashed.

#### `runBenchmark() → void` (v12 §15)
Measures training throughput by running 500 consecutive forward + backward passes and reporting samples/second, total time, and milliseconds per sample.

The Outils tab also hosts the UI entry points for several other features documented in their own sections below: the **📈 LR Scheduler** controls, the **🏁 Competition Mode** (optimiser race) launcher, the **✨ Effects & Sound** toggles (Matrix rain, sound, data flow, confetti), **🧪 Data Perturbation**, **📷 Export image** (PNG / clipboard), **🗓 Weight Heatmap**, and — as full standalone modal workspaces — the **CNN Lab** and **RNN Lab** launch buttons (`openCNNLab()`, `openRNNLab()`; see their dedicated sections below).

### Dataset tab

A dedicated tab (separate from the left-panel dataset preview) for full dataset editing and I/O:

- Rebuilds/refreshes the row-by-row dataset editor via `rebuildDSEditor()`.
- `applyDS()` — validates and commits the editor's rows back into the live `DS` array, then rebuilds the network's input/output layer sizes if needed.
- `renderDSPreview()` — renders a compact read-only preview table of the current dataset.
- `exportDSJSON() → void` — downloads the current dataset as a `.json` file.
- `importDSJSON(input) → void` — reads a JSON file from a file input and replaces the current dataset.

### Guide tab

A static, in-app reference tab (no computation) with written explanations of the interface, workflow tips, and a summary of keyboard shortcuts — a quick-start help page for new users.

### Options tab (formerly "Settings")

Beyond the visual customisation controls (neuron radius, font size, line width, colours, grid mode, curve/arrow toggles — all applied immediately via `redraw()`), the Options tab is organised into sections:

- **Langue** — the FR → EN auto-translation toggle (see [Automatic FR → EN Translation Layer](#automatic-fr--en-translation-layer)).
- **Thème & Couleurs** — colour theme presets, applied via `applyTheme(name)` (see [v12 Extension Features](#v12-extension-features)).
- **Géométrie** — canvas geometry controls (neuron radius, spacing, line width, curve style).
- **Effets visuels** — toggles for animation/visual effects on the canvas.
- **Performance & Accélération matérielle** — GPU detection, canvas rendering mode (`cCanvasRender`: Auto / Force GPU / Force CPU, stored in `nl_canvas_render`), reduced effects during fast training, WebGPU training toggle (`cGPUTrain`) and a "Vérifier GPU vs CPU" button.
- **Techniques d'entraînement** — Batch Normalization (`cBN`, `cBNMom`), data augmentation (`cAug`, `cAugNoise`), early stopping (`cES`, `cESPat`, `cESDelta`).
- **Algorithme avancé** — the regularisation/optimiser hyper-parameters described in [Optimiser Engine](#optimiser-engine): weight-init method, momentum, Adam β₁/β₂/ε, L2 coefficient, dropout rate, Huber delta, gradient-clip threshold, and per-100-epoch LR decay factor.
- **Sauvegarde/Export** — shortcuts into the JSON dataset import/export, network save/load, and code-export features.

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

---

## v12 Extension Features

All features below are defined in the second `<script>` block appended at the end of the document.

### 1. Net Stats Bar — `updateNetStatsBar() → void`
Refreshes the `#net-stats-bar` strip at the top of the canvas area, displaying architecture, total parameter count, epoch, loss, accuracy, LR, and a keyboard shortcut hint.

### 2. Decision Boundary — `renderBoundary() → void`
Renders a 2-D classification boundary in a modal canvas (`#boundary-cvs`). Sweeps a grid of `res×res` pixels over the input space `[−0.2, 1.2]²`, runs a forward pass for each point, and colours pixels cyan (class 0) or purple (class 1) proportionally to the output confidence. Overlays data points as coloured dots.

- `showBoundary() → void` — opens the boundary modal and calls `renderBoundary`.
- `startBoundaryLive() / stopBoundaryLive() → void` — starts/stops a 350 ms refresh interval for live updates during training.

### 3. Network Mutation — `mutateNet(evalAfter?) → void`
Randomly perturbs 40% of weights and 30% of biases by ±`str` (from `mutStr` input). If `evalAfter` is true, evaluates the new loss; if the loss has worsened by more than 50%, the mutation is reverted using a backup. Plays a visual animation on the canvas.

### 4. Save / Load — localStorage-backed model persistence
A fixed number of named slots (`NUM_SLOTS`) stored in `localStorage` under a save key. Opened from the Outils tab via `showSaveLoad()`, which shows the **Save/Load modal** and calls `renderSaveSlots()` to list each slot's architecture summary, timestamp, and last loss (or an empty-slot placeholder with a 💾 save button).

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
- `abldRemoveLayer(i)` — removes a layer row by index.
- `abldPreset(name)` — applies a named preset configuration.
- `drawAbldPreview()` — redraws the architecture preview canvas.

---

## CNN Lab (Convolutional Networks)

Opened from the Outils tab via `openCNNLab()`, the **CNN Lab** is a self-contained modal workspace (its own state object `CNN`, independent of the main MLP `NET`) dedicated to convolutional-network concepts. It has 5 sub-tabs, switched with `cnnShowTab(name)`.

### Conv — convolution playground
An interactive single-kernel convolution demo on a drawable `n×n` grid:
- `cnnSetGridSize(n)` / `cnnRenderInputGrid()` / `cnnToggleCell(y, x)` — resize and draw a black/white pixel grid by clicking cells.
- `cnnPreset(name)` — loads a built-in kernel preset from `CNN_KERNELS`: **Identity, Blur, Sharpen, Sobel X, Sobel Y** (each with a short French explanation of what the kernel detects).
- `cnnKernelPresetChange()` / `cnnRenderKernelGrid()` / `cnnKernelEdit(y, x, val)` — display and hand-edit the 3×3 kernel values.
- `cnnPadInput(input, kH, kW, pad)` — zero-pads the input matrix.
- `cnnConvRaw(padded, kernel, stride)` / `cnnConv2d(input, kernel, stride, pad)` — the raw convolution math (sliding-window dot product).
- `cnnRelu(m)` — applies ReLU element-wise to a matrix.
- `cnnMaxPool2(m)` / `cnnAvgPool2(m)` — 2×2 max/average pooling.
- `cnnCompute()` — runs the full pipeline (convolution → optional ReLU → optional pooling) and renders each stage matrix via `cnnRenderMatrix(m, cellSize)` with a colour legend (`cnnColorLegend`).
- `cnnRenderProgressiveMatrix(m, useRelu, cell)`, `cnnToggleAnim()` / `cnnStopAnim()` — an animated "sliding kernel window" visualisation that steps the convolution across the input one position at a time.

### Gallery — kernel gallery
`cnnGalleryRender()` renders every preset kernel from `CNN_KERNELS` applied to a sample image side-by-side, so the effect of each classic kernel (blur, sharpen, edge detectors, …) can be compared at a glance.

### Pipe — multi-layer pipeline
Lets the user chain several conv/pool/ReLU layers and see the shrinking feature-map sizes end-to-end:
- `cnnPipeInitDefault()`, `cnnPipeAddLayer()`, `cnnPipeRemoveLayer(i)`, `cnnPipeUpdateLayer(i, field, val)` — build/edit an ordered list of pipeline layers (kernel choice, pooling, stride).
- `cnnPipeRender()` — draws the layer chain.
- `cnnPipeCompute()` — executes the full pipeline on the current input grid, showing every intermediate feature map.

### Train — kernel training (gradient descent on a single kernel)
A minimal supervised-learning demo where a 3×3 kernel is learned from scratch:
- `cnnMSE(pred, target)` — mean-squared error between the conv output and a target feature map.
- `cnnTrainGrad(padded, pred, target)` — computes the gradient of the MSE loss with respect to every kernel weight (correlation between the input and the output error) and updates the kernel.
- `cnnLossSparkline(hist, w, h, color)` — draws a small loss-history sparkline.
- `cnnTrainSetup()`, `cnnTrainStep()`, `cnnTrainRender()`, `cnnTrainToggle()` (auto-run), `cnnTrainStop()`, `cnnTrainReset()` — the training-loop controls, mirroring the main app's step/auto/reset pattern but scoped to a single kernel.

### Model — end-to-end image classifier
A genuine (small) trainable CNN, independent of the main `NET`/`applyGrad` engine, for classifying hand-drawn grid images into user-defined classes:
- **Dataset sub-tab**: `cnnModelRenderClasses()`, `cnnModelAddClass()`, `cnnModelRemoveClass(i)` manage class labels with colour swatches (`cnnModelClassColor(i)`); `cnnModelSetSize(n)`/`cnnModelResizeGrid()` control image resolution; `cnnModelRenderDraftGrid()`/`cnnModelToggleDraftCell(y,x)`/`cnnModelDraftPreset(name)` draw a sample by hand (or import the current Conv-tab grid via `cnnModelImportFromConv()`); `cnnModelAugment(grid)` generates augmented variants (flips/shifts/noise) when adding a sample with `cnnModelAddSample(withVariations)`; `cnnModelDeleteSample(idx)`/`cnnModelClearDataset()` manage the sample list; `cnnModelRenderDsStats()`/`cnnModelRenderDatasetGrid()` show dataset statistics and thumbnails; `cnnModelGenerateDemo()` auto-generates a demo dataset; `cnnModelExportJSON()`/`cnnModelImportJSON()` save/load the dataset.
- **Model sub-tab**: `cnnModelBuild()` constructs the network from configurable conv blocks (`cnnModelShape`, `cnnModelConfigChanged`) — each block is a conv layer + activation + optional max-pool, computed via `cnnModelBlockForward`/`cnnModelMaxPool2WithArgmax` — followed by a flatten and dense classification head; `cnnModelResetWeights()` reinitialises; `cnnModelRenderArch()` draws an architecture diagram.
- **Forward/backward engine**: `cnnModelForward(grid)` runs the image through every conv block and the dense head to produce class probabilities; `cnnModelBackward(cache, labelIdx)` performs full backpropagation through the dense layer, unpooling (using stored max-pool argmax indices), and every convolution to compute gradients for every kernel/weight; `cnnModelApplyGrad(grads, lr)` applies the SGD update.
- **Run sub-tab**: `cnnModelEpoch()` / `cnnModelEpochs(n)` run one or several training epochs over the dataset; `cnnModelTrainToggle()` (auto-run) / `cnnModelTrainStop()` control continuous training; `cnnModelRenderTrain()` plots the training loss/accuracy curve.
- **Test sub-tab**: `cnnModelRenderTestGrid()`/`cnnModelToggleTestCell(y,x)`/`cnnModelTestPreset(name)`/`cnnModelTestImportFromConv()` let the user draw or import a test image; `cnnModelPredict()` runs inference and shows the predicted class with per-class confidence; `cnnModelTestAll()` evaluates the whole dataset and reports overall accuracy.

---

## RNN Lab (Recurrent Networks)

Opened from the Outils tab via `openRNNLab()`, the **RNN Lab** is a second self-contained modal workspace dedicated to recurrent networks, with 4 sub-tabs (function names use the `rnn` prefix).

### Cellule — single recurrent-cell step
Visualises one RNN cell's computation, `h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)`, for a single time step: input, previous hidden state, weight matrices, and the resulting new hidden state are all shown numerically and graphically, so the user can see exactly how memory (`h_{t-1}`) blends with new input at each step.

### Déroulement — sequence unrolling
Feeds a full input sequence through the same recurrent cell step-by-step ("unrolling" the recurrence in time), rendering each time step's hidden state side-by-side so the propagation of information across the sequence is visible.

### Mémoire & gradients — vanishing/exploding gradients (BPTT)
A dedicated demonstrator for the central RNN training challenge: it runs backpropagation-through-time (BPTT) over a configurable sequence length and plots how the gradient magnitude shrinks (vanishes) or grows (explodes) as it is propagated backward through many time steps — with an explanation of how gating mechanisms (LSTM/GRU) mitigate the problem.

### Entraîner — trainable character-level RNN
A genuine mini character-level recurrent language model, trained end-to-end with BPTT, with its own **Dataset**, **Modèle**, **Run**, and **Test** sub-tabs (mirroring the CNN Lab's Model tab structure): the user supplies or generates a short text corpus, configures the hidden-state size and sequence length, trains the RNN to predict the next character, and can then sample generated text from the trained model or test it on custom prefixes.

---

## Code & Data Export

Beyond PNG/clipboard/LaTeX/Markdown log export and the save-slot JSON export already covered above, two further export paths exist:

### `exportCode(lang) → void`
Generates a complete, runnable standalone implementation of the **current network** (architecture, weights, biases, activation functions) in either **Python** (NumPy) or **JavaScript**, including a forward-pass function, so the trained model can be used outside the browser. The generated source is shown in a code-preview modal with a copy-to-clipboard button.

### Dataset JSON import/export
`exportDSJSON()` downloads the current dataset (`DS.inputs`/`DS.outputs`) as a `.json` file; `importDSJSON(input)` reads a JSON file back in and rebuilds the dataset editor (`rebuildDSEditor()`) and preview (`renderDSPreview()`), calling `applyDS()` to commit it as the live dataset.

---

## Automatic FR → EN Translation Layer

A self-contained script block (the 4th `<script>` tag, appended at the very end of the document) adds a one-click interface translator, toggled from the **Options → Langue** section via `toggleTranslation()`.

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
Appends a **collapsible structured block** to the log with a coloured header bar (forward=cyan, loss=red, backprop=orange, update=green). Used during verbose training.

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
| `C` | `openCompare()` — comparison mode |
| `T` | `showTab('tools')` — switch to Tools tab |
| `G` | `analyzeGradients()` — gradient analysis |
| `H` | `renderHeatmap()` — weight heatmap |
| `Escape` | Close all modals (boundary, save/load, race, compare, CNN, RNN, RL, Transformer, visual dataset editor) and stop live animations |
| `←` / `→` / `Space` | While step-by-step mode is open: previous / next step, play/pause |

Shortcuts are disabled when focus is inside `<input>`, `<textarea>`, or `<select>` elements.

---

## Mobile Support

On viewports ≤ 900 px the layout switches entirely via CSS media queries:

- The three panels stack and are shown one at a time, each taking the full viewport.
- A **top bar** (`#mobile-topbar`) replaces the left panel header, showing the app title plus a live epoch/loss pill.
- A **quick-train bar** (`#mobile-trainbar`) below the top bar provides one-tap access to Step, Epoch, Auto, and ×100 buttons.
- A **bottom navigation bar** (`#mobile-nav`) switches between Config, Network, and Details panels.
- Resize handles are hidden on mobile.
- Modals slide up from the bottom as sheets.
- The log panel has a fixed 160 px height.

#### `mobNav(panel) → void`
Activates the named panel (`'left'`, `'center'`, or `'right'`) by toggling the `mob-active` class. On switching to the canvas panel, a 60 ms timeout ensures `resizeCvs()` fires after the layout is applied.

#### `syncMobStats() → void`
Called every 400 ms via `setInterval`. Copies the epoch and loss text content from the left panel stats display to the top bar pill elements.

---

### Local Dev

Tip to have **hotreload** (*nodejs* must be installed)

```bash
npx live-server .
```

---

## v13 Additions

### WebGPU Training Engine

Optional engine (Options → *Performance & Accélération matérielle*) that runs forward pass, loss gradient, backprop, gradient clipping, L2 and the optimiser as WGSL compute shaders (`GPU_WGSL`). Browsers cannot call CUDA/cuDNN; WebGPU is vendor-neutral and still uses the machine's GPU.

| Function | Role |
|---|---|
| `initWebGPU()` / `buildGPUPipelines()` | Adapter/device setup and pipeline creation |
| `gpuAllocateBuffers()` / `gpuBuildLayerSet()` | Per-architecture buffer bundle (`GPU.main`, keyed by `netSig = NET.L.join(",")`) |
| `gpuSyncWeightsToDevice()` / `gpuReadWeightsFromDevice()` / `gpuCommitResults()` | CPU ⇄ GPU weight synchronisation |
| `gpuTrainOneSample(x,y)` / `gpuTrainEpochs(n, checkEarlyStop)` | Single sample / batched epochs (no per-step readback) |
| `cpuReferenceStep()` / `verifyGPUvsCPU()` | Numerical check against the CPU reference before the GPU path is trusted |
| `setGPUTrain(on)` / `manualVerifyGPU()` | UI toggle (`nl_gpu_train` in `localStorage`) and manual verification |
| `gpuOnNetChanged()` | Invalidates buffers when the network changes |
| `gpuEligible()` | True only if GPU is ready/verified, the layer signature matches, a dataset exists, and **no** Batch Norm, dropout, node patterns, frozen weights or frozen nodes are active |

Coverage: 10 activations + softmax (`ACT_CODE`), 5 losses (`LOSS_CODE`), 6 optimisers (`OPT_CODE`). Otherwise it silently falls back to the CPU. In GPU mode the detailed per-step logs are not generated, and the **Manuel** tab always stays on the CPU. On small networks the CPU can be faster.

### Training Techniques

- **Batch Normalization** — `bnOn()`, `bnMomentum()`, `ensureBN()`; γ/β are learned (`BN_GRAD`), running mean/variance are updated during training, hidden layers only.
- **Data augmentation** — `augOn()`, `augNoise()`; Gaussian input noise during `trainSample`.
- **Early stopping** — `earlyOn()`, `earlyPatience()` (default 30), `earlyMinDelta()`, `earlyStopUpdate()`, `stopAutoWithMessage()`.

### Visual Dataset Editor

Opened with `openVDS()` / `closeVDS()` from the Dataset tab; two tabs (`vdsShowTab('class'|'reg')`). The active tab drives `DS`.

- **Classification** (2 inputs → N one-hot outputs): `vdsClassAddClass/RemoveClass`, left-click adds a point of the selected class, right-click removes the nearest (`vdsClassRemoveNearest`), live coloured decision-boundary background (`vdsRenderClass`, refreshed every 400 ms via `vdsStartLive`). Demo sets via `vdsClassGenerate`: `circles`, `spiral`, `checker`, `blobs`.
- **Regression** (1→1): click to add points on a curve, `vdsRegGenerate`: `linear`, `quad`, `sine`; the network's current prediction curve is drawn live.
- `vdsArchMismatchMsg()` / `vdsApplyArch()` detect an input/output size mismatch with the network and rebuild the architecture in one click.

### Comparison Mode

`openCompare()` (shortcut `C`) trains **two independent networks** with their own layers, activations, optimiser and learning rate on the current dataset and loss. Engine: `cmpReadCfg`, `cmpBuildNet`, `cmpForward/Backward/Update`, `cmpAccuracy`, `cmpRenderBoundaries`, `drawCompareChart`, `startCompare/stopCompare/resetCompare`, `cmpCopyMain` (load main net into a slot), `cmpApplyToMain` (adopt a trained net). Input/output sizes are forced to match the dataset.

### Matrix Calculation Mode

A switch (`calcSwitchHTML()`, `setCalcMode('indiv'|'matrix')`) rendered in every `.calc-sw-slot` (log, Test, Manuel, Formules, details, labs) chooses between per-neuron and matrix/vector notation. `calcRerenderAll()` re-renders existing views without changing tab. Matrix helpers (`cm*`): `cmFwdAll`, `cmLossHTML`, `cmBackAll`, `cmUpdAll`, `cmBatchHTML`, `cmManFwdFields/DeltaFields/UpdFields` and result grids, `cmFormulaMatrixHTML`, `cmExportBody` (NumPy-style code), and mini-lab variants `cmCnnCalcHTML`, `cmRnnCellHTML`, `cmTfDetHTML`.

### Step-by-Step Explanation Mode

Button **🎓 Pas à pas** (next to *Step*). `build()` computes a full pass on a chosen sample without touching the network; `makeSteps()` produces phases **Entrée → Somme z → Activation a → Erreur → Gradient → Mise à jour**, each with a KaTeX formula and numeric substitution. The canvas highlights the neurons/links involved and reveals values progressively. Controls: first/prev/play/next, slider, speed, sample selector, fold. The update phase really changes the weights: closing before the last step restores the snapshot. Normal training (`doStep/doEpoch/doN/toggleAuto`) closes the mode cleanly.

### Quiz

`QUIZ_BANK` holds **126 questions** — 42 `qcm`, 42 `tf` (true/false), 42 `open` (accepted answers compared after `quizNorm`: lower-case, accents and punctuation stripped). Chapters in `QUIZ_CATS`: Bases (18), Forward (18), Loss & Backprop (18), Optimisation (24), CNN (24), RNN (24); `rl` and `transformer` chapters are declared but have no questions yet. Features: chapter/type filters, random N-question mode (`quizStartRandom`), per-question explanation, end screen with `quizRestartWrong()`.

### Transformer Lab

`openTFLab()`, tabs via `tfShow`: **Théorie** (six-step pipeline: tokens, embeddings, attention, Add & Norm, FFN, output), **Attention** (`tfAttn`: Q/K/V, scores, softmax heat map, per-head detail), **Positions** (`tfPos`: sinusoidal encoding, RoPE), **Décodage** (`tfDec`: temperature, top-k, top-p, entropy), **Taille & coût** (`tfSize`: parameters, FLOPs/token, weight memory, KV-cache with GQA; presets such as GPT-2 small, Llama-3 8B/70B). Educational and deterministic (seeded matrices); not a trainable model.

### RL Lab

`openRLLab()`, tabs via `rlShowTab`:

1. **Concepts** — return, discount γ.
2. **Bandits** — Bernoulli or Gaussian arms; greedy, ε-greedy, UCB1, softmax, Thompson and a manual "Vous" player compared live.
3. **Calculateur** — return Gₜ, TD(0), Q-learning vs SARSA vs Expected SARSA, GAE, all step by step.
4. **Q-Learning / SARSA** — editable grid world (`rlgPaint`, presets, resize, stochastic parameters), ε-greedy, greedy path, comparison with DP.
5. **Bellman (DP)** — value iteration / policy improvement (`rldSweep`, `rldConverge`, `rldImprove`).
6. **Policy Gradient** — REINFORCE and Actor-Critic with baseline.
7. **DQN (CartPole)** — physical CartPole (`rlcpStep`), hand-written dense ReLU network with Adam (`rlNetForward/Backward/Adam`), replay buffer, target network, ε decay (`rlDqnNew/Act/Train`).

### Visual Layer & Performance

- **Aurora pack**, compaction, merged bars, collapsible left-panel sections and journal, mobile readability (type scale, touch targets, bottom-sheet labs).
- **Satisfaction packs**: ripple, magnetic spotlight, particle sparks, animated counters with accuracy milestones, modal cascade, slider value bubble; all disabled under `prefers-reduced-motion`.
- **Sliding tab indicator** for every tab group.
- Canvas context attributes follow the chosen render mode (`canvasCtxAttrs()`); a `ResizeObserver` keeps the canvas in sync with its container.
- Formula library gains **Bases**, **RL** and **Transformer** categories, a general symbol legend and matrix/individual forms.
- CNN Lab: kernel presets now include **Laplacian** and **Emboss**; the Model tab supports image sizes up to 768×768 (pixel grid ≤ 32×32, free-drawing canvas editor above).

---

## Colour Palette Reference

All colours are defined as CSS custom properties on `:root`:

| Variable | Value | Usage |
|---|---|---|
| `--bg` | `#05060c` (Aurora overrides to `#04050b`) | Main background |
| `--p1` | `#0a0c18` | Panel background |
| `--p2` | `#0d1020` | Section/header background |
| `--brd` | `#192040` | Border colour |
| `--a1` | `#00e5ff` | Primary accent (cyan) |
| `--a2` | `#bf5fff` | Secondary accent (purple) |
| `--grn` | `#00ff9d` | Success / positive |
| `--ylw` | `#ffd600` | Warning / highlight |
| `--red` | `#ff3d5a` | Error / loss |
| `--org` | `#ff8c2a` | Backprop / alert |
| `--text` | `#b8cce8` | Primary text |
| `--dim` | `#445577` | Secondary / muted text |

This is the default theme; the Options tab's "Thème & Couleurs" section offers several alternate colour presets (applied via `applyTheme(name)`) which override these CSS variables at runtime.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| KaTeX | 0.16.9 | Mathematical formula rendering in the formula library |
| JetBrains Mono | Google Fonts | Monospace UI font |
| Syne | Google Fonts | Display / heading font |

All other functionality is implemented in vanilla JavaScript with no bundled runtime dependencies. The only exception is the optional **auto-translation layer**, which makes live network calls to a public translation API when enabled by the user (see [Automatic FR → EN Translation Layer](#automatic-fr--en-translation-layer)) — the app is otherwise fully offline-capable.
