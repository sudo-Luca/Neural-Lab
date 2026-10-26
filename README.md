# 🧠 Neural Lab v13

> **A visual playground to understand neural networks — built from scratch in a single HTML file.**

<div align="center">

![Neural Lab Banner](screenshots/banner.png)

**Train. Visualise. Experiment. Understand.**

[Features](#-features) • [Screenshots](#-screenshots) • [Quick Start](#-quick-start) • [How it works](#-how-it-works) • [Try it online](https://sudo-luca.github.io/Neural-Lab/Neural-Lab-v13.html)

</div>

---

## ✨ What is Neural Lab?

Neural Lab is an interactive deep learning simulator designed to make neural networks understandable.

No framework.  
No installation.  
No server.

Just open **one HTML file** and start experimenting.

Every computation happens directly in your browser:

- Forward propagation
- Backpropagation
- Gradient calculation
- Weight updates
- Optimiser behaviour
- Decision boundaries
- Network visualisation

Built with **pure HTML + CSS + Vanilla JavaScript** (optionally accelerated with **WebGPU**).

Beyond multilayer perceptrons, v13 also ships four self-contained labs — **CNN, RNN, Transformer and Reinforcement Learning** — plus a **140-question quiz**, a **step-by-step explainer** and a switch between **neuron-by-neuron** and **matrix** calculations everywhere the maths is shown.

> 🌐 **Language note:** the interface is in French by default, with a built-in FR ⇄ EN translation switch (in the Options tab) if you prefer English.

---

## 🎥 See it in action

Watch a neural network learn in real time:
- neurons activating
- weights changing
- loss decreasing
- decision boundaries evolving

---

# 🚀 Features

## 🕸️ Visual Neural Network Playground

Create and observe multilayer perceptrons directly on the canvas.

![Network view](screenshots/network-view.png)

Features:

✅ Custom architectures  
✅ Unlimited hidden layers  
✅ Per-layer activation functions  
✅ Live neuron activation display  
✅ Weight visualisation  
✅ Bias inspection  
✅ Interactive neuron and connection editing  
✅ Guided creation wizard  
✅ Advanced layer-by-layer builder (with ready-made presets: XOR, deep net, autoencoder, 3-class classifier, regression, wide net)  
✅ CNN lab  
✅ RNN lab  
✅ Transformer lab  
✅ Reinforcement-learning lab


---

## 🔥 Train Neural Networks Step by Step

Understand what happens inside a model.

![Training controls](screenshots/training.png)

Training modes:

- Single step
- Single epoch
- Multiple epochs
- Automatic training
- Train until target loss
- Early stopping (patience + minimum improvement)
- 🎓 **Step-by-step explainer**: replay one sample as input → z → a → error → gradient → update, with the active neurons and connections highlighted on the canvas

Every step can be inspected:

```
Input
 ↓
Forward Pass
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient Update
```

---

# 🧪 Experiment with Modern Deep Learning

Neural Lab includes many core concepts used in real machine learning.

## Activation Functions

Examples:

- Sigmoid
- ReLU
- Tanh
- Leaky ReLU
- ELU
- Swish
- GELU
- SELU
- Softsign
- Linear
- Softmax (output layer)


## Loss Functions

Available:

- Mean Squared Error (MSE)
- MAE
- Log Loss / Binary Cross Entropy
- Huber Loss
- Hinge Loss


## Optimisers

Compare different learning strategies:

- SGD
- Momentum
- RMSProp
- Adam
- AdamW
- Nesterov (NAG)

Fine-grained control is also available: momentum β, Adam β₁/β₂, epsilon, and L2 regularisation.

## Weight Initialisation

- Xavier / Glorot
- He / Kaiming
- Uniform
- Normal
- Small
- Zero (debug)

## Training Techniques

- Batch Normalisation (learned γ/β, running statistics, hidden layers only)
- Dropout
- L2 regularisation and global gradient clipping
- Gaussian-noise data augmentation
- Early stopping
- Learning-rate scheduler and per-100-epoch LR decay

## GPU Training (WebGPU)

An opt-in **WebGPU** engine runs forward pass, backpropagation and the optimiser update on your GPU. It is numerically checked against the CPU path before use and falls back to the CPU automatically whenever a feature it does not replicate (Batch Norm, Dropout, forced neurons, frozen weights) is active. Small networks are often just as fast on the CPU, so compare both.


---

# 📊 Visual Debugging Tools

Neural Lab is not just a trainer.

It is an exploration environment.

![Tools panel](screenshots/tools.png)


Included tools:

### Gradient Analysis

Find:

- Vanishing gradients
- Exploding gradients
- Layer imbalance


### Weight Heatmap

See how your network parameters evolve.

![Heatmap](screenshots/heatmap.png)


### Decision Boundary

Visualise what your classifier has learned.

![Decision boundary](screenshots/boundary.png)


### Optimiser Race

Run several optimisers against each other and watch who learns faster.

![Optimizer race](screenshots/race.png)


### Compare two networks

Configure two networks (architecture, activations, optimiser, learning rate), train them in parallel on the same dataset, and watch the loss curves and both decision boundaries side by side. Send the winner back to the main editor in one click.

### More tools

- Interactive loss curve (click to zoom / clear)
- Learning-rate scheduler
- Layer-by-layer inspection
- Speed benchmark
- Data perturbation testing
- Network diagram export (PNG)


---

# 🎨 Designed for Learning

Neural Lab includes educational features:

- Formula library (~110 formulas in 12 categories, including a **Bases** category and every RL / Transformer formula), each with a symbol legend
- **Individual ⇄ matrix calculation switch**: every calculation view (Test, Details, Formulas, Manual, detailed logs, code export) can show the maths neuron by neuron or as real matrix products, with a matrix-notation box (m, L, n_l, shapes of X, W, b, Z, A, δ…)
- Step-by-step explanations
- Mathematical notation rendering
- Detailed training logs
- Interactive inspection
- **Manual calculation mode**: compute the forward pass, the delta (error gradient) and the weight updates yourself, then check your answers against the simulator
- **🧠 Quiz**: 140 questions (multiple choice, true/false, open answer) across 8 chapters, with explanations, random mode, personal records and a "review my mistakes" replay
- Keyboard shortcuts for common actions (step, epoch, auto-train, mutate...)
- Training log export to LaTeX or Markdown


![Formula library](screenshots/formulas.png)


Instead of only seeing:

```
Loss = 0.034
```

you can explore:

```
How was the prediction computed?
Why did the gradient change?
Which weights were updated?
```

---

# 🧬 Go Further: Labs & Evolution

All labs open from the **Outils** tab. The CNN, RNN and Transformer labs also follow the individual / matrix calculation switch.

## CNN Lab

Step-by-step convolution with 7 classic kernels (+ custom), a gallery comparing every filter, a multi-layer pipeline with shrinking feature maps, kernel training via gradient descent, and a full "complete model" mode to build, train (real backpropagation) and test a multi-layer CNN on an image dataset you draw yourself — from 8×8 pixels up to 1536×1536 with a real bitmap editor (brush, pencil, eraser, line, rectangle, ellipse, fill, undo).

## RNN Lab

Four tabs: a single recurrent cell, sequence unrolling, vanishing / exploding gradients, and a trainable character-level RNN (real BPTT, Adagrad) with its own dataset, model, training and test screens — including text generation.

## Transformer Lab

Five tabs on the architecture behind GPT, Llama or Claude: theory (tokens → embeddings → attention → output), live multi-head attention on your own sentence, positional encodings (sinusoidal, learned, RoPE, ALiBi), decoding strategies (greedy, temperature, top-k, top-p), and size / cost estimates with presets (GPT-2 small, Llama-3 8B / 70B, a toy model).

## Reinforcement Learning Lab

Seven tabs: concepts, multi-armed bandits (greedy, ε-greedy, UCB1, softmax, Thompson — or play against them), a step-by-step RL calculator, Q-learning / SARSA / Expected SARSA on an editable grid world, Bellman dynamic programming (value iteration and policy evaluation), policy gradient (REINFORCE with or without baseline, actor-critic), and a DQN agent balancing a CartPole with a neural network trained from scratch (replay buffer, target network, optional Double DQN).

## Weight Mutation

Randomly mutate a trained network's weights and evaluate the result — a lightweight way to experiment with evolutionary/genetic approaches.

## Visual Dataset Editor

Draw a 2D classification dataset (several classes, one-hot outputs) or a regression point cloud with the mouse, or generate circles / spiral / checkerboard / blobs and linear / quadratic / sine demos, then train the main network on it while the decision regions or learned curve update live.

---

# 💾 Save, Export & Code Generation

- Multiple save/load slots for your trained networks
- Import / export a network as JSON
- Export a trained network as ready-to-use JavaScript code
- Export the full training log as LaTeX or Markdown

---

# 🎨 Customisation

- 6 color themes: Dark, Neon, Ocean, Fire, Matrix, Pastel
- Visual effects & sound (Matrix rain, sound toggle, particle flow, celebration effects, subtle micro-interactions — all disabled when your system asks for reduced motion)
- Adjustable diagram geometry (node spacing, glow intensity, connection opacity...)
- Performance options (hardware-accelerated canvas, lighter rendering during fast training)
- Collapsible left-panel sections and log

---

# ⚡ Quick Start

## Option 1 — Open directly

Download the project:

```bash
git clone https://github.com/sudo-Luca/Neural-Lab.git
```

Open:

```
Neural-Lab/Neural-Lab-v13.html
```

That's it.

No dependencies.
No build step.

---

## Option 2 — Try online

Any modern browser works:

- Chrome
- Firefox
- Edge
- Safari
- Opera
- Opera GX

Click [here](https://sudo-luca.github.io/Neural-Lab/Neural-Lab-v13.html) and enjoy your experience

---

# 🧩 Example Experiments

## XOR Problem

A classic neural network challenge.

```
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

Try:

```
Input: 2
Hidden: 4
Output: 1
Activation: ReLU
Optimizer: Adam
```

---

## Spiral Classification

A harder nonlinear problem. Generate it in one click from the **Visual Dataset Editor** (Dataset tab).


---

## Regression

Approximate mathematical functions:

```
y = sin(x)
```

Watch the network learn a curve (use the regression tab of the Visual Dataset Editor and its "sine" demo).

---

## Logic gates and small circuits

Built-in presets: XOR, AND, OR, NAND, XNOR, a half adder (2→2) and a 4-bit identity (4→4).

---

## CNN problems

Learn about CNN with the CNN lab in tools tab

---

## Reinforcement learning

Open the RL lab and let Q-learning solve the "cliff" grid, or train a DQN to balance a CartPole.

---

# 📱 Works Everywhere

Desktop:

![Desktop](screenshots/desktop.png)


Mobile:

![Mobile](screenshots/mobile.png)


The interface automatically adapts:

- Responsive layout
- Touch-friendly controls (larger buttons, sliders and checkboxes)
- Mobile navigation
- Full-screen panels
- Readable type scale and labs that open as full-width bottom sheets

The desktop layout is left untouched by the mobile styles.


---

# 🏗️ Architecture

Neural Lab is intentionally simple:

```
Single HTML file

├── HTML
│   └── Interface
│
├── CSS
│   └── Dark UI system
│
└── JavaScript
    ├── Neural network engine (CPU + optional WebGPU)
    ├── Training algorithms
    ├── Canvas renderer
    ├── Calculation views (individual / matrix)
    ├── Quiz & step-by-step explainer
    └── Labs (CNN, RNN, Transformer, RL)
```

No frameworks.

No hidden magic.

Everything is visible.

---

# 📚 Technical Documentation

For a complete technical description of:

- internal architecture
- neural computation
- backpropagation implementation
- optimiser formulas
- rendering engine
- dataset system
- calculation modes, quiz and step-by-step explainer
- WebGPU training
- CNN, RNN, Transformer and RL labs
- v12 extensions

See:

➡️ [`Technical-Description.md`](Technical-Description.md)

---

# 🎯 Project Goals

Neural Lab was created with one idea:

> **Understanding neural networks should not require treating them as a black box.**

The goal is to make deep learning concepts:

- visual
- interactive
- experimental
- approachable


---

# 🛠️ Technologies

Built with:

| Technology | Usage |
|-|-|
| HTML5 Canvas | Neural visualisation |
| Vanilla JavaScript | Neural engine |
| CSS Variables | UI system |
| KaTeX | Mathematical rendering |
| WebGPU (optional) | GPU training |


---

# 🌟 Roadmap

Possible future improvements:

- [x] More architectures (CNN, RNN)
- [x] More datasets (logic gates, visual 2D editor)
- [x] Transformer lab (theory, attention, positions, decoding, cost)
- [x] Reinforcement-learning lab
- [x] WebGPU acceleration (opt-in training on the GPU)
- [ ] Trainable Transformers, GANs...
- [ ] More built-in datasets
- [ ] Collaborative experiments

> ✅ The CNN / RNN / Transformer / RL labs and network export/import are already available — see [Go Further](#-go-further-labs--evolution) and [Save, Export & Code Generation](#-save-export--code-generation) above.


---

# 🤝 Contributing

Ideas, improvements and experiments are welcome.

Feel free to open:

- Issues
- Feature requests
- Pull requests


---

# 📜 License

See repository license information.

---

<div align="center">

**Made for learning. Built for curiosity. 🧠**

</div>
