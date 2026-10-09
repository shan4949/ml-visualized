# ML Visualized: Machine Learning from Scratch, Step by Step

A series of Google Colab notebooks that build machine learning up from first principles, with an animation, an interactive slider or a from-scratch implementation at nearly every step. The spirit is Andrej Karpathy's *Zero to Hero*: nothing is a black box. You derive it, build it, watch it work, then check it against the library version.

The series covers the full content of two Georgia Tech courses, **CX 4240** (Computing for Data Analysis) and **ISYE 4600** (Methods & Applications of ML). It then continues through deep learning to transformers and LLM post-training.

Every notebook runs on the **free Colab tier**. Click a badge and choose `Runtime → Run all`.

## How each notebook works

- **Intuition first:** a small example you could work on paper.
- **Math, derived step by step:** each line follows from the one before.
- **Built from scratch** in NumPy, one small function at a time.
- 🎬 **Animations** of the algorithm running (press ▶).
- 🎚️ **Sliders** to change parameters and see the effect live.
- 🤔 **Questions** people commonly ask at that point, answered in place.
- ✏️ **Exercises** and references at the end.

## Notebooks

### Part 0 — Foundations 🚧 *(in progress)*

| # | Notebook | What you build / see | Open |
|---|---|---|---|
| 00 | **Roadmap** | model → loss → optimizer on apartment rents; you vs. gradient descent; overfitting; map of the series | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/00_roadmap.ipynb) |
| 01 | **Linear algebra** | matrices as transformations of space, eigenvectors, PSD bowls vs. saddles, the SVD animated, image compression, the normal equation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/01_linear_algebra.ipynb) 🚧 |
| 02 | **Probability & statistics** | law of large numbers, CLT, Bayes' rule, covariance ellipses, MLE, MAP, and how every loss function comes from a likelihood | 🔜 coming soon |
| 03 | **Optimization** | random search → gradient descent → adaptive step sizes → SGD → Momentum/RMSProp/Adam racing on Rosenbrock; conditioning, convexity, KKT and duality | 🔜 coming soon |
| 04 | **Gradients & backprop from zero** | a micrograd-style autograd engine, computation graphs, the CX 4240 lecture network by hand, an MLP trained on moons, gradient checking, vectorized backprop vs. PyTorch | 🔜 coming soon |

### Part 1 — Classical supervised learning *(coming soon)*

05 Linear regression · 06 Bias–variance & cross-validation · 07 Regularization & feature selection · 08 Logistic regression · 09 Naive Bayes & generative classifiers · 10 KNN & evaluation metrics · 11 Support vector machines · 12 Decision trees & random forests · 13 Boosting

### Part 2 — Classical unsupervised learning *(coming soon)*

14 Clustering · 15 Gaussian mixtures & EM · 16 PCA · 17 Recommender systems · 18 Anomaly detection

### Part 3 — Deep learning *(coming soon)*

19 Neural networks · 20 PyTorch essentials · 21 CNNs · 22 RNNs & LSTMs

### Part 4 — Deep generative & representation learning *(coming soon)*

23 Variational autoencoders · 24 Contrastive learning (SimCLR)

### Part 5 — Transformers & LLMs *(coming soon)*

25 Attention & a mini-GPT from scratch · 26 LLM post-training (SFT, RLHF, DPO, GRPO)

## Running locally

```bash
git clone https://github.com/shan4949/ml-visualized.git
cd ml-visualized
pip install -r requirements.txt     # plus the Graphviz system package for notebook 04
jupyter lab notebooks/
```

## Repository layout

```
notebooks/   the notebooks (saved with outputs so they preview on GitHub)
src/         notebook sources (*.nb.txt) + build.py / finalize.py that compile and execute them
```

Notebooks are generated from `src/`: `python src/build.py` compiles the sources and `python src/finalize.py` executes them. Sliders only appear when the notebook runs with a live kernel (e.g. in Colab).

## References and credits

- CX 4240, Computing for Data Analysis (Bo Dai), Georgia Tech, Spring 2026
- ISYE 4600, Methods & Applications of ML (Mohsen Moghaddam), Georgia Tech, Spring 2026
- Andrej Karpathy, *Neural Networks: Zero to Hero* and `micrograd`
- 3Blue1Brown, *Essence of Linear Algebra*; Stanford CS231n notes
- C. Bishop, *Pattern Recognition and Machine Learning*; Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*; Goodfellow, Bengio & Courville, *Deep Learning*

Datasets are public: scikit-learn's built-in samples and generators; later notebooks add MNIST and others, downloaded automatically.
