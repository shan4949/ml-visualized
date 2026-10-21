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

### Part 0 — Foundations ✅

| # | Notebook | What you build / see | Open |
|---|---|---|---|
| 00 | **Roadmap** | model → loss → optimizer on apartment rents; you vs. gradient descent; overfitting; map of the series | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/00_roadmap.ipynb) |
| 01 | **Linear algebra** | matrices as transformations of space, eigenvectors, PSD bowls vs. saddles, the SVD animated, image compression, the normal equation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/01_linear_algebra.ipynb) |
| 02 | **Probability & statistics** | law of large numbers, CLT, Bayes' rule, covariance ellipses, MLE, MAP, and how every loss function comes from a likelihood | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/02_probability_statistics.ipynb) |
| 03 | **Optimization** | random search → gradient descent → adaptive step sizes → SGD → Momentum/RMSProp/Adam racing on Rosenbrock; conditioning, convexity, KKT and duality | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/03_optimization.ipynb) |
| 04 | **Gradients & backprop from zero** | a micrograd-style autograd engine, computation graphs, the CX 4240 lecture network by hand, an MLP trained on moons, gradient checking, vectorized backprop vs. PyTorch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/04_gradients_backprop.ipynb) |

### Part 1 — Classical supervised learning ✅

| # | Notebook | What you build / see | Open |
|---|---|---|---|
| 05 | **Linear regression** | Gaussian likelihood → least squares, the normal equation and its projection geometry, GD vs. SGD, basis functions, ridge as MAP, coefficient uncertainty | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/05_linear_regression.ipynb) |
| 06 | **Bias–variance & cross-validation** | the bias²+variance+noise decomposition measured by simulation, learning curves, K-fold CV from scratch, data leakage on pure noise, double descent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/06_bias_variance_cv.ipynb) |
| 07 | **Regularization & feature selection** | best-subset and stepwise search, soft thresholding, the Lasso by coordinate descent, regularization paths, elastic net, compressed-sensing image recovery | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/07_regularization_feature_selection.ipynb) |
| 08 | **Logistic regression** | the sigmoid and log-odds, cross-entropy by MLE, convexity, separable data and MAP, Newton/IRLS, softmax regression on handwritten digits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/08_logistic_regression.ipynb) |
| 09 | **Naive Bayes & generative classifiers** | Bayes decision rule and Bayes error, QDA/LDA, why LDA's posterior is logistic, a spam filter with Laplace smoothing, sampling new digits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/09_naive_bayes_generative.ipynb) |
| 10 | **KNN & classifier evaluation** | KNN from scratch, the curse of dimensionality, confusion matrices, precision/recall/F1, ROC and PR curves built step by step, cost-based thresholds, calibration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/10_knn_evaluation.ipynb) |
| 11 | **Support vector machines** | maximum margin, the dual and support vectors, soft margins as hinge loss, Pegasos, the kernel trick lifted to 3-D, an RBF SVM solved from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/11_svm.ipynb) |
| 12 | **Decision trees & random forests** | CART split search by hand and in code, Gini vs. entropy, pruning, the bootstrap, bagging and random forests from scratch, OOB error, importance pitfalls | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/12_trees_forests.ipynb) |
| 13 | **Boosting** | AdaBoost round by round, the exponential-loss derivation, margins, label noise, gradient boosting as fitting residuals, shrinkage and early stopping | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/13_boosting.ipynb) |

### Part 2 — Classical unsupervised learning ✅

| # | Notebook | What you build / see | Open |
|---|---|---|---|
| 14 | **Clustering** | distances and representation, k-means animated with Voronoi cells, convergence and k-means++, elbow and silhouette, color-quantization image compression, k-medians/k-medoids, dendrograms, DBSCAN and spectral clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/14_clustering.ipynb) |
| 15 | **Gaussian mixtures & EM** | mixtures as a hidden-label story, EM from scratch in 1-D and 2-D, the ELBO lower bound drawn step by step, k-means as hard EM, collapsing components, BIC, sampling new digits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/15_gmm_em.ipynb) |
| 16 | **Principal component analysis** | the widest-shadow direction, the eigen-derivation, SVD vs. covariance, scaling pitfalls, compression and denoising, eigenfaces and face-space detection, yield-curve factors, probabilistic PCA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/16_pca.ipynb) |
| 17 | **Recommender systems** | a sparse rating matrix, alternating least squares by hand, matrix factorization with biases, learned genre embeddings, cosine vs. Pearson collaborative filtering, ranking metrics and their biases | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/17_recommender_systems.ipynb) |
| 18 | **Anomaly detection** | Mahalanobis distance and χ² thresholds, robust covariance (MCD), KDE, the one-class SVM dual from scratch, isolation forests, novelty detection compared, CUSUM change points, network changes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/18_anomaly_detection.ipynb) |

### Part 3 — Deep learning 🚧 *(in progress)*

| # | Notebook | What you build / see | Open |
|---|---|---|---|
| 19 | **Neural networks** | XOR and the perceptron learning rule, stacked perceptrons, universal approximation with sigmoid bumps, an MLP with backprop in NumPy, hidden layers bending space, vanishing gradients, weight decay | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/19_neural_networks.ipynb) |
| 20 | **PyTorch essentials** | tensors and autograd checked against hand derivations, linear regression four ways, `nn.Module`, Fashion-MNIST data loaders, the training loop, optimizers and LR range tests, dropout/BN, debugging by overfitting one batch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shan4949/ml-visualized/blob/main/notebooks/20_pytorch_essentials.ipynb) 🚧 |
| 21 | **Convolutional neural networks** | the pixel-shuffle test, convolution animated and built with im2col, padding/stride, pooling, a CNN vs. an MLP, learned filters and feature maps, receptive fields, augmentation, residual networks, saliency and occlusion | 🔜 coming soon |
| 22 | **RNNs & LSTMs** | an RNN cell from scratch, backprop through time, vanishing/exploding gradients and clipping, the LSTM conveyor belt and gates, a long-memory test, forecasting, a character-level Shakespeare model | 🔜 coming soon |

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
