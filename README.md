# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# gauravjha850's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/eigenvector850.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Argmax | Implement a parallel CUDA argmax reduction that returns the lowest index when multiple elements share the maximum. | https://www.tensortonic.com/problems/argmax |
| Argmin | Implement a parallel CUDA argmin reduction that returns the lowest index when multiple elements share the minimum. | https://www.tensortonic.com/problems/argmin |
| 1D Convolution | Implement 1D cross-correlation in CUDA with one thread per output element and register accumulation over the kernel. | https://www.tensortonic.com/problems/conv-1d |
| 2D Convolution | Implement 2D cross-correlation in CUDA with a two-dimensional grid, row-major indexing, and per-output accumulation. | https://www.tensortonic.com/problems/conv-2d |
| 3D Convolution | Implement naive 3D cross-correlation in CUDA with a three-dimensional launch grid and row-major flattened indexing. | https://www.tensortonic.com/problems/conv-3d |
| Cross-Entropy Loss | Implement numerically stable mean cross-entropy loss in CUDA using row-wise log-sum-exp and batch reduction. | https://www.tensortonic.com/problems/cross-entropy |
| Cosine Similarity | Compute cosine similarity in CUDA by reducing the dot product and both vector norms into one scalar result. | https://www.tensortonic.com/problems/cuda-cosine-similarity |
| Dot Product | Implement a multi-block CUDA dot-product reduction that combines partial sums into one scalar output. | https://www.tensortonic.com/problems/cuda-dot-product |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/problems/cuda-gelu |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/problems/cuda-leaky-relu |
| Matrix Transpose | Implement matrix transpose in CUDA with a two-dimensional launch grid, row-major buffers, and bounds-checked writes. | https://www.tensortonic.com/problems/cuda-matrix-transpose |
| Decision Tree Best Split | Find the best decision-tree split by evaluating candidate feature thresholds and selecting the largest impurity reduction. | https://www.tensortonic.com/problems/decision-tree-split |
| Dropout | Implement inverted dropout in CUDA using a supplied binary mask and scaling retained elements by the keep probability. | https://www.tensortonic.com/problems/dropout |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Gaussian Naive Bayes | Fit Gaussian Naive Bayes class statistics and predict labels from priors and feature likelihoods. | https://www.tensortonic.com/problems/gaussian-naive-bayes |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Matrix-Vector Multiplication | Implement row-major CUDA matrix-vector multiplication with one thread computing each output row. | https://www.tensortonic.com/problems/gemv |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Hadamard Product | Implement elementwise matrix multiplication in CUDA using a two-dimensional grid and row-major bounds-checked indexing. | https://www.tensortonic.com/problems/hadamard-product |
| Compute Information Gain for a Split | Compute information gain for a decision-tree split from parent entropy and weighted child entropies. | https://www.tensortonic.com/problems/information-gain |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| K-Means Centroid Update | Update K-means centroids as cluster means while applying the required behavior for empty clusters. | https://www.tensortonic.com/problems/k-means-centroid-update |
| KNN Distance + Neighbor Lookup | Find the nearest neighbors of a query point by computing and ordering Euclidean distances to training samples. | https://www.tensortonic.com/problems/knn-distance |
| L1 Normalization | Implement CUDA L1 vector normalization by reducing absolute values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l1-normalize |
| L2 Normalization | Implement CUDA L2 vector normalization by reducing squared values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l2-normalize |
| Layer Normalization | Implement fused row-wise LayerNorm in CUDA with shared-memory mean and variance reduction, affine scale, and bias. | https://www.tensortonic.com/problems/layer-norm |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Implement Majority Class Classifier | Fit a majority-class baseline and predict the most frequent training label for every requested sample. | https://www.tensortonic.com/problems/majority-classifier |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/problems/matrix-addition |
| Matrix Multiplication | Implement row-major matrix multiplication in CUDA with one thread per output element and inner-product accumulation. | https://www.tensortonic.com/problems/matrix-multiplication |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Max of Array | Implement a multi-block CUDA maximum reduction that combines block-local maxima into one scalar output. | https://www.tensortonic.com/problems/max-of-array |
| Max Pool 2D | Implement 2D max pooling in CUDA with one thread per output, configurable windows and strides, and row-major indexing. | https://www.tensortonic.com/problems/max-pool-2d |
| Mean and Variance | Compute population mean and variance in CUDA with device-side reductions and synchronized scalar outputs. | https://www.tensortonic.com/problems/mean-variance |
| Min of Array | Implement a multi-block CUDA minimum reduction that combines block-local minima into one scalar output. | https://www.tensortonic.com/problems/min-of-array |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Naive Bayes Log-Likelihood (Bernoulli) | Compute Bernoulli Naive Bayes log-likelihoods from binary features, class priors, and feature probabilities. | https://www.tensortonic.com/problems/naive-bayes-bernoulli |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| Outer Product | Compute a vector outer product in CUDA with a two-dimensional grid, row-major output, and bounds-checked indexing. | https://www.tensortonic.com/problems/outer-product |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Random Forest Majority Vote | Combine multiple decision-tree predictions with majority voting and deterministic handling of tied classes. | https://www.tensortonic.com/problems/random-forest-vote |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/problems/relu |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Ridge Regression | Fit ridge regression with L2 regularization using the closed-form solution required by the problem. | https://www.tensortonic.com/problems/ridge-regression |
| RMS Normalization | Implement row-wise RMS normalization in CUDA with sum-of-squares reduction, numerical stability, and learnable scaling. | https://www.tensortonic.com/problems/rms-norm |
| Robust Scaling | Scale numeric features using their median and interquartile range with constant-spread handling. | https://www.tensortonic.com/problems/robust-scaling |
| Scaled Dot-Product Attention | Implement single-head scaled dot-product attention in CUDA with stable row-wise softmax and value aggregation. | https://www.tensortonic.com/problems/scaled-dot-product-attention |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/problems/sigmoid |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/problems/softmax |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Sum of Array | Implement a multi-block CUDA sum reduction that combines partial block sums into one scalar output. | https://www.tensortonic.com/problems/sum-of-array |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/problems/swish |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/problems/tanh |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Tiled Matrix Multiplication | Implement shared-memory tiled matrix multiplication in CUDA with synchronized tile loads and partial-tile handling. | https://www.tensortonic.com/problems/tiled-matmul |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/problems/triton-fused-softmax |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/problems/triton-logsumexp |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/problems/triton-mean-variance |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-subtract |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/eigenvector850)
<!-- tensortonic:end -->
