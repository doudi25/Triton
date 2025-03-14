# Triton 100 days challenge
This repository will track my journey as I code kernels in Triton every day. Triton is a domain-specific language (DSL) and compiler embedded within Python. It allows you to write GPU kernels using a Python-like syntax, which is then compiled and executed on GPUs.
## Day 1: Triton Addition Kernel
Implemented a basic element-wise addition kernel using Triton to add two vectors. This serves as an introductory exercise to understand Triton's programming model.

## Day 2: SwiGLU Kernel
  Developed a kernel for the SwiGLU activation function, an advanced activation function that can enhance model performance. This function combines the Swish and Gated Linear Unit (GLU) activations.

## Day 3: RGB to Grayscale
Created a kernel to convert RGB images to grayscale, a common preprocessing step in image analysis. This operation reduces the complexity of image data by eliminating color information.

## Day 4: Tanh Kernel
Implemented a kernel to compute the hyperbolic tangent (tanh) activation function, commonly used in neural networks. This function introduces non-linearity into models, enabling them to learn complex patterns.

## Day 5: Flash Attention 2
Developed an efficient implementation of the Flash Attention mechanism, enhancing the speed and scalability of attention-based models. This technique reduces memory usage and computational overhead in transformer architectures.

## Day 6: Layer Norm
Created a kernel for layer normalization, ensuring consistent scaling and shifting of inputs across different layers. This technique is crucial for maintaining the stability of neural network training.

## Day 7: Element-wise Addition
Developed a kernel for element-wise addition of two tensors, showcasing Triton's ability to handle parallel computations efficiently. This builds upon the initial addition kernel by extending the operation to tensors.

## Day 8: Fused Softmax
Created a fused kernel to compute the softmax function, optimizing performance by combining multiple operations into a single kernel. This approach reduces memory bandwidth requirements and enhances computational efficiency.

## Day 9: Triton Conv2d
Implemented a kernel for 2-dimensional convolution, essential in image processing and computer vision tasks. This operation is fundamental in extracting spatial features from images.

## Day 10: Triton Conv1d
Developed a kernel for 1-dimensional convolution, fundamental in processing sequential data such as time-series or audio signals. This operation is essential in extracting features from sequential data.

## Day 11: MaxPool1d
Created a kernel for 1-dimensional max pooling, a down-sampling operation commonly used in convolutional neural networks to reduce spatial dimensions. This helps in reducing the computational load and controlling overfitting.

## Day 12: MaxPool2d
Developed a kernel for 2-dimensional max pooling, extending the max pooling operation to two spatial dimensions. This is widely used in image processing tasks to down-sample feature maps.

## Day 13: Inverting Colors
Implemented a kernel to invert the colors of an image, transforming each pixel to its complementary color. This operation is useful in various image processing applications.

## Day 14: BatchNorm
Created a kernel for batch normalization, a technique to stabilize and accelerate neural network training by normalizing inputs across a mini-batch. This helps in reducing internal covariate shift.

## Day 15: RMS LayerNorm
Developed a kernel for Root Mean Square Layer Normalization (RMSNorm), a variant of layer normalization that normalizes inputs to improve training stability. This technique is beneficial in stabilizing and accelerating neural network training.

## Day 16: L2 Norm
Implemented a kernel to calculate the L2 norm (Euclidean norm) of a vector, useful in normalization processes. This is commonly used in machine learning to measure vector magnitudes.

## Day 17: Inner Product
Created a kernel to compute the inner product (dot product) of two vectors, fundamental in various linear algebra computations. This operation is essential in numerous machine learning algorithms.
## Day 18: Outer Product 
Developed a kernel to compute the outer product of two vectors, producing a matrix where each element is the product of elements from the input vectors. This operation is crucial in linear algebra, with applications in machine learning models, tensor operations, and feature space expansions.
## Day 19: Dropout Kernel
Implemented a custom dropout kernel using Triton, designed to randomly zero out elements in a tensor with a given probability pb. The surviving elements are scaled by 1/(1-pb) to maintain the expected value of the output.
The kernel is optimized with autotuning, testing various block sizes and warp configurations to maximize GPU efficiency. Wrapped in a simple PyTorch function, it seamlessly integrates with CUDA tensors for fast and flexible dropout operations.
## Day 20: Sum Kernel
Built a Triton-based kernel to compute the sum of elements along dim=1 of a tensor. This operation is key for tasks like feature aggregation or dimensionality reduction in machine learning. Leveraging Triton's GPU optimizations, the kernel ensures efficient parallel execution and memory handling.
## Day 21: Optimized_Conv2d
I developed a highly optimized 2D convolution kernel, a key operation in image processing and computer vision for extracting spatial features from images. My implementation significantly outperforms PyTorch's native CUDA kernel by strategically parallelizing computations across all dimensions—batch size, number of kernels, channels, height, and width—resulting in superior performance.
## Day 22: AvgPool2d
Implemented an AvgPool2d kernel from scratch! 
It reduces image size, typically used after convolutional layers. The kernel efficiently works in parallel across batch, channels, height, and width—yielding results roughly equivalent to PyTorch's native implementation.
## Day 23: BatchNorm2d
Implemented a Triton-based BatchNorm2d kernel, a crucial layer in deep learning for stabilizing and accelerating training. The kernel normalizes activations across the batch and spatial dimensions, achieving performance on par with PyTorch's native implementation. By leveraging Triton's efficient parallelization and memory management, the kernel ensures optimal GPU utilization while maintaining numerical accuracy.
## Day 24: LayerNorm
I implemented a Triton-based LayerNorm kernel, a fundamental layer in deep learning that normalizes activations across feature dimensions. By parallelizing computations across batches and features, the kernel efficiently reduces variance and stabilizes training.
## Day 25: Silu 
I implemented a Triton-based SiLU (Sigmoid Linear Unit) kernel, covering both forward and backward passes. The kernel utilizes the maximum number of warps per streaming multiprocessor (64 warps) on Tesla T4 GPUs, ensuring full parallelism and optimal hardware utilization. Additionally, I tested the kernel's compatibility with PyTorch's autograd engine, confirming smooth integration for automatic differentiation.
## Day 26: LayerNorm_backward
I implemented the Triton-based LayerNorm backward pass, specifically calculating gradients for the learnable weight parameter 𝛾 and beta . The kernel leverages parallel reductions to efficiently compute the sum of element-wise products between the output gradients and the normalized inputs,parallelization is applied across batch size and feature dimension for maximum efficiency.
## Day 27: GELU
I implemented a Triton-based GeLU (Gaussian Error Linear Unit) forward pass, using the approximation version for faster computation. This approach balances accuracy and performance by leveraging a smooth, non-linear activation function while avoiding expensive operations like the error function (erf). The kernel efficiently utilizes parallelism across batches and feature dimensions.
## Day 28: Mish
I implemented a Triton-based Mish activation kernel, implementing both forward and backward passes for this smooth, non-monotonic activation function defined as f(x) = x * tanh(softplus(x)). The kernel optimizes computation by parallelizing across batch and feature dimensions, achieving efficient GPU utilization with 64 warps on Tesla T4 hardware. I validated the implementation against PyTorch's built-in Mish function, ensuring gradient accuracy through autograd integration, while maintaining numerical stability for the complex backward pass computation.

