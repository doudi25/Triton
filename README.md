# Triton 100 days challenge
This repository will track my journey as I code kernels in Triton every day. Triton is a domain-specific language (DSL) and compiler embedded within Python. It allows you to write GPU kernels using a Python-like syntax, which is then compiled and executed on GPUs.
-Day 1: Triton Addition Kernel
Implemented a basic element-wise addition kernel using Triton to add two vectors. This serves as an introductory exercise to understand Triton's programming model.

-Day 2: SwiGLU Kernel
  Developed a kernel for the SwiGLU activation function, an advanced activation function that can enhance model performance. This function combines the Swish and Gated Linear Unit (GLU) activations.

-Day 3: RGB to Grayscale
Created a kernel to convert RGB images to grayscale, a common preprocessing step in image analysis. This operation reduces the complexity of image data by eliminating color information.

-Day 4: Tanh Kernel
Implemented a kernel to compute the hyperbolic tangent (tanh) activation function, commonly used in neural networks. This function introduces non-linearity into models, enabling them to learn complex patterns.

-Day 5: Flash Attention 2
Developed an efficient implementation of the Flash Attention mechanism, enhancing the speed and scalability of attention-based models. This technique reduces memory usage and computational overhead in transformer architectures.

-Day 6: Layer Norm
Created a kernel for layer normalization, ensuring consistent scaling and shifting of inputs across different layers. This technique is crucial for maintaining the stability of neural network training.

-Day 7: Element-wise Addition
Developed a kernel for element-wise addition of two tensors, showcasing Triton's ability to handle parallel computations efficiently. This builds upon the initial addition kernel by extending the operation to tensors.

-Day 8: Fused Softmax
Created a fused kernel to compute the softmax function, optimizing performance by combining multiple operations into a single kernel. This approach reduces memory bandwidth requirements and enhances computational efficiency.

-Day 9: Triton Conv2d
Implemented a kernel for 2-dimensional convolution, essential in image processing and computer vision tasks. This operation is fundamental in extracting spatial features from images.

-Day 10: Triton Conv1d
Developed a kernel for 1-dimensional convolution, fundamental in processing sequential data such as time-series or audio signals. This operation is essential in extracting features from sequential data.

-Day 11: MaxPool1d
Created a kernel for 1-dimensional max pooling, a down-sampling operation commonly used in convolutional neural networks to reduce spatial dimensions. This helps in reducing the computational load and controlling overfitting.

-Day 12: MaxPool2d
Developed a kernel for 2-dimensional max pooling, extending the max pooling operation to two spatial dimensions. This is widely used in image processing tasks to down-sample feature maps.

-Day 13: Inverting Colors
Implemented a kernel to invert the colors of an image, transforming each pixel to its complementary color. This operation is useful in various image processing applications.

-Day 14: BatchNorm
Created a kernel for batch normalization, a technique to stabilize and accelerate neural network training by normalizing inputs across a mini-batch. This helps in reducing internal covariate shift.

-Day 15: RMS LayerNorm
Developed a kernel for Root Mean Square Layer Normalization (RMSNorm), a variant of layer normalization that normalizes inputs to improve training stability. This technique is beneficial in stabilizing and accelerating neural network training.

-Day 16: L2 Norm
Implemented a kernel to calculate the L2 norm (Euclidean norm) of a vector, useful in normalization processes. This is commonly used in machine learning to measure vector magnitudes.

-Day 17: Inner Product
Created a kernel to compute the inner product (dot product) of two vectors, fundamental in various linear algebra computations. This operation is essential in numerous machine learning algorithms.


