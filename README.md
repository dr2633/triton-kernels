# Triton GPU Optimization Tutorials

Welcome to the **Triton GPU Optimization Tutorials**! This project is a collection of Jupyter notebooks designed to introduce users to GPU programming and optimization using [Triton](https://github.com/openai/triton), with performance comparisons to PyTorch (CUDA).

Each notebook is structured to progressively build on the previous, covering concepts ranging from GPU fundamentals to advanced custom kernels for deep learning workflows. By the end, users should have a solid foundation in GPU programming and be familiar with Triton’s capabilities for high-performance computation.

---

## Project Outline

### Notebooks
Each notebook has links to run in Google Colab for immediate experimentation with GPU access.

1. **01_GPU_Intro_and_Elementwise_Operations**: [Colab](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebooks/01_GPU_Intro_and_Elementwise_Operations.ipynb) | [GitHub](notebooks/01_GPU_Intro_and_Elementwise_Operations.ipynb)
   - **Objective**: Introduces basic GPU concepts, Triton setup, and elementwise operations.

2. **02_Fused_Operations_and_Memory_Optimization**: [Colab](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebooks/02_Fused_Operations_and_Memory_Optimization.ipynb) | [GitHub](notebooks/02_Fused_Operations_and_Memory_Optimization.ipynb)
   - **Objective**: Explores fused operations and memory optimization.

3. **03_Softmax_Optimization_and_Memory_Reuse**: [Colab](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebooks/03_Softmax_Optimization_and_Memory_Reuse.ipynb) | [GitHub](notebooks/03_Softmax_Optimization_and_Memory_Reuse.ipynb)
   - **Objective**: Implements softmax with memory reuse.

4. **04_Matrix_Multiplication_with_Blocking_and_Tiling**: [Colab](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebooks/04_Matrix_Multiplication_with_Blocking_and_Tiling.ipynb) | [GitHub](notebooks/04_Matrix_Multiplication_with_Blocking_and_Tiling.ipynb)
   - **Objective**: Demonstrates matrix multiplication with tiling strategies.

5. **05_Custom_Loss_Functions_and_Backpropagation_in_Triton** (Advanced): [Colab](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebooks/05_Custom_Loss_Functions_and_Backpropagation_in_Triton.ipynb) | [GitHub](notebooks/05_Custom_Loss_Functions_and_Backpropagation_in_Triton.ipynb)
   - **Objective**: Develops custom loss functions and backpropagation.

---

## Installation Instructions

### Compatibility
Before installation, please review the [Compatibility section on Triton’s GitHub page](https://github.com/triton-lang/triton) for supported platforms, operating systems, and hardware requirements.

> **Note for macOS Users**: Triton is currently not compatible with macOS. To try Triton on a macOS system, you can use [Google Colab](https://colab.research.google.com/), which provides a Linux-based environment with access to GPUs.

### Binary Installation (Recommended)

#### Stable Release

To install the latest stable release of Triton, use `pip`:


```bash
pip install triton
```
**Using Google Colab on macOS**

If you're on macOS, follow these steps to set up Triton in Google Colab:

**1. Open Google Colab:**

Go to Google Colab and create a new notebook.

**2. Enable GPU Runtime:**

- In Colab, click on Runtime > Change runtime type.
- Set **Hardware accelerator to GPU**, then click **Save**.


**Note:** Google Colab often provides NVIDIA T4 GPUs, which are compatible with Triton and ideal for running practice experiments. The T4 GPU has ample memory and compute power for experimenting with Triton kernels, making it suitable for learning and small-scale testing.

**Install Triton in Colab:**
In the first cell of your Colab notebook, run the following command to install Triton:

```python
!pip install triton
```

**4. Verify the Installation**

After installing, you can run a simple command to verify the installation: 

```python
import triton 
print("Triton version:", triton.__version__)
```

**5. Run Triton Code**

With your installation verified, you're now ready to start using Triton! 

You can write Triton kernels or run existing scripts directly within the Colab environment. 





