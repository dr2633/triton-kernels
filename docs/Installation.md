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





