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


**Install Triton in Colab:**
In the first cell of your Colab notebook, run the following command to install Triton:

```python
!pip install triton
```


