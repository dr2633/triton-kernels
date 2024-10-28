
Keynote: Phillipe Tillet
https://www.youtube.com/watch?v=o3DrHb-mVLM

Developed tools to improve user productivity 

**Interpreter**: a numpy-based interpreter for Triton kernels 

Useful for: 

- Running programs line by line sequentially 
- Setting breakpoints and investigating intermediate values 
- Saving time 


**Proton**: a CUPTI-based profiler for Triton 

- Trace kernels that were called 
- Locate bottlenecks in programs 
- Analyze timings and compute roofline 
- Flag performance regressions 


_Integrate into the tutorials_


TritonGPU-IR cannot be easily re-used across backends 

- **Linear Layouts**: A new framework for unifying layouts within and across backends: 

Blocked/DotOperand/MMA/Slice etc. --> Linear 

NvidiaMMAv2/v3/ AmdMFMA --> Linear 

- **Structured Memory Accesses**: still does not have portable abstractions for specialized memory units (ie. TMAs)
https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
Tensor Memory Accelerator (TMA)
TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory, in both directions, as well as between the shared memory regions of different SMs in the same cluster
Additionally, for writes from shared memory to global memory, it allows specifying element wise reduction operations such as add/min/max as well as bitwise and/or for most common data types.



Adding modularity to the codebase (reducing code complexity)




Keynote: Aparna Ramani 
https://www.youtube.com/watch?v=WgbrAe7kQSE

Ajit Matthews 
https://www.youtube.com/watch?v=nglpa_6cYYI

Pipelining Persistent Kernels 
https://www.youtube.com/watch?v=PAsL680eWUw

Hardware Heterogeneity (Meta)
https://www.youtube.com/watch?v=g90lgPNkBh4

Dev Tools: Proton/Interpreter 
https://www.youtube.com/watch?v=Av1za_0o2Qs

AWS Trainium and Inferentia2
https://www.youtube.com/watch?v=t5dh54d8sTE

Triton CPU 
https://www.youtube.com/watch?v=obGM7nujV00

Compiler Tools: Writing an MLIR Pass
https://www.youtube.com/watch?v=etlFyqSsmL0

EXO: Exocompilation 
https://exo-lang.dev

We’ve shown that we can use Exo to quickly write code that’s as performant as Intel’s hand-optimized Math Kernel Library. We also have an ongoing collaboration with UC Berkeley to create code for GEMMINI, their open-source machine learning accelerator. We’re actively working with engineers and researchers at a number of companies, and are eager to work with more!

**Complete Afternoon Session** 
https://www.youtube.com/watch?v=ONrKkI7KhU4
