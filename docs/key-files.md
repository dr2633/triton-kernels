### [Introduction and Motivations](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)

GPUs remain incredibly challenging to optimize for locality and parallelism, especially for computations that cannot be efficiently implemented using a combination of pre-existing optimized primitives. To make matters worse, GPU architectures are also rapidly evolving and specializing, as evidenced by the addition of tensor cores to NVIDIA (and more recently AMD) micro-architectures.

This tension between the computational opportunities offered by DNNs and the practical difficulty of GPU programming has created substantial academic and industrial interest for Domain-Specific Languages (DSLs) and compilers. Regrettably, these systems – whether they be based on polyhedral machinery (e.g., Tiramisu [BAGHDADI2021], Tensor Comprehensions [VASILACHE2018]) or scheduling languages (e.g., Halide [JRK2013], TVM [CHEN2018]) – remain less flexible and (for the same algorithm) markedly slower than the best handwritten compute kernels available in libraries like cuBLAS, cuDNN or TensorRT.

The main premise of this project is the following: programming paradigms based on blocked algorithms [LAM1991] can facilitate the construction of high-performance compute kernels for neural networks. We specifically revisit traditional “Single Program, Multiple Data” (SPMD [AUGUIN1983]) execution models for GPUs, and propose a variant in which programs – rather than threads – are blocked. For example, in the case of matrix multiplication, CUDA and Triton differ as follows:


**Blocked Program, Scalar Threads**


CUDA
```python

#pragma parallel
for(int m = 0; m < M; m++)
#pragma parallel
for(int n = 0; n < N; n++){
  float acc = 0;
  for(int k = 0; k < K; k++)
    acc += A[m, k] * B[k, n];

  C[m, n] = acc;
}
```

Triton 
```python
#pragma parallel
for(int m = 0; m < M; m += MB)
#pragma parallel
for(int n = 0; n < N; n += NB){
  float acc[MB, NB] = 0;
  for(int k = 0; k < K; k += KB)
    acc +=  A[m:m+MB, k:k+KB]
          @ B[k:k+KB, n:n+NB];
  C[m:m+MB, n:n+NB] = acc;
}
```


A key benefit of this approach is that it leads to block-structured iteration spaces that offer programmers more flexibility than existing DSLs when implementing sparse operations, all while allowing compilers to aggressively optimize programs for data locality and parallelism.

_Compare Triton and CUDA -- provide visualization of block-structured iteration spaces with indexing_


### Challenges and Tritan Implementation of Block-Level Data-Flow Analysis 

The main challenge posed by our proposed paradigm is that of work scheduling, i.e., how the work done by each program instance should be partitioned for efficient execution on modern GPUs. To address this issue, the Triton compiler makes heavy use of block-level data-flow analysis, a technique for scheduling iteration blocks statically based on the control- and data-flow structure of the target program. The resulting system actually works surprisingly well: our compiler manages to apply a broad range of interesting optimization automatically (e.g., automatic coalescing, thread swizzling, pre-fetching, automatic vectorization, tensor core-aware instruction selection, shared memory allocation/synchronization, asynchronous copy scheduling). Of course doing all this is not trivial; one of the purposes of this guide is to give you a sense of how it works.



### [Related Work](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html)

Highlight its differences with the two leading approaches in this domain: polyhedral compilation and scheduling languages.

### [Debugging Triton](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)



### [Original Press Release](https://openai.com/index/triton/)

