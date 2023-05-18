# GPU vs CPU Time Comparison

This Python script calculates the time it takes to perform matrix multiplication using both the CPU and GPU. It utilizes the Python **numba** library which allows for just-in-time (JIT) compilation to optimize the performance of Python code.

## Prerequisites

- Python 3.6+
- numba
- A compatible NVIDIA GPU for utilizing GPU acceleration. If you don't have one, the code will default to CPU computation.

## Usage

1. Clone this repository.
```bash
git clone https://github.com/juliansprt/CUDATestProcess
```
2. Install the required Python package.
```bash
pip install numba
```
3. Make sure you have CUDA toolkit installed. You can download it from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

## Running the Script
Simply run the Python script app.py in the cloned directory:
```bash
python app.py
```

## How it works
The script creates two large matrices of size **NxN**, where **N** is 1000 in this case.
The script first calculates the multiplication of these two matrices using the CPU. It does this using the **@jit(nopython=True)** decorator which allows **numba** to optimize the function.
Then the script calculates the multiplication of the matrices using the GPU. It does this using the **@cuda.jit** decorator which tells **numba** to compile this function for the GPU.
Finally, the script prints out the time taken for both CPU and GPU calculations for comparison.

## Configurable Parameters
You can modify the **N** value at the top of the script to change the size of the matrices used in the calculations. This allows you to see how increasing the size of the matrices affects the CPU and GPU calculation times.

## Notes
- The GPU computation time includes the time it takes to transfer data from the CPU to the GPU and vice versa.
- The exact execution times will depend heavily on your specific hardware and how Python has been compiled and optimized for your system.

Please modify the script as needed for your use case. If you encounter any issues, please let me know.

## Benchmarking Results
The following are the benchmarking results obtained when running the script on a system with an **NVidia RTX 3080 GPU** and a **Core i9 12900KF CPU**.

| Matriz Size | CPU Time (Seconds) | GPU Time (Seconds) |
| --- | --- | --- |
|100x100|0.36|0.30|
|500x500|0.41|0.35|
|1000x1000|0.85|0.23|
|1500x1500|1.98|0.25|
|2000x2000|11.85|0.32|
|5000x5000|579.82 (9 min)|1.37|

## Conclusions
These times indicate the performance advantage of using GPU-based calculations for larger matrices due to its parallel processing capabilities. Note that the GPU's performance might vary significantly based on the specific computational task, the characteristics of the GPU, and the efficiency of the GPU computation libraries. It's important to benchmark your own specific applications to understand the performance characteristics.

Based on the benchmarking results, we can observe a significant speedup when using the GPU for matrix multiplication, particularly as the size of the matrices increases. The GPU times remain fairly consistent and scale much better compared to the CPU times, which increase considerably with the matrix size.

This demonstrates the inherent parallelism provided by the GPU. Unlike a CPU which has a few cores optimized for sequential serial processing, a GPU has a parallel architecture consisting of thousands of smaller cores designed for handling multiple tasks simultaneously.

For tasks like matrix multiplication, which can be broken down into many smaller independent calculations, a GPU can perform many calculations simultaneously, leading to a significant reduction in time.

**This underscores the importance of GPU-accelerated computing for large-scale calculations or data-intensive tasks.**

However, it's worth noting that the exact performance gains depend on the specifics of the computation task, the size of the data, and the particularities of the GPU and CPU hardware. Also, one must consider the time it takes to transfer data between the CPU and GPU. In cases where the dataset is small or computations can't be highly parallelized, CPU might still perform better.

Therefore, it's important to understand the nature of the computational task and do benchmarking to decide the optimal approach for your use case.