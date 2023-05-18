import numpy as np
import time
from numba import jit, cuda

# Matrix size
N = 1000

# Create arrays of size NxN
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# CPU version
@jit(nopython=True)
def matriz_mult_cpu(A, B):
    result = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result

# GPU version
@cuda.jit
def matriz_mult_gpu(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# Create a result object
C = np.zeros_like(A)

# Calculates CPU time
start_time_cpu = time.time()
C_cpu = matriz_mult_cpu(A, B)
end_time_cpu = time.time()

# Configure the grid and block
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Calculates time for GPU
start_time_gpu = time.time()
matriz_mult_gpu[blockspergrid, threadsperblock](A, B, C)
end_time_gpu = time.time()

print("CPU Time:  %s seconds" % (end_time_cpu - start_time_cpu))
print("GPU Time:  %s seconds" % (end_time_gpu - start_time_gpu))