from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return
        to_index(i, out_shape, out_index[:out_shape.size])
        broadcast_index(out_index[:out_shape.size], out_shape, in_shape, in_index[:in_shape.size])
        out_pos = index_to_position(out_index[:out_shape.size], out_strides)
        in_pos = index_to_position(in_index[:in_shape.size], in_strides)
        out[out_pos] = fn(in_storage[in_pos])
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return
        to_index(i, out_shape, out_index[:out_shape.size])
        broadcast_index(out_index[:out_shape.size], out_shape, a_shape, a_index[:a_shape.size])
        broadcast_index(out_index[:out_shape.size], out_shape, b_shape, b_index[:b_shape.size])
        out_pos = index_to_position(out_index[:out_shape.size], out_strides)
        a_pos = index_to_position(a_index[:a_shape.size], a_strides)
        b_pos = index_to_position(b_index[:b_shape.size], b_strides)
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    # TODO: Implement for Task 3.3.
    if i >= size:
        return
    cache[pos] = a[i]
    cuda.syncthreads()
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and i + stride < size:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]
    # raise NotImplementedError("Need to implement for Task 3.3")


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        if out_pos >= out_size:
            return
        if pos >= a_shape[reduce_dim]:
            return
        to_index(out_pos, out_shape, out_index[:out_shape.size])
        out_index[reduce_dim] = pos
        out_pos = index_to_position(out_index[:out_shape.size], out_strides)
        cache[pos] = a_storage[index_to_position(out_index, a_strides)]
        cuda.syncthreads()
        if pos == 0:
            cache[0] = fn(cache[0], reduce_value)
            for i in range(1, a_shape[reduce_dim]):
                cache[0] = fn(cache[0], cache[i])
            out[out_pos] = cache[0]


    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    idx_x = cuda.threadIdx.x
    idx_y = cuda.threadIdx.y
    if idx_x >= size or idx_y >= size:
        return
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    c_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    a_shared[idx_y, idx_x] = a[idx_y * size + idx_x]
    b_shared[idx_y, idx_x] = b[idx_y * size + idx_x]
    c_shared[idx_y, idx_x] = 0
    cuda.syncthreads()
    for i in range(size):
        c_shared[idx_y, idx_x] += a_shared[idx_y, i] * b_shared[i, idx_x]
    out[idx_y * size + idx_x] = c_shared[idx_y, idx_x]

jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")
    if batch >= out_shape[0]:
        return
    if i >= out_shape[1] and j >= out_shape[2]:
        return

    out_idx = cuda.local.array(3, numba.int32)
    out_idx[0] = batch
    out_idx[1] = i
    out_idx[2] = j

    a_idx_bcasted = cuda.local.array(3, numba.int32)
    b_idx_bcasted = cuda.local.array(3, numba.int32)
    a_idx_bcasted[0] = batch
    b_idx_bcasted[0] = batch
    a_idx_bcasted[1] = i
    b_idx_bcasted[2] = j

    a_shape_bcasted = cuda.local.array(3, numba.int32)
    a_shape_bcasted[0] = out_shape[0]
    a_shape_bcasted[1] = a_shape[1]
    a_shape_bcasted[2] = a_shape[2]
    b_shape_bcasted = cuda.local.array(3, numba.int32)
    b_shape_bcasted[0] = out_shape[0]
    b_shape_bcasted[1] = b_shape[1]
    b_shape_bcasted[2] = b_shape[2]

    a_idx = cuda.local.array(3, numba.int32)
    b_idx = cuda.local.array(3, numba.int32)

    vec_len = a_shape[2]

    out_value = 0
    offset = 0
    while offset < vec_len:
        a_idx_bcasted[2] = offset + pj
        b_idx_bcasted[1] = offset + pi
        broadcast_index(a_idx_bcasted, a_shape_bcasted, a_shape, a_idx)
        broadcast_index(b_idx_bcasted, b_shape_bcasted, b_shape, b_idx)
        a_shared[pi, pj] = a_storage[index_to_position(a_idx, a_strides)]
        b_shared[pi, pj] = b_storage[index_to_position(b_idx, b_strides)]
        cuda.syncthreads()
        residual = vec_len - offset
        k_range = min(BLOCK_DIM, residual)
        for k in range(k_range):
            out_value += a_shared[pi, k] * b_shared[k, pj]
        offset += BLOCK_DIM
        cuda.syncthreads()
    if i < out_shape[1] and j < out_shape[2]:
        out[index_to_position(out_idx, out_strides)] = out_value



tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
