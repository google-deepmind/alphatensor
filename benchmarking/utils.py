# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for working with matrix multiplication tensor factorizations."""

import gc
import timeit
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tree

BlockMatrix = List[List[jnp.ndarray]]


def block_split(matrix: jnp.ndarray, n_rows: int, n_cols: int) -> BlockMatrix:
  """Splits `matrix` into a `n_rows x n_cols` block matrix."""
  rows = jnp.split(matrix, n_rows, axis=0)
  return [jnp.split(row, n_cols, axis=1) for row in rows]


def get_matrix_multiplication_tensor(n: int) -> np.ndarray:
  """Returns the matrix multiplication tensor T_n = <n, n, n>.

  For n >= 1, T_n is a 3D tensor of shape [n*n, n*n, n*n] representing
  - the bilinear operation (A, B) -> (AB)^T where A, B are two [n, n] matrices,
  - or equivalently the trilinear operation (A, B, C) -> trace(ABC), where
    A, B, C are three [n, n] matrices.

  Args:
    n: Size of the matrix multiplication tensor to be returned.
  Returns:
    NumPy array of shape [n^2, n^2, n^2] representing the tensor <n, n, n>.
  """
  result = np.full((n ** 2, n ** 2, n ** 2), 0, dtype=np.int32)
  for i in range(n):
    for j in range(n):
      for k in range(n):
        result[i * n  + j][j * n + k][k * n + i] = 1
  return result


def algorithm_from_factors(
    factors: np.ndarray) -> Callable[[BlockMatrix, BlockMatrix], BlockMatrix]:
  """Returns a JAX function implementing the algorithm described by `factors`.

  Args:
    factors: Matricized factorization of a matrix multiplication tensor, i.e.
      an array of shape [3, n, n, rank].
  Returns:
    Function, which given two block matrices `a` and `b` returns the block
    matrix `c` given by `c = a @ b`.
  """
  assert factors[0].shape[0] == factors[1].shape[0]
  assert factors[1].shape[0] == factors[2].shape[0]
  factors = [factors[0].copy(), factors[1].copy(), factors[2].copy()]
  n = int(np.sqrt(factors[0].shape[0]))
  rank = factors[0].shape[-1]
  factors[0] = factors[0].reshape(n, n, rank)
  factors[1] = factors[1].reshape(n, n, rank)
  factors[2] = factors[2].reshape(n, n, rank)
  # The factors are for the transposed matrix multiplication tensor. So to
  # use the factors, we need to transpose back.
  factors[2] = factors[2].transpose(1, 0, 2)

  def f(a: BlockMatrix, b: BlockMatrix) -> BlockMatrix:
    """Multiplies block matrices `a` and `b`."""
    n = len(a)
    result = [[None] * n for _ in range(n)]
    for alpha in range(rank):
      left = None
      for i in range(n):
        for j in range(n):
          if factors[0][i, j, alpha] != 0:
            curr = factors[0][i, j, alpha] * a[i][j]
            if left is None:
              left = curr
            else:
              left += curr
      right = None
      for j in range(n):
        for k in range(n):
          if factors[1][j, k, alpha] != 0:
            curr = factors[1][j, k, alpha] * b[j][k]
            if right is None:
              right = curr
            else:
              right += curr

      matrix_product = left @ right

      for i in range(n):
        for k in range(n):
          if factors[2][i, k, alpha] != 0:
            curr = factors[2][i, k, alpha] * matrix_product
            if result[i][k] is None:
              result[i][k] = curr
            else:
              result[i][k] += curr
    return result

  return f


def _get_n_from_factors(factors: np.ndarray) -> int:
  """Computes the matrix multiplication tensor size n based on `factors`.

  E.g. when multiplying 2x2 matrices with Strassen, the `factors` are of shape
  [4, 7], and this function will return 2.

  Args:
    factors: [3, n^2, R] shaped NumPy array representing a factorization of T_n.
  Returns:
    n, the size of matrices being multiplied by the algorithm represented by
    `factors`.
  """
  u, v, w = factors
  # Assert that the tensor is a cube.
  assert u.shape[0] == v.shape[0]
  assert u.shape[0] == w.shape[0]
  n = int(np.sqrt(u.shape[0]))
  assert u.shape[0] == n ** 2
  return n


def _generate_random_matrices(matrix_dims: Tuple[int, int, int],
                              seed: int) -> Tuple[np.ndarray, np.ndarray]:
  """Generates two random NumPy matrices to be multiplied."""
  np.random.seed(seed)
  a = np.random.randn(matrix_dims[0], matrix_dims[1])
  b = np.random.randn(matrix_dims[1], matrix_dims[2])
  return a, b


def _device_put(*arrays, dtype: jnp.dtype) -> ...:
  """Converts NumPy arrays into JAX arrays and sends them to GPU."""
  return tree.map_structure(
      lambda x: jax.device_put(jnp.array(x).astype(dtype)), arrays)


def _get_baseline_op(matrix_dims: Tuple[int, int, int],
                     dtype: jnp.dtype,
                     n_repeat: int,
                     seed: int) -> Callable[[], None]:
  """Returns a function that applies `jnp.dot` `n_repeat` times."""
  full_a, full_b = _generate_random_matrices(matrix_dims, seed=seed)
  full_a, full_b = _device_put(full_a, full_b, dtype=dtype)

  # TODO(matejb): Why no `jax.jit` around `jnp.dot`?

  def _vanilla_single_timing() -> None:
    c = full_b
    for _ in range(n_repeat):
      c = jnp.dot(full_a, c)
    c.block_until_ready()

  return _vanilla_single_timing


def _get_factorization_op(factors: np.ndarray,
                          matrix_dims: Tuple[int, int, int],
                          dtype: jnp.dtype,
                          n_repeat: int,
                          seed: int) -> Callable[[], None]:
  """Returns an op that applies the `factors` algorithm `n_repeat` times."""
  n = _get_n_from_factors(factors)
  full_a, full_b = _generate_random_matrices(matrix_dims, seed=seed)
  a = block_split(full_a, n, n)
  b = block_split(full_b, n, n)
  a, b = _device_put(a, b, dtype=dtype)

  jitted_algorithm = jax.jit(algorithm_from_factors(factors))

  def _jitted_algorithm_timing() -> None:
    c = b
    for _ in range(n_repeat):
      c = jitted_algorithm(a, c)
    c[0][0].block_until_ready()

  return _jitted_algorithm_timing


def _benchmark_op(op: Callable[[], None], num_trials: int) -> List[float]:
  """Benchmarks `op` `num_trials` times and returns all timings."""
  # Warmup.
  for _ in range(10):
    op()

  gc.disable()  # Prevent garbage collection from interfering with timings.
  timings = []
  for _ in range(num_trials):
    s = timeit.default_timer()
    op()
    e = timeit.default_timer()
    timings.append(e - s)
  gc.enable()
  return timings


def benchmark_jnp_dot(matrix_dims: Tuple[int, int, int],
                      num_trials: int,
                      dtype: jnp.dtype = jnp.float32,
                      average: int = 20,
                      seed: int = 42) -> np.ndarray:
  """Benchmark jnp.dot."""
  baseline_op = _get_baseline_op(matrix_dims, dtype, average, seed)
  timings = _benchmark_op(baseline_op, num_trials)
  return np.array(timings) / average


def benchmark_factorized_algorithm(factors: np.ndarray,
                                   matrix_dims: Tuple[int, int, int],
                                   num_trials: int,
                                   dtype: jnp.dtype = jnp.float32,
                                   average: int = 20,
                                   seed: int = 42) -> np.ndarray:
  """Benchmark a fast-matrix-multiplication algorithm."""
  factorization_algorithm_op = _get_factorization_op(
      factors, matrix_dims, dtype, average, seed)
  timings = _benchmark_op(factorization_algorithm_op, num_trials)
  return np.array(timings) / average
