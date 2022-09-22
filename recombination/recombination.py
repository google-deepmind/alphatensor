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

"""Implement (a generalized version of) the idea from papers [1, 2].

[1] Alexandre Sedoglavic, "A non-commutative algorithm for multiplying (7 × 7)
matrices using 250 multiplications" (2017).

[2] Drevet, Charles-Éric, Md Nazrul Islam, and Éric Schost. "Optimization
techniques for small matrix multiplication." Theoretical Computer Science 412.22
(2011): 2219-2236.
"""

from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from alphatensor.recombination import sota


def _tensor_shape_to_matrix_sizes(
    tensor_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
  """Returns the sizes of the multiplied matrices from the matmul tensor shape.

  When multiplying an [a, b] and [b, c] matrix, the size of the corresponding
  matrix multiplication tensor T_{a, b, c} is [ab, bc, ca]. This function
  computes the inverse mapping from the tensor size to the matrix sizes.

  Args:
    tensor_shape: Shape of a 3D matrix multiplication tensor T_{a, b, c}.

  Returns:
    The three integers a, b, c describing the matrix sizes being multiplied.
  """
  ab, bc, ca = tensor_shape
  a = int(np.sqrt(ab * ca // bc))
  b = int(np.sqrt(ab * bc // ca))
  c = int(np.sqrt(bc * ca // ab))
  assert a * b == ab and b * c == bc and c * a == ca
  return a, b, c


def _factorization_2d_to_3d(
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Converts factorization with `u` of shape [a*b, rank] into [a, b, rank]."""
  u, v, w = factors
  a, b, c = _tensor_shape_to_matrix_sizes((u.shape[0], v.shape[0], w.shape[0]))
  rank = u.shape[-1]
  u = u.reshape(a, b, rank)
  v = v.reshape(b, c, rank)
  w = w.reshape(c, a, rank)
  return u, v, w


def _block_fillings(num_blocks: int, budget: int) -> Iterator[List[int]]:
  """Iterates over all options of filling `num_blocks` with `budget` balls."""
  if num_blocks == 1:
    yield [budget]
    return
  for i in range(budget + 1):
    for rest in _block_fillings(num_blocks - 1, budget - i):
      yield [i] + rest


def _process_additions(matricized_factor_vector: np.ndarray,
                       row_nonzeros: Sequence[int],
                       col_nonzeros: Sequence[int]) -> Tuple[int, int]:
  """Returns the nonzero matrix size after adding multiple matrices together.

  Nonzero elements of a factor vector stipulate that the corresponding entries
  of the base matrix (which in this case are themselves matrices) are to be
  added up. The number of potentially nonzero rows in this sum is the maximum
  over the number of nonzero rows in each summand, and similarly for the number
  of columns. See Supplementary Information of the paper for an illustrative
  example.

  Args:
    matricized_factor_vector: [x, y]-shaped array representing a single factor
      vector (`u`, or `v`, or `w`) in matrix form. For example, [x, y] = [a, b]
      when this is a `u` vector.
    row_nonzeros: List of length x, with the i-th entry specifying the number of
      rows of the target matrix that were allocated to the i-th row of the base
      matrix on the first level of recursion.
    col_nonzeros: List of length y, with the i-th entry specifying the number of
      columns of the target matrix that were allocated to the i-th column of the
      base matrix on the first level of recursion.

  Returns:
    Two integers describing respectively the largest number of nonzero rows and
    columns after the submatrices corresponding to nonzero entries of the factor
    vector `matricized_factor_vector` are added up.
  """
  max_rows = 0
  max_cols = 0
  for i, j in zip(*np.nonzero(matricized_factor_vector)):
    max_rows = max(max_rows, row_nonzeros[i])
    max_cols = max(max_cols, col_nonzeros[j])
  return max_rows, max_cols


def recombine(target_matrix_sizes: Tuple[int, int, int],
              base_factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
             ) -> Dict[str, Any]:
  """Decomposes T_{a, b, c} using `base_factors` as the 1st level of recursion.

  See Supplementary Information of the paper for more details.

  Args:
    target_matrix_sizes: Triplet (a, b, c) specifing the matrix multiplication
      problem of multiplying an [a, b] matrix by a [b, c] matrix. Equivalently,
      specifies a matrix multiplication tensor T_{a, b, c} to be decomposed.
    base_factors: Three arrays providing a factorization of a (usually smaller)
      matrix multiplication tensor T_{base_a, base_b, base_c}. This algorithm
      will be used on the first level of recursion when decomposing T_{a, b, c}.

  Returns:
    Dictionary with information about the best rank discovered for T_{a, b, c}.
  """
  base_rank = base_factors[0].shape[-1]
  base_tensor_shape = tuple(v.shape[0] for v in base_factors)
  base_a, base_b, base_c = _tensor_shape_to_matrix_sizes(base_tensor_shape)

  u, v, w = _factorization_2d_to_3d(base_factors)
  # The matrix multiplication tensor T_{a, b, c} by convention represents the
  # operation (A, B) -> (AB)^T, i.e. with an additional transposition. Here we
  # will work with the non-transposed version for simplicity.
  w = w.transpose(1, 0, 2)

  best = {}

  # To apply an algorithm for (base_a, base_b, base_c) to the target problem
  # (target_a, target_b, target_c), we try all possibilities of how to allocate
  # the `target_a` rows of the original problem to the `base_a` rows of the
  # algorithm to be applied on the first level of recursion; and similarly for
  # the `target_b` and `target_c` dimensions.
  target_a, target_b, target_c = target_matrix_sizes
  for allocation_a in _block_fillings(base_a, target_a):
    for allocation_b in _block_fillings(base_b, target_b):
      for allocation_c in _block_fillings(base_c, target_c):
        total_rank = 0
        small_matrix_sizes = []
        for r in range(base_rank):
          u1, u2 = _process_additions(u[:, :, r], allocation_a, allocation_b)
          v1, v2 = _process_additions(v[:, :, r], allocation_b, allocation_c)
          w1, w2 = _process_additions(w[:, :, r], allocation_a, allocation_c)

          # We now need to compute the product of [u1, u2] and [v1, v2]-shaped
          # matrices (with appropriate zero-padding), and then extract the
          # [w1, w2] upper-left portion of the resulting product. Note that this
          # can be achieved by an algorithm that multiplies matrices of sizes
          # [min(u1, w1), min(u2, v1)] and [min(u2, v1), min(v2, w2)] since it
          # is not necessary to compute elements that will end up zero/unused.
          current_matrix_sizes = min(u1, w1), min(u2, v1), min(v2, w2)
          total_rank += sota.get_sota_rank(*current_matrix_sizes)
          small_matrix_sizes.append(current_matrix_sizes)

        if not best or total_rank < best['rank']:
          best = {
              'rank': total_rank,
              'small_matrix_sizes': small_matrix_sizes,
              'allocation_pattern': (allocation_a, allocation_b, allocation_c),
          }

  return best
