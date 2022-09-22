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

"""Test correctness of the presented fast matrix multiplication algorithms."""

import unittest

from absl.testing import absltest
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np

from alphatensor.benchmarking import factorizations
from alphatensor.benchmarking import utils


class MatrixMultiplicationCorrectnessTest(unittest.TestCase):
  """Tests of matrix multiplication correctness."""

  def testTensorDecompositionGPU(self):
    """Tests that the factors decompose the matrix multiplication tensor."""
    u, v, w = factorizations.get_4x4x4_alphatensor_gpu()
    reconstructed_tensor = np.einsum('ir,jr,kr->ijk', u, v, w)
    expected_tensor = utils.get_matrix_multiplication_tensor(4)
    np.testing.assert_array_equal(reconstructed_tensor, expected_tensor)

  def testTensorDecompositionTPU(self):
    """Tests that the factors decompose the matrix multiplication tensor."""
    u, v, w = factorizations.get_4x4x4_alphatensor_tpu()
    reconstructed_tensor = np.einsum('ir,jr,kr->ijk', u, v, w)
    expected_tensor = utils.get_matrix_multiplication_tensor(4)
    np.testing.assert_array_equal(reconstructed_tensor, expected_tensor)

  def testGPUMatrixMultiplicationPrecision(self):
    """Compare direct multiplication A @ B against using the proposed algorithm.

    Compare the result of multiplying two matrices via jnp.dot vs via the
    proposed fast algorithm (up to some precision).
    """
    config.update('jax_enable_x64', True)
    factors = factorizations.get_4x4x4_alphatensor_gpu()
    matrix_multiplication_algorithm = utils.algorithm_from_factors(factors)

    # Generate the matrices.
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(42))
    full_a = jax.random.uniform(rng1, (1024, 1024), dtype=jnp.float64)
    full_b = jax.random.uniform(rng2, (1024, 1024), dtype=jnp.float64)

    a = utils.block_split(full_a, 4, 4)
    b = utils.block_split(full_b, 4, 4)
    actual_result = matrix_multiplication_algorithm(a, b)
    actual_result = np.bmat(actual_result)
    desired_result = full_a @ full_b
    np.testing.assert_allclose(actual_result, desired_result)


if __name__ == '__main__':
  absltest.main()
