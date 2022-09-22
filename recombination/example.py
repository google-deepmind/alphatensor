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

"""Example of using recombination to find rank 255 for <4, 9, 10>."""

from typing import Sequence, Tuple

from absl import app
import numpy as np

from alphatensor.recombination import recombination


def get_3x3x3_factorization() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns a rank-23 factorization of the matrix multiplication tensor T_3."""
  u = np.array([
      [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, -1, 1, 1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, -1, 0, 0],
      [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0,
       -1],
      [0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
  ], dtype=np.int32)
  v = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
      [-1, -1, 0, 0, -1, 0, -1, -1, 1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 1, 0, 0, 1, 0, 1, 1, -1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, -1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
      [-1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, 0, 1, 0, 1, 0, -1, 0, 0],
      [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
  ], dtype=np.int32)
  w = np.array([
      [0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
      [-1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, -1, 1, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
      [0, 0, 0, 1, -1, 0, 1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0, -1, 0, 0, -1, 0,
       1],
      [-1, 1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0],
  ], dtype=np.int32)
  return u, v, w


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  base_factors = get_3x3x3_factorization()
  # The result depends on the order of axes; (4, 9, 10) won't work as well.
  target_mamu = (10, 4, 9)
  # Should return rank 255, which is better than the previously known 259.
  result = recombination.recombine(target_mamu, base_factors)
  print(result)
  assert result['rank'] == 255


if __name__ == '__main__':
  app.run(main)
