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

"""Unit testing for the recombination code."""

import unittest

from absl.testing import absltest

from alphatensor.recombination import example
from alphatensor.recombination import recombination


class RecombinationTest(unittest.TestCase):

  def test_example(self):
    base_factors = example.get_3x3x3_factorization()
    results = recombination.recombine((10, 4, 9), base_factors)
    self.assertEqual(results['rank'], 255)


if __name__ == '__main__':
  absltest.main()
