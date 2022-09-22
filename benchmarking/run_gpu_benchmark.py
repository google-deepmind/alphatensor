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

r"""Reproducing GPU speedup reported in the AlphaTensor paper.

Showing GPU speedup of the provably correct fast matrix multiplication
algorithm discovered by AlphaTensor compared to the Strassen^2 baseline.

You should get around 8.5% speedup for multiplying matrices of size 8192 x 8192
in float32 precision on NVIDIA V100 GPU.

This code requires sudo access to set the GPU clock frequency (to reduce
benchmarking variance).

Ideally this code should be run on a server that is not used by others at the
same time to remove interference.

The code is tested on n1-standard-96 Google Cloud VM instance with 8 V100 GPUs
and Intel Skylake CPU with "TensorFlow Enterprise 2.9 (CUDA 11.3)" image with
Jax installed by running the following command:
```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" \
  -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
"""
import subprocess

import numpy as np
# Might be needed on GCP because of the following bug:
# https://github.com/google/jax/issues/9218
import scipy.signal  # pylint: disable=unused-import

from alphatensor.benchmarking import factorizations
from alphatensor.benchmarking import utils


def main():
  process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
  output, _ = process.communicate()
  if 'V100' not in str(output):
    raise ValueError('To reproduce the results from the paper, please run on a'
                     'server with V100 GPU.')
  print('Fixing GPU clock frequency to 1530 to reduce benchmarking variance...')
  process = subprocess.Popen(
      'sudo nvidia-smi -pm ENABLED -i 0'.split(' '), stdout=subprocess.PIPE)
  output, _ = process.communicate()
  process = subprocess.Popen(
      'sudo nvidia-smi --lock-gpu-clocks=1530,1530'.split(' '),
      stdout=subprocess.PIPE)
  output, _ = process.communicate()
  print('Done.')

  num_trials = 10
  matrix_sizes = [8192, 10240, 12288, 14336, 16384, 18432, 20480]

  factorized_algorithms = [
      ('Strassen^2', factorizations.get_4x4x4_strassen_squared()),
      ('AlphaTensor GPU-optimized', factorizations.get_4x4x4_alphatensor_gpu()),
      ('AlphaTensor TPU-optimized', factorizations.get_4x4x4_alphatensor_tpu()),
  ]

  for s in matrix_sizes:
    print(f'Multiplying {s} x {s} matrices')
    print('='*40)
    results_dot = utils.benchmark_jnp_dot((s, s, s), num_trials=num_trials)

    for algorithm_name, factorization in factorized_algorithms:
      if algorithm_name == 'AlphaTensor TPU-optimized' and s > 19000:
        continue  # This TPU-optimized algorithm runs OOM on a V100 GPU.
      results_algorithm = utils.benchmark_factorized_algorithm(
          factorization, (s, s, s), num_trials=num_trials)
      ratio = np.median(results_dot / results_algorithm)
      improvement = 100 * ratio - 100
      print('%s vs `jnp.dot`: %0.2f%% speedup' % algorithm_name, improvement)

    print('\n\n')


if __name__ == '__main__':
  main()
