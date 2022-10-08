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

The code was tested on an n1-standard-96 Google Cloud VM instance with eight
V100 GPUs, Intel Skylake CPU, using the "TensorFlow Enterprise 2.9 (CUDA 11.3)"
image, and with Jax installed by running the following commands:
```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" \
  -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
"""
import subprocess
import sys
from   argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
# Might be needed on GCP because of the following bug:
# https://github.com/google/jax/issues/9218
import scipy.signal  # pylint: disable=unused-import

from alphatensor.benchmarking import factorizations
from alphatensor.benchmarking import utils

def parse_args(arg_list = sys.argv[1:]):
  parser = ArgumentParser(
    description='Script for benchmarking the AlphaTensor paper.',
    formatter_class=ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
    '--custom-gpu',
    action='store_true',
    help="Use a GPU that is not the V100 used for the original paper."
  )
  parser.add_argument(
    '--gpu-clock',
    type=int,
    default=1530,
    help="Fixed GPU clock to reduce benchmarking variance."
  )
  parser.add_argument(
    '--matrix-sizes',
    choices=['8192', '10240', '12288', '14336', '16384', '18432', '20480'],
    action='append',
    help="Matrix sizes to be benchmarked, default is all of the choices."
  )

  args, unknown = parser.parse_known_args(arg_list)

  if len(unknown) > 0:
    print(f"The following command line arguments were not recognized: {unknown}.")
  return args

def main(args=None):
  if args is None:
        args = parse_args()

  # Reproduce the original results of the AlphaTensor paper on a V100 GPU.
  if not args.custom_gpu:
    # Check if the GPU is a V100.
    process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    if 'V100' not in str(output):
      raise ValueError('To reproduce the results from the paper, please run on a'
                       ' server with V100 GPU.')

    # Make sure that the benchmarking variance is low by fixing clocks.
    print(f'Fixing GPU clock frequency to {args.gpu_clock} to reduce benchmarking variance...')
    process = subprocess.Popen(
        'sudo nvidia-smi -pm ENABLED -i 0'.split(' '), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    process = subprocess.Popen(
        f'sudo nvidia-smi --lock-gpu-clocks={args.gpu_clock},{args.gpu_clock}'.split(' '),
        stdout=subprocess.PIPE)
    output, _ = process.communicate()
    print('Done.')

  # Benchmark the AlphaTensor paper on different GPUs.
  else:
    print('Benchmarking matrix multiplication algorithm discovered by AlphaTensor'
          ' on a custom GPU. Results may vary, since the algorithm was focused on'
          ' the V100 GPU.'
    )

    # Make sure that the benchmarking variance is low by fixing clocks.
    process = subprocess.Popen(
        'sudo nvidia-smi -pm ENABLED -i 0'.split(' '), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    if 'All done' not in str(output):
      print('Failed to enable persistent mode, this would increase benchmarking'
            ' variance.'
      )
    process = subprocess.Popen(
        f'sudo nvidia-smi --lock-gpu-clocks={args.gpu_clock},{args.gpu_clock}'.split(' '),
        stdout=subprocess.PIPE)
    output, _ = process.communicate()
    if 'All done' not in str(output) or 'not supported' in str(output):
      print('Failed to fix clock of the GPU, this would increase benchmarking'
            ' variance.'
      )
    print('Done.')

  num_trials = 10

  if (args.matrix_sizes):
    matrix_sizes = list(map(int, args.matrix_sizes))
  else:
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
      print('%s vs `jnp.dot`: %0.2f%% speedup' % (algorithm_name, improvement))

    print('\n\n')


if __name__ == '__main__':
  main(parse_args())
