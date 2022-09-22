# benchmarking

Code for the benchmarking results of the AlphaTensor paper.

- `factorizations.py` provides the factorizations found by AlphaTensor when
optimizing for the running time of some particular case of matrix multiplication
(e.g. multiplying 8192x8192 matrices on a V100 GPU in `float32` precision).

- `utils.py` provides functions to convert a factorization into a (fast) matrix
multiplication algorithm.

- `test_correctness.py` provides unit tests that check that the factorizations
do indeed decompose the matrix multiplication tensor, and that the resulting
matrix multiplication algorithm produces the same results as standard matrix
multiplication implementations, up to numerical precision.

- `run_gpu_benchmark.py` benchmarks the GPU-tailored algorithm found in
work against Strassen-square, and against the standard matrix multiplication
algorithm. The benchmark is tested on V100 GPU and requires `sudo` access to fix
the GPU clock frequency (otherwise the benchmarking variance is rather high).

## Installation

To run the benchmark code you would need an NVIDIA V100 GPU. To install JAX and
other required dependencies, you can run:

1. Clone the `alphatensor` repository:

   ```bash
   git clone https://github.com/deepmind/alphatensor.git
   ```

1. *Recommended*: set up a virtual Python environment:

   ```bash
   python3 -m venv benchmarking_env
   source benchmarking_env/bin/activate
   ```
   (To leave the virtual environment, type `deactivate`.)

1. Install the `benchmarking` dependencies:

   ```bash
   pip3 install -r alphatensor/benchmarking/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

## Usage

To run the tests, execute **from the parent directory that contains the
`alphatensor` repository as a subdirectory**

```bash
python3 -m alphatensor.benchmarking.test_correctness
```

To run benchmarking, execute **from the parent directory that contains the
`alphatensor` repository as a subdirectory**

```bash
python3 -m alphatensor.benchmarking.run_gpu_benchmark
```
