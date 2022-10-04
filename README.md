# AlphaTensor

This is code accompanying the publication

> Fawzi, A. et al. [Discovering faster matrix multiplication algorithms with
reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4).
*Nature* **610** (2022)

There are 4 independent directories:

- `algorithms` contains algorithms discovered by AlphaTensor, represented as
factorizations of matrix multiplication tensors, and a Colab showing how to load
these.

- `benchmarking` contains a script that can be used to measure the actual speed
of matrix multiplication algorithms on an NVIDIA V100 GPU.

- `nonequivalence` contains 14,236 nonequivalent algorithms discovered by
AlphaTensor for the same matrix multiplication problem (multiplying 4x4
matrices), and a Colab that verifies their nonequivalence.

- `recombination` contains the code we used to decompose larger matrix
multiplication tensors by recombining factorizations of smaller ones.


## Installation

- `algorithms`: No installation required.

- `benchmarking`: See `README` in the subdirectory.

- `nonequivalence`: No installation required.

- `recombination`: A machine with Python 3 installed is required. The required
dependencies (`numpy` and `absl-py`) can be installed by executing
`pip3 install -r alphatensor/recombination/requirements.txt`.

## Usage

- `algorithms`: The notebook `explore_factorizations.ipynb` can be opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/alphatensor/blob/master/algorithms/explore_factorizations.ipynb).
When running the code, you will be asked to upload a file containing the
factorizations. Please select either of the compressed NumPy files
`factorizations_r.npz` (containing algoritms in standard arithmetic) or
`factorizations_f2.npz` (algorithms in arithmetic modulo 2).

- `benchmarking`: See `README` in the subdirectory, and Supplement D of the
paper.

- `nonequivalence`: The notebook `inspect_factorizations_notebook.ipynb` can be
opened via
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/alphatensor/blob/master/nonequivalence/inspect_factorizations_notebook.ipynb).
When running the code, you will be asked to upload a file. Please select the
compressed NumPy file `alphatensor_14236_factorizations.npz`. This will upload
the factorizations found by AlphaTensor, and then compute invariants certifying
that they are all nonequivalent. For more details, see Supplement B of the
paper.

- `recombination`: Execute `python3 -m alphatensor.recombination.example` on the
command line, **from the parent directory that contains the `alphatensor`
repository as a subdirectory**. For more details, see Supplement H of the paper.

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{AlphaTensor2022,
  author  = {Fawzi, Alhussein and Balog, Matej and Huang, Aja and Hubert, Thomas and Romera-Paredes, Bernardino and Barekatain, Mohammadamin and Novikov, Alexander and Ruiz, Francisco J. R. and Schrittwieser, Julian and Swirszcz, Grzegorz and Silver, David and Hassabis, Demis and Kohli, Pushmeet},
  journal = {Nature},
  title   = {Discovering faster matrix multiplication algorithms with reinforcement learning},
  year    = {2022},
  volume  = {610},
  number  = {7930},
  pages   = {47--53},
  doi     = {10.1038/s41586-022-05172-4}
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
