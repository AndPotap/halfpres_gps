# Low Precision Arithmetic for Fast Gaussian Processes

This repository contains PyTorch code for for the paper

[Low Precision Arithmetic for Fast Gaussian Processes](https://openreview.net/pdf?id=S3NOX_Ij9xc)

by Wesley J. Maddox\*, Andres Potapczynski\*, and Andrew Gordon Wilson. 

## Introduction

In this paper, we show how to speed up Gaussian processes by representing numbers in lower precision, while retaining accuracy. These methods involve a modification to conjugate gradients with re-orthogonalization, compact kernels, pre-conditioners, and mixed-precision representations. In many cases, these approaches can be used as a drop-in replacement for standard Gaussian process routines, or combined with other scalable inference approaches. In short, you should try this out! 

Please cite our work if you find it useful:
```bibtex
@inproceedings{gplowprec2022,
  title={Low precision arithmetic for fast Gaussian processes},
  author={Maddox, Wesley J and Potapczynski, Andres and Wilson, Andrew Gordon},
  booktitle={Uncertainty in Artificial Intelligence},
  year={2022}
}
```

## Installation instructions
* For Bayesian Benchmarks do not use the setup.py file. Instead (1) git clone the repo, (2) add the repo to the PYTHONPATH (needed for properly loading modules on the repo)
* Create a `logs` directory.

## Experiments
To replicate the experimental results you can run

```shell
python3 experiments/runner.py \
    --dataset=wilson_pol \
    --mll=mixed \
    --sample_size=-1 \
    --max_cg_iterations=50 \
    --num_random_probes=10 \
    --cg_tolerance=1.0 \
    --eval_cg_tolerance=0.01 \
    --precond_size=15 \
    --device=0 \
    --split=73 \
    --kernel=matern \
    --nu=0.5 \
    --ard \
    --rel_tol=-1000 \
    --seed=0 \
    --maxiter=50 \
    --database_path=. \
    --root_decomp=100 \
    --with_adam
```

## Variables / Arguments Explanation
| Name | Description |
| :------------ |  :-----------: |
| `dataset` | Specifies dataset to use. |
| `mll` | Selects the marginal likelihood model. |
| `sample_size` |  Number of observations to use (set to -1 for full dataset). |
| `max_cg_iterations` |  Maximum number of CG iterations. |
| `num_random_probes` |  Number of random probes to estimate the loss. |
| `cg_tolerance` |  Tolerance that CG must reach during training. |
| `eval_cg_tolerance` |  Tolerance that CG must reach during testing. |
| `precond_size` |  Preconditioner rank. |
| `device` |  Which GPU device to use (set to 0 if default). |
| `split` |  Set random seed to use (set an int). |
| `kernel` | Select that type of kernel to use. |
| `nu` |  If using Matern, determine if 0.5, 1.5 or 2.5. |
| `ard` |  Flag to use ARD. |
| `rel_tol` | Relative tolerance to stop the optimization (set to -1000 to ignore). |
| `maxiter` | Maximum number of optimization steps. |
| `database_path` |  Path to save results. |
| `with_adam` |  Flag for using ADAM as an optimizer. |

## Notebooks
In `notebooks` you can also find an example of how to run the CG solvers.
