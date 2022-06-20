# Low Precision Arithmetic for Fast Gaussian Processes

<br>**Abstract**<br>
*Low precision arithmetic has had a transformative effect on the training of neural
networks, reducing computation, memory and energy requirements. However, despite its
promise, low precision arithmetic has received little attention for Gaussian process (GP)
training, largely because GPs require sophisticated linear algebra routines that are
unstable in low precision. We study the different failure modes that can occur when
training GPs in half precision. To circumvent these failure modes, we propose a
multi-faceted approach involving conjugate gradients with re-orthogonalization, compact
kernels, and preconditioners. Our approach significantly improves the numerical stability
and practical performance of conjugate gradients in low precision over a wide range of
settings, enabling GPs to train on 1.8 million data points in 10 hours on a single
GPU, without requiring any sparse approximations*

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
