import torch
import time
import gpytorch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.approximate_gp import _select_inducing_points
# from botorch.optim.fit import fit_gpytorch_torch, fit_gpytorch_scipy
from botorch.optim.fit import fit_gpytorch_torch
# from botorch import fit_gpytorch_model
from gpytorch.means import ZeroMean
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.settings import (
    deterministic_probes,
    cg_tolerance,
    max_cg_iterations,
    max_preconditioner_size,
    max_cholesky_size,
    num_trace_samples,
    cholesky_jitter,
)

from optim import fit_gpytorch_minibatch

try:
    import sys
    sys.path.append("../../../gp-bfgs/lbfgs/PyTorch-LBFGS/functions")
    from LBFGS import FullBatchLBFGS
except ImportError:
    print("warning: no lbfgs")


def _wang_like_lbfgs(mll, maxiter=10):
    full_X = mll.model.train_inputs[0]
    full_Y = mll.model.train_targets

    # shuffled so this should be okay
    mll.model.set_train_data(
        full_X[:10000],
        full_Y[:10000],
        strict=False
    )

    optimizer = FullBatchLBFGS(mll.model.parameters())

    # with deterministic_probes(True):
    # fit_gpytorch_scipy(mll, options={"maxiter": 10})

    def closure():
        optimizer.zero_grad()
        output = mll.model(*mll.model.train_inputs)
        with cholesky_jitter(1e-3):
            loss = -mll(output, mll.model.train_targets)
        return loss

    loss = closure()
    loss.backward()

    for i in range(10):
        # perform step and update curvature
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

    # now reset
    mll.model.set_train_data(full_X, full_Y, strict=False)

    with deterministic_probes(False):
        return fit_gpytorch_torch(mll, options={"maxiter": maxiter, "lr": 0.1})


class DefaultGPyTorchModel(object):
    def __init__(
        self,
        device,
        dtype,
        with_adam,
        seed,
        ard,
        kernel,
        mll_cls,
        nu,
        **kwargs,
    ):
        self.model = None
        self.with_adam = with_adam
        self.ard = ard
        self.nu = nu
        maxiter = kwargs.pop("maxiter", 2000)
        rel_tol = kwargs.pop("rel_tol", 1e-5)
        print(maxiter, rel_tol)
        # self.optimizer = lambda mll: fit_gpytorch_torch(
        #             mll, options={"maxiter": maxiter, "lr": 0.1}
        #     )
        if with_adam:
            self.optimizer = lambda mll: fit_gpytorch_torch(
                mll, options={"maxiter": maxiter, "lr": 0.1, "rel_tol": rel_tol}
            )
        else:
            self.optimizer = lambda mll: _wang_like_lbfgs(mll, maxiter=maxiter)

        # if with_adam else fit_gpytorch_model
        self.device = device
        self.dtype = dtype

        torch.random.manual_seed(seed)

        if kernel == "matern":
            self.kernel_init = gpytorch.kernels.keops.MaternKernel
        elif kernel == "rbf":
            self.kernel_init = gpytorch.kernels.keops.RBFKernel

        self.mll_cls = mll_cls

    def fit(self, X, Y):
        # botorch defaults but with a sp transform on noise
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            # batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                1e-4,
                # transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        self.model = SingleTaskGP(
            torch.tensor(X, device=self.device, dtype=self.dtype),
            torch.tensor(Y, device=self.device, dtype=self.dtype).view(-1, 1),
            covar_module=gpytorch.kernels.ScaleKernel(
                self.kernel_init(
                    nu=self.nu,
                    ard_num_dims=1 if self.ard is False else X.shape[-1],
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            ),
            likelihood=likelihood
        )
        self.model.mean_module = ZeroMean()

        mll = self.mll_cls(self.model.likelihood, self.model)
        start = time.time()
        self.opt_output = self.optimizer(mll)
        end = time.time()
        print("Model fitting time: ", end - start)
        self.fitting_time = end - start

    def predict(self, Xs):
        # nuke caches so that we can use the same model for different root decomp sizes
        self.model.train()

        # now set in posterior mode
        self.model.eval()
        self.model.likelihood.eval()
        start = time.time()

        posterior = self.model.posterior(torch.tensor(
            Xs, device=self.device, dtype=self.dtype), observation_noise=True)
        m = posterior.mean.detach().cpu().numpy()
        v = posterior.variance.detach().cpu().numpy()
        end = time.time()
        print("Model prediction time: ", end - start)
        self.pred_time = end - start

        return m, v


class CholeskyGP(DefaultGPyTorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_init = gpytorch.kernels.MaternKernel

    def fit(self, X, Y):
        with max_cholesky_size(int(1.e7)):
            super().fit(X, Y)

    def predict(self, Xs):
        with max_cholesky_size(int(1.e7)):
            return super().predict(Xs)


class IterativeGP(DefaultGPyTorchModel):
    def __init__(self, device, dtype, max_cg_iterations,
                 num_random_probes, with_adam,
                 seed, cg_tolerance, precond_size, **kwargs):
        super().__init__(device=device, dtype=dtype, with_adam=with_adam, seed=seed, **kwargs)
        self.cg_tolerance = cg_tolerance
        self.max_cg_iterations = max_cg_iterations
        self.num_random_probes = num_random_probes
        self.precond_size = precond_size

    def fit(self, X, Y):
        with deterministic_probes(not self.with_adam), \
                cg_tolerance(self.cg_tolerance), \
                num_trace_samples(self.num_random_probes), \
                max_cg_iterations(self.max_cg_iterations), \
                max_preconditioner_size(self.precond_size):
            super().fit(X, Y)

    def predict(self, Xs):
        with cg_tolerance(self.cg_tolerance), max_preconditioner_size(self.precond_size):
            return super().predict(Xs)

class VariationalGP(DefaultGPyTorchModel):
    def __init__(self, device, dtype, inducing_points, with_adam,
                 seed, cg_tolerance, precond_size, **kwargs):
        super().__init__(
            device=device, dtype=dtype, with_adam=with_adam, seed=seed, **kwargs
        )
        kernel = kwargs.get("kernel", "matern")
        # we don't want to be using keops here
        if kernel == "matern":
            self.kernel_init = gpytorch.kernels.MaternKernel
        elif kernel == "rbf":
            self.kernel_init = gpytorch.kernels.RBFKernel

        self.inducing_points = inducing_points

        maxiter = kwargs.pop("maxiter", 2000)
        rel_tol = kwargs.pop("rel_tol", 1e-5)
        self.optimizer = lambda mll: fit_gpytorch_minibatch(
                mll, options={"maxiter": maxiter, "lr": 0.01, "rel_tol": rel_tol, "batch_size": inducing_points}
        )

    def fit(self, X, Y):
        # botorch defaults but with a sp transform on noise
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            # batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                1e-4,
                # transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        kernel = gpytorch.kernels.ScaleKernel(
            self.kernel_init(
                nu=self.nu,
                ard_num_dims=1 if self.ard is False else X.shape[-1],
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        ).to(self.device, self.dtype)
        tsr_x = torch.tensor(X, device=self.device, dtype=self.dtype)
        inducing_pts = _select_inducing_points(
            inputs=tsr_x[:10000], covar_module=kernel, num_inducing=self.inducing_points,
            input_batch_shape=torch.Size(),
        )
        self.model = SingleTaskVariationalGP(
            tsr_x,
            torch.tensor(Y, device=self.device, dtype=self.dtype).view(-1, 1),
            covar_module=kernel,
            inducing_points=inducing_pts,
            likelihood=likelihood,
            mean_module=ZeroMean().to(self.device, self.dtype)
        )
        self.model = self.model.to(self.device, self.dtype)

        mll = self.mll_cls(self.model.likelihood, self.model.model, num_data=X.shape[-2])
        start = time.time()
        self.opt_output = self.optimizer(mll)
        end = time.time()
        print("Model fitting time: ", end - start)
        self.fitting_time = end - start
