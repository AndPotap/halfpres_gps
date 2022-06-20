import torch
from gpytorch.utils.deprecation import bool_compat


def _default_preconditioner(x):
    return x.clone()


class GPyTorchCGSolverRe:

    def __init__(self, tolerance=1.e-1, max_iters=20, preconditioner=None):
        self.tolerance = tolerance
        self.max_iters = max_iters
        if preconditioner is None:
            self.preconditioner = _default_preconditioner
        else:
            self.preconditioner = preconditioner
        self.stop_updating_after = 1.e-10
        self.eps = 1.e-10

    def set_matrix_and_probes(self, A_fn, b):
        self.A = A_fn
        self.b = b
        self.x0 = torch.zeros_like(b)

    def run_mbcg_with_tracking(self):
        rhs_norm = self.b.norm(2, dim=-2, keepdim=True)
        rhs_is_zero = rhs_norm.lt(self.eps)
        rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
        rhs = self.b.div(rhs_norm)

        state, out = initialize_cg(
            self.A, rhs, self.stop_updating_after, self.eps, self.preconditioner, self.max_iters)
        x0, has_converged, r0, batch_shape, residual_norm = state
        (p0, gamma0, mul_storage, beta, alpha, is_zero, z0, u_all) = out
        self.initialize_trackers()
        self.update_trackers(x0, r0, gamma0, p0, k=0)
        for k in range(self.max_iters):
            Ap0 = self.A(p0)
            take_cg_step(
                Ap0, x0, r0, gamma0, p0, alpha, beta,
                z0, mul_storage, has_converged, self.eps, is_zero, u_all, k)
            if cond_fn(k, self.max_iters, self.tolerance,
                       r0, has_converged, residual_norm, self.stop_updating_after,
                       rhs_is_zero):
                break

            print_analysis(k, alpha, residual_norm, gamma0, beta)
            self.update_trackers(x0, r0, gamma0, p0, k)
        return x0.mul(rhs_norm)

    def update_trackers(self, x0, r0, gamma0, p0, k):
        self.Us.append(x0.clone())
        self.Rs.append(r0.clone())
        self.gammas.append(gamma0.clone())
        self.ps.append(p0.clone())
        self.k = k

    def initialize_trackers(self):
        self.Us, self.Rs, self.gammas, self.ps, self.k = [], [], [], [], -1


def linear_cg_re(
    matmul_closure,
    rhs,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
    initial_guess=None,
    preconditioner=None,
):
    if preconditioner is None:
        preconditioner = _default_preconditioner

    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    state, out = initialize_cg(
        matmul_closure, rhs, stop_updating_after, eps, preconditioner, max_iter)
    x0, has_converged, r0, batch_shape, residual_norm = state
    (p0, gamma0, mul_storage, beta, alpha, is_zero, z0, u_all) = out

    for k in range(max_iter):
        Ap0 = matmul_closure(p0)
        take_cg_step(
            Ap0=Ap0,
            x0=x0,
            r0=r0,
            gamma0=gamma0,
            p0=p0,
            alpha=alpha,
            beta=beta,
            z0=z0,
            mul_storage=mul_storage,
            has_converged=has_converged,
            eps=eps,
            is_zero=is_zero,
            u_all=u_all,
            k=k,
        )

        if cond_fn(k, max_iter, tolerance, r0, has_converged, residual_norm,
                   stop_updating_after, rhs_is_zero):
            break

    x0 = x0.mul(rhs_norm)
    return x0


def initialize_cg(matmul_closure, rhs, stop_updating_after, eps, preconditioner, max_iter):
    initial_guess = torch.zeros_like(rhs)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    result = initial_guess.expand_as(residual).contiguous()

    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    state = (result, has_converged, residual, batch_shape, residual_norm)
    out = create_placeholders(rhs, residual, preconditioner, batch_shape, max_iter)
    return state, out


def create_placeholders(rhs, residual, preconditioner, batch_shape, max_iter):
    precond_residual = preconditioner(residual)
    curr_conjugate_vec = precond_residual
    residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

    mul_storage = torch.empty_like(residual)
    alpha = torch.empty(*batch_shape, 1, rhs.size(-1),
                        dtype=residual.dtype, device=residual.device)
    beta = torch.empty_like(alpha)
    u_all = torch.zeros(size=(max_iter,) + rhs.shape,
                        dtype=rhs.dtype, device=rhs.device)
    is_zero = torch.empty(*batch_shape, 1, rhs.size(-1),
                          dtype=bool_compat, device=residual.device)
    return (curr_conjugate_vec, residual_inner_prod, mul_storage, beta, alpha, is_zero,
            precond_residual, u_all)


def take_cg_step(
        Ap0, x0, r0, gamma0, p0, alpha, beta, z0, mul_storage, has_converged, eps,
        is_zero, u_all, k):

    torch.mul(p0, Ap0, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(gamma0, alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)
    alpha.masked_fill_(has_converged, 0)

    torch.addcmul(r0, -alpha, Ap0, out=r0)
    torch.addcmul(x0, alpha, p0, out=x0)

    for i in range(k - 1):
        dotprod = torch.sum(r0 * u_all[i], dim=-2) * u_all[i]
        r0 = r0 - dotprod

    precond_residual = r0.clone()
    beta.resize_as_(gamma0).copy_(gamma0)
    torch.mul(r0, precond_residual, out=mul_storage)
    torch.sum(mul_storage, -2, keepdim=True, out=gamma0)
    torch.lt(beta, eps, out=is_zero)
    beta.masked_fill_(is_zero, 1)
    torch.div(gamma0, beta, out=beta)
    beta.masked_fill_(is_zero, 0)
    u_all[k] = (r0 / torch.sqrt(gamma0)).clone()

    p0.mul_(beta).add_(precond_residual)


def cond_fn(k, max_iter, tolerance, residual, has_converged, residual_norm,
            stop_updating_after, rhs_is_zero):
    torch.norm(residual, 2, dim=-2, keepdim=True, out=residual_norm)
    residual_norm.masked_fill_(rhs_is_zero, 0)
    torch.lt(residual_norm, stop_updating_after, out=has_converged)
    flag = k >= min(10, max_iter - 1) and bool(residual_norm.mean() < tolerance)
    return flag


def print_analysis(k, alpha, residual_norm, gamma0, beta):
    print('\n===================================================')
    print(f'Iter {k}')
    print('alpha')
    print(alpha)
    print(f'Alpha mean: {torch.mean(alpha)}')
    print(residual_norm)
    print(f'Residual norm mean: {torch.mean(residual_norm):1.3e}')
    print('gamma')
    print(f'Gamma mean: {torch.mean(gamma0)}')
    print(gamma0)
    print('beta')
    print(f'Beta mean: {torch.mean(beta)}')
    print(beta)
