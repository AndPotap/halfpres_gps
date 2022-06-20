import argparse
from random import randint
import logging
import numpy as np
from datetime import datetime
from scipy.stats import norm
import torch
from bayesian_benchmarks.data import get_regression_data
from gpytorch import settings
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from mlls.mixedpresmll import HalfEMLL, ReHalfEMLL, WarmHalfEMLL, WarmReHalfEMLL
from models import CholeskyGP, IterativeGP, VariationalGP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cholesky", action="store_true")
    parser.add_argument("--cg_tolerance", default=1.0, type=float)
    parser.add_argument("--max_cg_iterations", default=1000, type=int)
    parser.add_argument("--num_random_probes", default=10, type=int)
    parser.add_argument("--precond_size", default=15, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--dtype", default="float", type=str)
    parser.add_argument("--with_adam", action="store_true")
    parser.add_argument("--ard", action="store_true")
    parser.add_argument("--rel_tol", default=1e-5, type=float)
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--kernel", default="matern", type=str,
                        choices=["matern", "rbf"])
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--maxiter", default=100, type=int)
    parser.add_argument("--database_path", default='.', nargs='?', type=str)
    parser.add_argument("--root_decomp", default=100, type=int)
    parser.add_argument("--mll", default="exact", type=str,
                        choices=["exact", "mixed", "remixed",
                                 "warm_mixed", "warm_remixed", "variational"])
    parser.add_argument("--eval_cg_tolerance", default=0.01, type=float)
    parser.add_argument("--sample_size", default=-1, type=int)
    parser.add_argument("--nu", default=2.5, type=float)
    parser.add_argument("--num_inducing", default=1024, type=int)
    return parser.parse_args()


def prediction_loop(model, data):
    m, v = model.predict(data.X_test)
    res = {}
    llpdf = norm.logpdf(data.Y_test, loc=m, scale=v**0.5)
    res['test_loglik'] = np.average(llpdf)

    lu = norm.logpdf(data.Y_test * data.Y_std, loc=m *
                     data.Y_std, scale=(v**0.5) * data.Y_std)
    res['test_loglik_unnormalized'] = np.average(lu)

    d = data.Y_test - m
    du = d * data.Y_std
    res['test_mae'] = np.average(np.abs(d))
    res['test_mae_unnormalized'] = np.average(np.abs(du))
    res['test_rmse'] = np.average(d**2)**0.5
    res['test_rmse_unnormalized'] = np.average(du**2)**0.5
    res['fitting_time'] = model.fitting_time
    res['prediction_time'] = model.pred_time
    print(res)
    logging.info(res)
    return res


def main(
    dataset,
    split,
    seed,
    device,
    dtype,
    with_adam,
    is_cholesky,
    cg_tolerance,
    precond_size,
    database_path=None,
    maxiter=100,
    max_cg_iterations=1000,
    num_random_probes=10,
    root_decomp=100,
    eval_cg_tolerance=0.01,
    ard=True,
    kernel="matern",
    rel_tol=1e-5,
    mll="exact",
    sample_size=-1,
    nu=2.5,
    num_inducing=1024,
):
    device = torch.device("cuda:"+str(device))
    dtype = torch.float if dtype == "float" else torch.double
    if mll == "exact":
        mll_cls = ExactMarginalLogLikelihood
    elif mll == 'mixed':
        mll_cls = HalfEMLL
    elif mll == 'remixed':
        mll_cls = ReHalfEMLL
    elif mll == 'warm_mixed':
        mll_cls = WarmHalfEMLL
    elif mll == 'warm_remixed':
        mll_cls = WarmReHalfEMLL
    elif mll == "variational":
        mll_cls = VariationalELBO

    model_init = CholeskyGP if is_cholesky else IterativeGP
    if mll == "variational":
        model_init = VariationalGP

    model = model_init(
        device=device,
        dtype=dtype,
        with_adam=with_adam,
        cg_tolerance=cg_tolerance,
        num_random_probes=num_random_probes,
        max_cg_iterations=max_cg_iterations,
        precond_size=precond_size,
        maxiter=maxiter,
        ard=ard,
        kernel=kernel,
        rel_tol=rel_tol,
        mll_cls=mll_cls,
        seed=seed,
        nu=nu,
        inducing_points=num_inducing,
    )

    data = get_regression_data(dataset, split=split)
    xtrain, ytrain = data.X_train[:sample_size], data.Y_train[:sample_size]
    model.fit(xtrain, ytrain)

    with settings.max_root_decomposition_size(root_decomp), \
            settings.eval_cg_tolerance(eval_cg_tolerance), \
            settings.cg_tolerance(eval_cg_tolerance), \
            settings.max_preconditioner_size(precond_size):
        res = prediction_loop(model, data)

    return [res, model.model.state_dict()]


if __name__ == "__main__":
    import pykeops
    import os

    args = parse_args()
    dataset = vars(args)["dataset"]
    print(f'Dataset: {dataset}')
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M_%S")
    string = dataset + '_' + time_stamp + '_' + str(randint(1, int(1.e5)))
    filename = "logs/" + string + ".log"
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    logging.info(f'Dataset: {dataset}')
    for k, v in vars(args).items():
        logging.info(f'Option: {k}: {v}')
    home_path = os.environ["HOME"]
    bin_folder_path = home_path + "/.cache/pykeops-1.5-cpython-39-gpu0/"+string+"/"
    pykeops.set_build_folder(bin_folder_path)

    res_list = main(**vars(args))
    [res.update(args.__dict__) for res in res_list]

    print("Now writing to :", args.database_path)
    torch.save(f=args.database_path+"/"+string+".pt", obj=res_list)
    print("Finished writing")
