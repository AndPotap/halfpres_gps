import numpy as np
from jax import numpy as jnp
from datetime import datetime
from random import randint
import logging
import math
import pickle


def analyze_while_loop(cond, body_fun, init_val):
    flag, state = True, init_val
    while flag:
        state = body_fun(state)
        flag = cond(state)
    return state


def transform_params_to_raw(params):
    out = [jnp.log(jnp.exp(p) - 1.) for p in params]
    return out


def determine_checkpoints():
    use_checkpoints = input('Use checkpoints? (Y/N): ')
    use_checkpoints = True if use_checkpoints == 'Y' else False
    checkpoint_freq = None
    if use_checkpoints:
        checkpoint_freq = input('Frequency: ')
        checkpoint_freq = int(checkpoint_freq) if checkpoint_freq.isdigit() else 1
    return use_checkpoints, checkpoint_freq


def save_checkpoint(params, idx, logger):
    file_path = logger.log_file_name[:-4] + f'_{idx}' + '.pkl'
    save_results(params, output_file=file_path)
    return params


def load_results(input_file):
    with open(input_file, mode='rb') as f:
        results = pickle.load(f)
    return results


def save_results(results, output_file):
    with open(file=output_file, mode='wb') as f:
        pickle.dump(obj=results, file=f)


def start_all_logging_instruments(settings, results_path, name_append=''):
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    log_file_name = results_path + settings['dataset_name'] + name_append
    log_file_name += '_' + time_stamp
    log_file_name += '_' + str(randint(1, 100000)) + '.log'
    logger_name = 'log_' + time_stamp + '_' + str(randint(1, 100000))
    logger = setup_logger(log_file_name, logger_name)
    logger.log_file_name = log_file_name
    return logger


def log_all_settings(settings, logger):
    for key, value in settings.items():
        logger.info(f'Hyper: {key}: {value}')


def setup_logger(log_file_name, logger_name: str = None):
    if logger_name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:    %(message)s')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(filename=log_file_name)
    file_handler.setFormatter(fmt=formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=stream_formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger


def break_if_problem(v):
    flag = has_problem(v)
    if flag:
        breakpoint()


def has_problem(v):
    flag = has_nan(v) or has_inf(v)
    return flag


def has_nan(v):
    flag = jnp.any(jnp.isnan(v))
    return flag


def has_inf(v):
    flag = jnp.any(jnp.isinf(v))
    return flag


def print_conditioning(A):
    print(jnp.linalg.cond(A.astype(jnp.float32)))


def save_ker_and_probes(ker_w_noise, probes):
    np.save(file='ker.npy', arr=ker_w_noise)
    np.save(file='probes.npy', arr=probes)


def print_time_taken(delta, text='Experiment took: ', logger=None):
    minutes = math.floor(delta / 60)
    seconds = delta - minutes * 60
    message = text + f'{minutes:4d} min and {seconds:4.2f} sec'
    if logger is not None:
        logger.info(message)
    else:
        print(message)
