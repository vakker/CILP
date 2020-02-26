import argparse
import json
import logging
import warnings
from os import path as osp

import ray
import yaml
from hyperopt import hp
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.hyperopt import HyperOptSearch

from cilp import CILP


def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


def train_cilp(params):
    with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore")
        # warnings.warn("deprecated", DeprecationWarning)
        # warnings.warn("user", UserWarning)

        model = CILP(params['data_dir'],
                     params,
                     cached=not params['no_cache'],
                     use_gpu=False,
                     no_logger=True)
        model.initialise()
        # model.train()
        val_acc = model.run_cv()
        tune.track.log(val_acc=val_acc)


def main(args):
    config = {
        'mlp_params': {
            'hidden_sizes': 2,
            'activation': 'ReLU'
        },
        'optim': 'Adam',
        'optim_params': {
            'lr': 0.01,
            'amsgrad': False
        },
        'max_epochs': args.max_epochs,
        'batch_size': 126,
        'data_dir': args.data_dir,
        'no_cache': args.no_cache,
    }

    tune_configs = yaml.safe_load(open(osp.join(args.log_dir, 'tune.yml')))
    space = {}
    for param_subset, params in tune_configs.items():
        space[param_subset] = {}
        for param, options in params.items():
            hp_type = getattr(hp, options['type'])
            space[param_subset][param] = hp_type(param, **options['args'])

    ray.init(memory=2000 * 1024 * 1024,
             object_store_memory=200 * 1024 * 1024,
             driver_object_store_memory=100 * 1024 * 1024)
    algo = HyperOptSearch(space,
                          max_concurrent=10,
                          metric="val_acc",
                          mode="max")

    reporter = CLIReporter()
    reporter.add_metric_column("val_acc")
    analysis = tune.run(train_cilp,
                        num_samples=args.num_samples,
                        config=config,
                        trial_name_creator=trial_str_creator,
                        progress_reporter=reporter,
                        search_alg=algo,
                        stop={"training_iteration": args.max_epochs},
                        local_dir=args.log_dir,
                        max_failures=3,
                        resources_per_trial={'cpu': 4},
                        resume=args.resume,
                        verbose=1)

    print("Best config: ")
    print(json.dumps(analysis.get_best_config(metric="val_acc"), indent=4))
    best_val = analysis.get_best_trial(
        metric="val_acc").metric_analysis['val_acc']['max']
    print(f'Val_acc max: {best_val}')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-v', '--verbose', action='store_true')
    PARSER.add_argument('-r', '--resume', action='store_true')
    PARSER.add_argument('--log-dir')
    PARSER.add_argument('--data-dir')
    PARSER.add_argument('--no-cache', action='store_true')
    PARSER.add_argument('--use-gpu', action='store_true')
    PARSER.add_argument('--num-samples', type=int, default=1)
    PARSER.add_argument('--max-epochs', type=int, default=10)

    ARGS = PARSER.parse_args()

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.WARNING)
    main(ARGS)
