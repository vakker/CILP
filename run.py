import argparse
import logging

from cilp import CILP


def main(args):
    params = {
        'mlp_params': {
            'hidden_sizes': [100],
            'activation': 'ReLU'
        },
        'optim': 'Adam',
        'optim_params': {
            'lr': 0.01,
            'amsgrad': False
        },
        'max_epochs': args.max_epochs,
        'batch_size': 126
    }

    model = CILP(args.data_dir,
                 params,
                 cached=not args.no_cache,
                 use_gpu=args.use_gpu)
    model.initialise()
    # model.train()
    model.run_cv()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-v', '--verbose', action='store_true')
    PARSER.add_argument('--log-dir')
    PARSER.add_argument('--data-dir')
    PARSER.add_argument('--no-cache', action='store_true')
    PARSER.add_argument('--use-gpu', action='store_true')
    PARSER.add_argument('--max-epochs', type=int, default=10)

    ARGS = PARSER.parse_args()

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.WARNING)
    main(ARGS)
