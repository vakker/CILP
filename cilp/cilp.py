import logging
from os import path as osp

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .bcp import run_bcp
from .utils import acc_score, get_features, load_json


class MLP(pl.LightningModule):
    def __init__(self, params, X_train, y_train, X_test, y_test):
        super().__init__()

        self.params = params
        # self.hparams = flatten_dict(params)
        # self.hparams = SimpleNamespace(**flatten_dict(params))
        hidden_sizes = params['mlp_params']['hidden_sizes']
        input_size = params['mlp_params']['input_size']
        activation = params['mlp_params']['activation']

        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        act = getattr(nn, activation)
        layers.append(act())

        for i, s in enumerate(hidden_sizes):
            if i < len(hidden_sizes) - 1:
                output_feats = hidden_sizes[i + 1]
                act = getattr(nn, activation)
            else:
                output_feats = 1
                act = nn.Identity
                # act = getattr(nn, activation)
                # act = nn.Sigmoid
            layers.append(nn.Linear(hidden_sizes[i], output_feats))
            layers.append(act())

        # layers.append(nn.Softmax())
        logging.info(layers)

        self.layers = nn.Sequential(*layers)
        self.loss = nn.BCEWithLogitsLoss()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.best_val = {'val_loss': np.inf, 'val_acc': 0}

    def forward(self, x):
        y = self.layers(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logit = self.forward(x)
        loss = self.loss(y_logit, y)

        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logit = self.forward(x)

        return {'y_logit': y_logit, 'y': y}

    def validation_end(self, outputs):
        y_logit = torch.cat([x['y_logit'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])

        acc = acc_score(y_logit, y, with_logits=True)
        loss = self.loss(y_logit, y)

        log = {'val_loss': loss, 'val_acc': acc}
        self.best_val['val_loss'] = min(self.best_val['val_loss'], loss)
        self.best_val['val_acc'] = max(self.best_val['val_acc'], acc)
        return {'val_loss': loss, 'val_acc': acc, 'log': log}

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.params['optim'])
        return optim(self.parameters(), **self.params['optim_params'])

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.tensor(self.X_train),
                                        torch.tensor(self.y_train)),
                          shuffle=True,
                          batch_size=self.params['batch_size'])

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.tensor(self.X_test),
                                        torch.tensor(self.y_test)),
                          shuffle=False,
                          batch_size=self.params['batch_size'])


class CILP:
    def __init__(self,
                 data_dir,
                 params,
                 cached=True,
                 use_gpu=True,
                 no_logger=False,
                 progress_bar=False):
        self.cached = cached
        self.data_dir = data_dir
        self.params = params
        # self.device = 'cuda:0' if use_gpu else 'cpu'
        self.use_gpu = use_gpu
        self.no_logger = no_logger
        self.progress_bar = progress_bar

        self.X = None
        self.y = None

        self.network = None
        LOGGER = logging.getLogger()
        LOGGER.setLevel(logging.WARNING)

    def bcp(self):
        run_bcp(self.data_dir, cached=self.cached, print_output=False)

    def featurise(self):
        examples_dict = load_json(osp.join(self.data_dir, 'bc.json'))

        logging.info(f"Loaded {len(examples_dict['pos'])} pos examples")
        logging.info(f"Loaded {len(examples_dict['neg'])} neg examples")
        # logging.info(
        #     f"Pos %: ", 100 * len(examples_dict['pos']) /
        #     (len(examples_dict['pos']) + len(examples_dict['neg'])))
        # logging.info(
        #     f"Neg %: ", 100 * len(examples_dict['neg']) /
        #     (len(examples_dict['pos']) + len(examples_dict['neg'])))
        bcp_examples = examples_dict['pos'] + examples_dict['neg']
        labels = np.concatenate([[1] * len(examples_dict['pos']),
                                 [0] * len(examples_dict['neg'])])

        feats_file = osp.join(self.data_dir, 'feats.npz')
        if osp.exists(feats_file) and self.cached:
            logging.info('Loading from cache')
            npzfile = np.load(feats_file)
            examples = npzfile['examples']
            bcp_features = npzfile['bcp_features']
        else:
            examples, bcp_features = get_features(bcp_examples)
            np.savez(feats_file, examples=examples, bcp_features=bcp_features)

        self.bcp_features = bcp_features
        self.X = examples.astype(np.float32)
        # self.y = np.squeeze(labels)
        self.y = np.expand_dims(labels, 1).astype(np.float32)

        logging.info(f'Num examples: {self.X.shape[0]}')
        logging.info(f'Num features: {self.X.shape[1]}')

        self.params['mlp_params'].update({'input_size': self.X.shape[1]})

    def initialise(self):
        self.bcp()
        self.featurise()

        if not self.no_logger:
            logger = TestTubeLogger("tt_logs", name="my_exp_name")
        else:
            logger = False
        # early_stop_callback = EarlyStopping(monitor='val_loss',
        #                                     min_delta=0.00,
        #                                     patience=3,
        #                                     verbose=False,
        #                                     mode='min')
        self.trainer = Trainer(
            logger=logger,
            max_epochs=self.params['max_epochs'],
            early_stop_callback=False,
            check_val_every_n_epoch=1,
            log_save_interval=1,
            row_log_interval=1,
            show_progress_bar=self.progress_bar,
            checkpoint_callback=None,
            gpus=1 if self.use_gpu else 0,
        )

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            test_size=0.2,
                                                            random_state=0)
        self.network = MLP(self.params,
                           X_train=X_train,
                           y_train=y_train,
                           X_test=X_test,
                           y_test=y_test)
        self.trainer.fit(self.network)
        # y_proba = torch.sigmoid(self.network.infer(self.X))
        # print(y_proba)
        # for tqdm

    # def score(self, X, y):
    #     y_proba = torch.sigmoid(self.network.infer(self.X))
    #     return accuracy_score(y, self.predict(X))

    def run_cv(self):
        n_splits = 5
        # cv_split = StratifiedShuffleSplit(n_splits=n_splits,
        #                                   test_size=0.2,
        #                                   random_state=0)
        cv_split = StratifiedKFold(n_splits=n_splits,
                                   random_state=0,
                                   shuffle=True)

        metrics = {'val_acc': [], 'val_loss': []}
        for train_index, test_index in tqdm(cv_split.split(self.X, self.y),
                                            total=n_splits,
                                            disable=True):
            self.network = MLP(self.params,
                               X_train=self.X[train_index],
                               y_train=self.y[train_index],
                               X_test=self.X[test_index],
                               y_test=self.y[test_index])
            self.trainer.fit(self.network)

            best_val = self.network.best_val
            for metric in metrics:
                metrics[metric].append(best_val[metric])

        value = 100 * np.array(metrics[metric])
        return value.mean()

        # for metric in ['val_acc']:
        #     value = 100 * np.array(metrics[metric])
        #     print('#########')
        #     print(metric)
        #     print(value)
        #     print(f"{metric}: {value.mean():.1f} (+/- {value.std() * 2:.1f})")
