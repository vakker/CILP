import time
from collections import defaultdict
from os import path as osp

import numpy as np
import scipy
import torch
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from .bcp import run_bcp
from .trepan import Trepan
from .utils import get_features, load_json, pjoin, save_params, to_numpy


def tng_step(data_loader, model, criterion, optimizer):
    model.train()
    tng_loss = []
    for X, y in data_loader:
        optimizer.zero_grad()

        if model.is_cuda():
            X = X.cuda()
            y = y.cuda()
        y_logit = model(X)
        loss = criterion(y_logit, y)
        loss.backward()
        optimizer.step()

        tng_loss.append(loss.item())

    return {'tng_loss': np.mean(tng_loss)}


@torch.no_grad()
def val_step(data_loader, model, criterion):
    model.eval()
    val_loss = []
    y_pred = []
    y_true = []
    for X, y in data_loader:
        if model.is_cuda():
            X = X.cuda()
            y = y.cuda()
        y_logit = model(X)
        loss = criterion(y_logit, y)
        val_loss.append(loss.item())

        y_pred.append((to_numpy(y_logit) >= 0).astype(int))
        y_true.append(to_numpy(y).astype(int))

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
    recall_mean = scipy.stats.mstats.gmean(recall)
    metrics = {f'val_recall_{i}': v for i, v in enumerate(recall)}
    metrics.update({
        'val_loss': np.mean(val_loss),
        'val_acc': acc,
        'val_recall_gmean': recall_mean
    })
    return metrics


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
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
        # print(layers)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y

    @torch.no_grad()
    def predict(self, x):
        x = torch.FloatTensor(x)
        if self.is_cuda():
            x = x.cuda()
        y_logit = self.forward(x)
        return (to_numpy(y_logit) >= 0).astype(int)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class CILP:
    def __init__(self,
                 data_dir,
                 log_dir,
                 params,
                 n_splits,
                 max_epochs,
                 dedup=False,
                 cached=True,
                 use_gpu=True,
                 no_logger=False,
                 progress_bar=False):
        self.cached = cached
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.params = params
        self.n_splits = n_splits
        self.max_epochs = max_epochs
        self.dedup = dedup
        # self.device = 'cuda:0' if use_gpu else 'cpu'
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar

        self.X = None
        self.y = None

        self.network = None

    def bcp(self):
        run_bcp(self.data_dir, cached=self.cached, print_output=False)

    def featurise(self):
        examples_dict = load_json(pjoin(self.data_dir, 'bc.json'))

        print(f"Loaded {len(examples_dict['pos'])} pos examples")
        print(f"Loaded {len(examples_dict['neg'])} neg examples")

        bcp_examples = examples_dict['pos'] + examples_dict['neg']
        labels = np.concatenate([[1] * len(examples_dict['pos']), [0] * len(examples_dict['neg'])])

        feats_file = pjoin(self.data_dir, 'feats.npz')
        if osp.exists(feats_file) and self.cached:
            print('Loading from cache')
            npzfile = np.load(feats_file)
            examples = npzfile['examples']
            bcp_features = npzfile['bcp_features']
        else:
            examples, bcp_features = get_features(bcp_examples)
            np.savez(feats_file, examples=examples, bcp_features=bcp_features)

        self.bcp_features = bcp_features
        X = examples.astype(np.float32)
        y = np.expand_dims(labels, 1).astype(np.float32)

        print(f'Num examples: {X.shape[0]}')
        print(f'Num features: {X.shape[1]}')

        data = np.concatenate([y, X], axis=1)
        u_data = np.unique(data, axis=0)
        print(f'Unique: {u_data.shape[0]}')

        if self.dedup:
            y = u_data[:, 0:1]
            X = u_data[:, 1:]

            print(f'Num unique examples : {X.shape[0]}')

        self.X = X
        self.y = y

        self.params['mlp_params'].update({'input_size': self.X.shape[1]})

    def init_data(self):
        self.bcp()
        self.featurise()

    def train(self, tng_idx, val_idx, with_trepan=False):
        X_tng = self.X[tng_idx]
        y_tng = self.y[tng_idx]
        X_val = self.X[val_idx]
        y_val = self.y[val_idx]

        tng_dl = DataLoader(TensorDataset(torch.tensor(X_tng), torch.tensor(y_tng)),
                            shuffle=True,
                            batch_size=self.params['batch_size'])

        val_dl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                            shuffle=False,
                            batch_size=self.params['batch_size'])

        network = MLP(self.params)
        if self.use_gpu:
            network.cuda()

        optim = getattr(torch.optim, self.params['optim'])
        optimizer = optim(network.parameters(), **self.params['optim_params'])
        criterion = nn.BCEWithLogitsLoss()

        metrics = defaultdict(list)
        for i in trange(self.max_epochs):
            epoch_metrics = {}
            epoch_metrics.update(tng_step(tng_dl, network, criterion, optimizer))
            epoch_metrics.update(val_step(val_dl, network, criterion))
            for k, v in epoch_metrics.items():
                metrics[k].append(v)

        clf = tree.DecisionTreeClassifier().fit(X_tng, y_tng)

        metrics['dec_tree_tng_acc'] = [clf.score(X_tng, y_tng)]
        metrics['dec_tree_val_acc'] = [clf.score(X_val, y_val)]

        if with_trepan:
            start = time.time()
            mlp_trepan = Trepan(network, maxsize=20)
            mlp_trepan.fit(X_tng, featnames=self.bcp_features)
            print('TREPAN took s', time.time() - start)

            print('MLP Val acc: ', metrics['val_acc'][-1])
            print('Trepan Tng accuracy: ', mlp_trepan.accuracy(X_tng, y_tng))
            print('Trepan Val accuracy: ', mlp_trepan.accuracy(X_val, y_val))
            print('Trepan Tng fidelity:', mlp_trepan.fidelity(X_tng))
            print('Trepan Val fidelity: ', mlp_trepan.fidelity(X_val))

            print('Decision Tree Tng accuracy: ', clf.score(X_tng, y_tng))
            print('Decision Tree Val accuracy: ', clf.score(X_val, y_val))

            dataset_name = osp.basename(self.params['data_dir'])
            mlp_trepan.draw_tree(f'{dataset_name}_trepan.dot')
            tree.export_graphviz(clf,
                                 out_file=f'{dataset_name}_dtc.dot',
                                 feature_names=self.bcp_features,
                                 class_names=['True', 'False'],
                                 impurity=False)

        return metrics

    def run_cv(self):
        cv_split = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=0)
        # cv_split = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)

        split_metrics = defaultdict(list)
        for split_idx, (tng_idx, val_idx) in enumerate(
                tqdm(cv_split.split(self.X, self.y), total=self.n_splits, disable=False)):
            metrics_ = self.train(tng_idx, val_idx)
            for k, v in metrics_.items():
                split_metrics[k].append(v)

        for k in split_metrics:
            split_metrics[k] = np.stack(split_metrics[k])

        params_id = save_params(self.log_dir, self.params)
        np.savez(pjoin(self.log_dir, params_id), **split_metrics)

    def run_trepan(self):
        cv_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        (tng_idx, val_idx) = next(cv_split.split(self.X, self.y))
        _ = self.train(tng_idx, val_idx, with_trepan=True)
