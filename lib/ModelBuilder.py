import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_matching_item_in_list(array, idx):
    try:
        return array[idx]
    except IndexError:
        return None


class Builder(nn.Module):
    __slots__ = ('layer_type', 'activations_type', 'layers', 'activations', 'output')

    def __init__(self, layers, activations=None):
        super(Builder, self).__init__()
        self.layer_type = self._validate_init(layers)
        # self.activations_type = self._validate_init(activations)

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations) if activations else []
        self.output = None

    @staticmethod
    def _validate_init(class_input):
        if not isinstance(class_input, list) and not isinstance(class_input, dict):
            raise Exception(
                'You must provide your layers and activations as a list or dict of nn.Modules, you provided: {}'.format(
                    type(class_input)))
        return type(class_input).__name__

    def forward(self, x):
        if self.layer_type == 'dict':
            # for name, layer in self.layers.items():
            pass  # todo: implement this in an elegant way to connect layer to activation
        if self.layer_type == 'list':
            for idx, layer in enumerate(self.layers):
                activation = _get_matching_item_in_list(self.activations, idx)
                x = activation(layer(x)) if activation else layer(x)
        self.output = x
        return self.output


class ModelRunner:
    __slots__ = (
        'outputs', 'losses', 'gradients', 'performance_index', 'optimizer',
        'dtype', 'device', 'data_size', 'batch_size', 'model', '_start',
        'is_image', 'dimensions', 'pred_output', 'collect_grad', 'results', 'stop_early_at','adjustment','drop'
    )

    def __init__(self,
                 performance_index=None,
                 optimizer=None,
                 model=None,
                 dtype=None,
                 device=None,
                 data_size=0,
                 batch_size=0,
                 is_image=False,
                 dimensions=None,
                 collect_grad=False,
                 stop_early_at=None,
                 ):
        if performance_index:
            self._validate_init(performance_index)
        if optimizer:
            self._validate_init(optimizer)
        self.performance_index = performance_index
        self.optimizer = optimizer
        self.dtype = dtype or torch.float
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.data_size = data_size
        self.batch_size = batch_size
        self.is_image = is_image
        self.dimensions = dimensions
        self.outputs = []
        self.losses = []
        self.gradients = []
        self.pred_output = []
        self.results = []
        self.collect_grad = collect_grad
        self.stop_early_at = stop_early_at or 0

    @staticmethod
    def _validate_init(class_input):
        class_name = getattr(class_input, '__module__', None)
        if not 'torch.optim' in class_name and not 'torch.nn.modules' in class_name:
            raise Exception(
                'You must provide torch.nn modules for performance_index and optimizer, you provided: {}'.format(
                    type(class_input)))

    def _to_device(self, *args):
        on_device = []
        for a in args:
            on_device.append(a.to(self.device))
        return on_device

    def generate_simple_data(self, batch_size=0, input_size=0, output_size=0):
        p = torch.randn(batch_size, input_size).to(self.device)
        t = torch.randn(batch_size, output_size, requires_grad=False).to(self.device)
        return p, t

    def add_optimizer(self, optimizer):
        self._validate_init(optimizer)
        self.optimizer = optimizer
        return optimizer

    def add_performance_index(self, performance_index):
        self._validate_init(performance_index)
        self.performance_index = performance_index
        return performance_index

    def add_model(self, model):
        self.model = model
        return self.model.to(self.device)

    def add_adjustment(self, adjustment):
        self.adjustment = adjustment

    def add_Droput(self, drop):
        self.drop = nn.Dropout(drop)

    def _run(self, inputs, targets, lr=1e-4):
        ipt, tgt = self._to_device(inputs, targets)
        self.outputs = self.model(ipt)
        loss = self.performance_index(self.outputs, tgt)
        if self.optimizer:
            self.optimizer.zero_grad()
        loss.backward()
        if self.optimizer:
            self.optimizer.step()
        if not self.optimizer:
            # update weights without optimization
            for n, param in enumerate(self.model.parameters()):
                param.data -= lr * param.grad
                if self.collect_grad:
                    self.gradients.append(param.grad)

        self.losses.append(loss.item())
        return loss

    def run_for_epochs(self, inputs=None, targets=None, data_loader=None, epochs=500, lr=1e-4):
        self._start = time.time()
        self.model.train()
        if not data_loader:
            for epoch in range(epochs):
                self._run(inputs, targets, lr)
        else:
            for epoch in range(epochs):
                for i, (inputs, targets) in enumerate(data_loader):
                    loss = self._run(inputs if not self.is_image else inputs.view(-1, self.dimensions), targets, lr)
                    if (i + 1) % 100 == 0:
                        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                              % (epoch + 1, epochs, i + 1, self.data_size // self.batch_size, loss.item()))
                    if self.stop_early_at and (i + 1) % self.stop_early_at == 0:
                        break

    def plot_losses_over_epoch(self, show=True, **kwargs):
        args = {}
        args.update(dict(label='Error'))
        args.update(kwargs)

        df = pd.DataFrame(self.losses, columns=['error'])
        ax = df.plot.line(y='error', **args)
        if show:
            plt.show()
        return ax

    def eval(self, test_loader):
        correct, total = 0, 0
        end = time.time()
        print('Time Spent: {}'.format(end - self._start))
        self.model.eval()
        counter = 0
        for features, labels in test_loader:
            if self.stop_early_at and (counter + 1) % self.stop_early_at == 0:
                break

            features, labels = self._to_device(features, labels)
            features = features if not self.is_image else features.view(-1, self.dimensions)
            outputs = self.model(features)

            if (self.adjustment > 0):
                # print("ADJUSTING")
                outputs = torch.ceil(outputs * self.adjustment)
                outputs = torch.clamp(outputs, min=0, max=1)
            else:
                outputs = torch.round(outputs)
            total += labels.numel()
            predicted = (outputs == labels).sum()
            correct += predicted
            self.pred_output = outputs
            counter = counter + 1

            stacked = np.dstack((torch.round(outputs).detach().cpu().numpy(), (labels.cpu().numpy())))
            self.results.append(stacked)
        return correct, total

    def eval_classes(self, test_loader):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in test_loader:
            features, labels = data
            features, labels = self._to_device(features, labels)
            features = features if not self.is_image else features.view(-1, self.dimensions)
            outputs = self.model(features)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.cpu().numpy()
            c = (predicted.cpu().numpy() == labels)
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1
        return class_correct, class_total

    def get_results(self):
        return np.asarray(self.results).reshape(-1, 2)
