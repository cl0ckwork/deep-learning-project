import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_matching_item_in_list(array, idx):
    try:
        return array[idx]
    except IndexError:
        return None


class LayerPackage:
    __slots__ = ('layer', 'activation')

    def __init__(self, layer, activation):
        self._validate_init(layer)
        self._validate_init(activation)
        self.layer = layer
        self.activation = activation

    @staticmethod
    def _validate_init(class_input):
        class_name = getattr(class_input, '__module__', None)
        if not 'torch.optim' in class_name and not 'torch.nn.modules' in class_name:
            raise Exception(
                'You must provide torch.nn modules for performance_index and optimizer, you provided: {}'.format(
                    type(class_input)))


class LayerBuilder:
    __slots__ = ('input_size', 'layer_modules', 'hidden_layers', 'output_size', 'n_layers')

    def __init__(self, input_size=0, hidden_layers=None, output_size=0, layer_modules=None, hidden_activation=None):
        self._validate_hidden_layers(hidden_layers)
        length_of_layers = len(hidden_layers) + 1
        self.input_size = input_size
        self.layer_modules = layer_modules or [
            LayerPackage(
                nn.Linear,
                hidden_activation or nn.LogSigmoid if n + 2 != length_of_layers else nn.ReLU
            ) for n in range(length_of_layers)
        ]
        self.hidden_layers = hidden_layers
        self.output_size = output_size

    @staticmethod
    def _validate_hidden_layers(hidden_layers):
        if not isinstance(hidden_layers, list):
            raise Exception(
                'You must provide your hidden_layers as a list of ints (output sizes), you provided: {}'.format(
                    type(hidden_layers)))

    def build_layers(self):
        layers = []
        activations = []
        if self.hidden_layers:
            for idx, size in enumerate(self.hidden_layers):
                module = _get_matching_item_in_list(self.layer_modules, idx)
                if module:
                    activations.append(module.activation())
                    if idx == 0:
                        layers.append(module.layer(self.input_size, size))
                    else:
                        prev_output_size = _get_matching_item_in_list(self.hidden_layers, idx - 1)
                        layers.append(module.layer(prev_output_size, size))
                else:
                    raise Exception(
                        'You are missing a layer module and activation for hidden layer at index {} with size {}'.format(
                            idx, size))
        # add last layer
        last_module = self.layer_modules[-1]
        layers.append(last_module.layer(self.hidden_layers[-1], self.output_size))
        return layers, activations


class Builder(nn.Module):
    __slots__ = ('layer_type', 'activations_type', 'layers', 'activations', 'output')

    def __init__(self, layers, activations):
        super(Builder, self).__init__()
        self.layer_type = self._validate_init(layers)
        self.activations_type = self._validate_init(activations)

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
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
        'is_image', 'dimensions', 'pred_output', 'collect_grad', 'results'
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
                 collect_grad=False
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

    def plot_losses_over_epoch(self, show=True, **kwargs):
        args = {**dict(label='Error'), **kwargs}
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
        for images, labels in test_loader:
            images, labels = self._to_device(images, labels)
            images = images if not self.is_image else images.view(-1, self.dimensions)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            self.pred_output = outputs
            stacked = np.dstack((predicted.cpu().numpy(), labels.cpu().numpy()))
            self.results.append(stacked)
        return correct, total

    def eval_classes(self, test_loader):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in test_loader:
            images, labels = data
            images, labels = self._to_device(images, labels)
            images = images if not self.is_image else images.view(-1, self.dimensions)
            outputs = self.model(images)
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


# GENERAL USE
"""
try:
    from lib.ModelBuilder import Builder, ModelRunner, LayerBuilder
    from lib.cm_heatmap import print_confusion_matrix
    from lib.auc import auc_classes
except:
    from .lib.ModelBuilder import Builder, ModelRunner, LayerBuilder
    from .lib.cm_heatmap import print_confusion_matrix
    from .lib.auc import auc_classes
"""

"""
# --------------------------------------------------------------------------------------------
def test(name, layers, activations, optim):
    print('\n** TEST: {} | # Layers: {} | # Activations: {} | Optimizer: {} **'.format(name, len(layers), len(activations), optim))

    net = Builder(layers=layers, activations=activations)
    print('Model: ', net)

    runner = ModelRunner(
        model=net,
        device=device,
        batch_size=batch_size,
        data_size=len(train_set),
        is_image=True,
        dimensions=dimensions
    )

    # --------------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    runner.add_performance_index(criterion)
    runner.add_optimizer(optimizer)
    # --------------------------------------------------------------------------------------------
    runner.run_for_epochs(data_loader=train_loader, epochs=num_epochs)
    # --------------------------------------------------------------------------------------------
    correct, total = runner.eval(test_loader)
    outputs = runner.pred_output
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # --------------------------------------------------------------------------------------------
    class_correct, class_total = runner.eval_classes(test_loader)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    # --------------------------------------------------------------------------------------------
    torch.save(net.state_dict(), 'model_{}.pkl'.format(name))
    print("END TEST: {}".format(name))


if __name__ == '__main__':
    layers_1 = [
        nn.Linear(input_size, hidden_size),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    ]
    activations_1 = []
    test(1, layers_1, activations_1, 'Adam')
"""

#CONFUSION MATRIX
"""
def test(name, layers, activations, optim):
    print('\n** TEST: {} | # Layers: {} | # Activations: {} | Optimizer: {} **'.format(name, len(layers), len(activations), optim))

    net = Builder(layers=layers, activations=activations)
    print('Model: ', net)

    runner = ModelRunner(
        model=net,
        device=device,
        batch_size=batch_size,
        data_size=len(train_set),
        is_image=True,
        dimensions=dimensions
    )

    # --------------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    runner.add_performance_index(criterion)
    runner.add_optimizer(optimizer)
    # --------------------------------------------------------------------------------------------
    runner.run_for_epochs(data_loader=train_loader, epochs=num_epochs)
    # --------------------------------------------------------------------------------------------
    correct, total = runner.eval(test_loader)
    outputs = runner.pred_output
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # --------------------------------------------------------------------------------------------
    class_correct, class_total = runner.eval_classes(test_loader)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    # --------------------------------------------------------------------------------------------
    torch.save(net.state_dict(), 'model_{}.pkl'.format(name))
    print("END TEST: {}".format(name))
    return net, runner


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix
    layers_1 = [
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    ]
    activations_1 = []
    model, runner = test(1, layers_1, activations_1, 'Adam')
    results = runner.get_results()
    cm = confusion_matrix(results[:, 1], results[:, 0])
    print(cm)
    print_confusion_matrix(cm, list(classes))
    
"""
# AUC
"""
def test(name, layers, activations, optim):
    print('\n** TEST: {} | # Layers: {} | # Activations: {} | Optimizer: {} **'.format(name, len(layers), len(activations), optim))

    net = Builder(layers=layers, activations=activations)
    print('Model: ', net)

    net.layers[-1].register_forward_hook(get_activation('last_layer'))

    runner = ModelRunner(
        model=net,
        device=device,
        batch_size=batch_size,
        data_size=len(train_set),
        is_image=True,
        dimensions=dimensions
    )

    # --------------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    runner.add_performance_index(criterion)
    runner.add_optimizer(optimizer)
    # --------------------------------------------------------------------------------------------
    runner.run_for_epochs(data_loader=train_loader, epochs=num_epochs)
    # --------------------------------------------------------------------------------------------
    correct, total = runner.eval(test_loader)
    outputs = runner.pred_output
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # --------------------------------------------------------------------------------------------
    class_correct, class_total = runner.eval_classes(test_loader)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    # --------------------------------------------------------------------------------------------
    torch.save(net.state_dict(), 'model_{}.pkl'.format(name))
    print("END TEST: {}".format(name))
    return net, runner


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix
    layers_1 = [
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    ]
    activations_1 = []
    model, runner = test(1, layers_1, activations_1, 'Adam')
    results = runner.get_results()
    auc_classes(classes, predictions=results[:, 0], labels=results[:, 1])
"""