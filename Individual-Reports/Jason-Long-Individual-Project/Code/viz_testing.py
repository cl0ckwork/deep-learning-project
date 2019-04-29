import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from lib.ModelBuilder import Builder, ModelRunner
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH

import numpy as np
import torch.nn as nn

import hiddenlayer as hl

# Visualization Loading
#-- One Chart --#
# A History object to store metrics
history1 = hl.History()

# A Canvas object to draw the metrics
canvas1 = hl.Canvas()

LOCAL = False
USE_LIVE_PRE_PROCESSORS = not LOCAL
CHUNK_SIZE = 100  # fails on chunks 250/500/750 (GPU limit over 750)
LOADER_ARGS = dict(
    batch_size=2,  # size of batches from the query, 1 === size of query, 2 = 1/2 query size
    num_workers=1,
    shuffle=True
)

dataset = LoanPerformanceDataset(
    chunk=CHUNK_SIZE,  # size of the query (use a large number here)
    conn=connect(local=LOCAL).connect(),  # connect to local or remote database (docker, google cloud)
    ignore_headers=['loan_id'],
    target_column='sdq',
    pre_process_pickle_path=LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if USE_LIVE_PRE_PROCESSORS else PRE_PROCESSING_ENCODERS_PICKLE_PATH,
    stage='train',
)

TRAIN_LOADER = DataLoader(dataset, **LOADER_ARGS)

sample, targets = next(iter(TRAIN_LOADER))

INPUT_SIZE = np.prod(sample.size())
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1  # JRL: EPOCHS are safe to mess with
BATCH_SIZE = 1  # JRL: changed from 1, does this do anything?, switched to 25 and no change
DATA_LEN = len(dataset)

print('\n** INFO ** ')
print('DATA_LEN:', len(dataset))
print('INPUT_SIZE:', INPUT_SIZE)


def main(name, layers, optim):
    print('\n** TEST: {} | # Layers: {} | Optimizer: {} **\n'.format(name, len(layers), optim))

    net = Builder(layers=layers)
    print('Model: ', net)

    runner = ModelRunner(
        model=net,
        batch_size=BATCH_SIZE,
        data_size=DATA_LEN,
        is_image=True,
        dimensions=INPUT_SIZE,
        stop_early_at=50 # optional, to speed up local testing, remove when done
    )

    # --------------------------------------------------------------------------------------------
    criterion = torch.nn.MSELoss(reduction='sum')
    opt = getattr(torch.optim, optim)
    optimizer = opt(net.parameters(), lr=LEARNING_RATE)
    runner.add_performance_index(criterion)
    runner.add_optimizer(optimizer)
    # --------------------------------------------------------------------------------------------
    runner.run_for_epochs(data_loader=TRAIN_LOADER, epochs=NUM_EPOCHS)
    # --------------------------------------------------------------------------------------------

    dataset.set_stage('test')  # use the test data
    TEST_LOADER = DataLoader(dataset, **LOADER_ARGS)

    correct, total = runner.eval(TEST_LOADER)
    # outputs = runner.pred_output

    # --------------------------------------------------------------------------------------------
    dataset.set_stage('validate')  # use the validation data
    VALIDATION_LOADER = DataLoader(dataset, **LOADER_ARGS)
    val_correct, val_total = runner.eval(VALIDATION_LOADER)
    # --------------------------------------------------------------------------------------------
    print('Accuracy of the network: %d %%' % (100 * correct / total))

    torch.save(net.state_dict(), 'model_{}_{}.pkl'.format(name, optim))
    print("END TEST: {}".format(name))
    return net, runner


if __name__ == '__main__':
    layers = [
        nn.Linear(INPUT_SIZE, CHUNK_SIZE),
        nn.ReLU(),
        nn.Linear(CHUNK_SIZE, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
        # nn.LogSoftmax(dim=1)
    ]
    model, runner = main('{}_layer'.format(len(layers)), layers, 'Adam')
    results = runner.get_results()
    cm = confusion_matrix(results[:, 1], results[:, 0])
    print(cm)
    #print_confusion_matrix(cm, [0, 1])





# print(model)
# print(runner)  # runner is empty
# print(net)  # NameError: name 'net' is not defined


# hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
#  RuntimeError: Expected object of backend CPU but got backend CUDA for argument #2 'mat2'

#  Trying to get the confusion matrix to function correctly

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# labels_all = []
# predicted_all = []
# print(predicted_all)
#
#
# for data in TRAIN_LOADER:
#     images, labels = data
#     images = Variable(images.view(INPUT_SIZE).cuda())
#     labels = labels.cpu()
#     outputs = model(images)
#     _, predicted = torch.max(outputs.data, 0)
#     labels_all.extend(labels.cpu().numpy().tolist())
#     predicted_all.extend(predicted.cpu().numpy().tolist())
#     labels = labels.cuda().numpy()
#     c = (predicted.cuda().numpy() == labels)
#
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1
#
#
#
#
# y_true = []
# for label in labels_all:
#     y_true.append(classes[label])
#
# y_pred = []
# for pred in predicted_all:
#     y_pred.append(classes[pred])
#
# print(ConfusionMatrix(y_true, y_pred))


# model = da_rnn(file_data = '{}/data/nasdaq100_padding.csv'.format(io_dir), logger = logger, parallel = False,
#               learning_rate = .001)
#
# model.train(n_epochs = 500)

# y_pred = model.predict()

# plt.figure()
# plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
# plt.show()

# plt.figure()
# plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
# plt.show()

# plt.figure()
# plt.plot(y_pred, label = 'Predicted')
# plt.plot(model.y[model.train_size:], label = "True")
# plt.legend(loc = 'upper left')
# plt.show()