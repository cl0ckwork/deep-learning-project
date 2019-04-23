import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from lib.ModelBuilder import Builder, ModelRunner
from lib.measurements.cm_heatmap import print_confusion_matrix
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH

LOCAL = True
USE_LIVE_PRE_PROCESSORS = True

dataset = LoanPerformanceDataset(
    chunk=10,  # size of the query (use a large number here)
    # conn=connect(local=False).connect(),  # connect to remote database instance ( google cloud )
    conn=connect(local=LOCAL).connect(),  # connect to local database (docker)
    ignore_headers=['loan_id'],
    target_column='sdq',
    pre_process_pickle_path=LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if USE_LIVE_PRE_PROCESSORS else PRE_PROCESSING_ENCODERS_PICKLE_PATH
)
print(len(dataset))
LOADER = DataLoader(
    dataset,
    batch_size=1,  # size of batches from the query, 1 === size of query, 2 = 1/2 query size
    num_workers=1,
    shuffle=False
)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 4
BATCH_SIZE = 1
DATA_LEN = 10  # len(dataset)
IS_IMAGE = False


# your epoch iteration here..
# for batch_idx, (features, targets) in enumerate(loader):
#     print('batch: {} size: {}'.format(batch_idx, targets.size()))
#     print(features.size())


# send data to model here


def main(name, layers, optim):
    print('\n** TEST: {} | # Layers: {} | Optimizer: {} **'.format(name, len(layers), optim))

    net = Builder(layers=layers)
    print('Model: ', net)

    runner = ModelRunner(
        model=net,
        batch_size=BATCH_SIZE,
        data_size=DATA_LEN,
        is_image=IS_IMAGE,
        dimensions=10 * 826
    )

    # --------------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    runner.add_performance_index(criterion)
    runner.add_optimizer(optimizer)
    # --------------------------------------------------------------------------------------------
    runner.run_for_epochs(data_loader=LOADER, epochs=NUM_EPOCHS)
    # --------------------------------------------------------------------------------------------
    # correct, total = runner.eval(test_loader)
    # outputs = runner.pred_output
    # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    #
    # _, predicted = torch.max(outputs.data, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # # --------------------------------------------------------------------------------------------
    # class_correct, class_total = runner.eval_classes(test_loader)
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    # # --------------------------------------------------------------------------------------------
    # torch.save(net.state_dict(), 'model_{}.pkl'.format(name))
    # print("END TEST: {}".format(name))
    return net, runner


if __name__ == '__main__':
    layers = [
        nn.Linear(20 * 826, 100),
        nn.ReLU(),
        nn.Linear(100, 2),
        nn.Softmax()
    ]
    model, runner = main(1, layers, 'Adam')
    results = runner.get_results()
    cm = confusion_matrix(results[:, 1], results[:, 0])
    print(cm)
    print_confusion_matrix(cm, [0, 1])
