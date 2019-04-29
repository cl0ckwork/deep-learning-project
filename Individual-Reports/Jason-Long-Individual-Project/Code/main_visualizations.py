import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from lib.ModelBuilder import Builder, ModelRunner
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH

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
        stop_early_at=500 # optional, to speed up local testing, remove when done
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
    print(torch.get_shape())  # JRL, added this query
    return net, runner


if __name__ == '__main__':
    layers = [
        nn.Linear(INPUT_SIZE, CHUNK_SIZE),
        nn.ReLU(),
        nn.Linear(CHUNK_SIZE, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
        nn.LogSoftmax(dim=1)
    ]
    model, runner = main('{}_layer'.format(len(layers)), layers, 'Adam')
    results = runner.get_results()
    cm = confusion_matrix(results[:, 1], results[:, 0])
    print(cm)
    #print_confusion_matrix(cm, [0, 1])

# hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
#  RuntimeError: Expected object of backend CPU but got backend CUDA for argument #2 'mat2'
