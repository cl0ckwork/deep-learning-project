import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

from lib.ModelBuilder import Builder, ModelRunner
# from lib.measurements.cm_heatmap import print_confusion_matrix
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH

LOCAL = True
USE_LIVE_PRE_PROCESSORS = not LOCAL
CHUNK_SIZE = 100
NUERONS_l1 = 1000
NUERONS_l2 = 1000
LOADER_ARGS = dict(
    batch_size=1,  # size of batches from the query, 1 === size of query, 2 = 1/2 query size
    num_workers=1,
    shuffle=True
)

HEADERS = [
    'borrower_credit_score_at_origination',
    'original_upb',
    'original_debt_to_income_ratio',
    'original_loan_to_value',
    'co_borrower_credit_score_at_origination',
    'primary_mortgage_insurance_percent',
    'first_payment_month_cos',
    'zip_code_short',
    'origination_month_cos'
]

dataset = LoanPerformanceDataset(
    chunk=CHUNK_SIZE,  # size of the query (use a large number here)
    conn=connect(local=LOCAL).connect(),  # connect to local or remote database (docker, google cloud)
    ignore_headers=['co_borrower_credit_score_at_origination'],
    headers=None,  # HEADERS
    target_column='sdq',
    pre_process_pickle_path=LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if USE_LIVE_PRE_PROCESSORS else PRE_PROCESSING_ENCODERS_PICKLE_PATH,
    stage='train',
)

TRAIN_LOADER = DataLoader(dataset, **LOADER_ARGS)

sample, targets = next(iter(TRAIN_LOADER))

INPUT_SIZE = np.prod(sample.size())
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 1
DATA_LEN = len(dataset)
STOP_EARLY = CHUNK_SIZE * 5

print('\n** INFO ** ')
print('DATA_LEN:', len(dataset))
print('INPUT_SIZE:', INPUT_SIZE)
print('STOP_EARLY:', STOP_EARLY)


def main(name, layers, optim, drop, adjust):
    print('\n** TEST: {} | # Layers: {} | Optimizer: {} **\n'.format(name, len(layers), optim))

    net = Builder(layers=layers)
    print('Model: ', net)

    runner = ModelRunner(
        model=net,
        batch_size=BATCH_SIZE,
        data_size=DATA_LEN,
        is_image=True,
        dimensions=INPUT_SIZE,
        stop_early_at=STOP_EARLY,  # optional, to speed up local testing, remove when done

    )

    # --------------------------------------------------------------------------------------------
    criterion = torch.nn.MSELoss(reduction='sum')
    opt = getattr(torch.optim, optim)
    optimizer = opt(net.parameters(), lr=LEARNING_RATE)
    runner.add_performance_index(criterion)
    runner.add_optimizer(optimizer)
    runner.add_adjustment(adjust)
    runner.add_Droput(drop)
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
    ## Trying with Adam
    layers = [
        nn.Linear(INPUT_SIZE, NUERONS_l1),
        nn.ReLU(),
        nn.Linear(NUERONS_l1, CHUNK_SIZE),
        nn.ReLU(),
    ]
    model, runner = main('{}_layer'.format(len(layers) / 2), layers, 'Adam', 0, 1)
    results = runner.get_results()
    # cm = confusion_matrix(results[:, 1],results[:, 0])
    cm = ConfusionMatrix(results[:, 1], results[:, 0])
    print("2 Layer Relu Model ")
    print(cm)
    cm.plot()
    plt.show()
    # print_confusion_matrix(cm, [0, 1])

    ## Sigmoid model
    layers2 = [
        nn.Linear(INPUT_SIZE, NUERONS_l1),
        nn.ReLU(),
        nn.Linear(NUERONS_l1, CHUNK_SIZE),
        nn.Sigmoid()
    ]
    model2, runner2 = main('{}_layer'.format(len(layers2) / 2), layers2, 'Adam', 0, 0)
    results2 = runner2.get_results()
    cm2 = ConfusionMatrix(results2[:, 1], results2[:, 0])
    print("2 Layer SIGMOID Model ")
    print(cm2)
    cm2.plot()
    plt.show()

    # print_confusion_matrix(cm, [0, 1])

    ## Relu model 3 layer
    layers3 = [
        nn.Linear(INPUT_SIZE, NUERONS_l1),
        nn.ReLU(),
        nn.Linear(NUERONS_l1, NUERONS_l2),
        nn.ReLU(),
        nn.Linear(NUERONS_l2, CHUNK_SIZE),
        nn.ReLU()
    ]
    model3, runner3 = main('{}_layer'.format(len(layers3) / 2), layers3, 'Adam', 0, 1)
    results3 = runner3.get_results()
    cm3 = ConfusionMatrix(results3[:, 1], results3[:, 0])
    print("3 Layer RELu Model ")
    print(cm3)
    cm3.plot()
    plt.show()

    # print_confusion_matrix(cm, [0, 1])

    ## Softmax model
    layers4 = [
        nn.Linear(INPUT_SIZE, NUERONS_l1),
        nn.ReLU(),
        nn.Linear(NUERONS_l1, CHUNK_SIZE),
        nn.Softmax(dim=1)
    ]
    model4, runner4 = main('{}_layer'.format(len(layers4) / 2), layers4, 'Adam', 0, 0.037)
    results4 = runner4.get_results()
    cm4 = ConfusionMatrix(results4[:, 1], results4[:, 0])
    print("2 Layer Softmax  Model ")
    print(cm4)
    cm4.plot()
    plt.show()

    # print_confusion_matrix(cm, [0, 1])

    ## dropout model
    layers5 = [
        nn.Linear(INPUT_SIZE, NUERONS_l1),
        nn.ReLU(),
        nn.Linear(NUERONS_l1, CHUNK_SIZE),
        nn.ReLU(),
    ]
    model5, runner5 = main('{}_layer'.format(len(layers5) / 2), layers5, 'Adam', 0.2, 1)
    results5 = runner5.get_results()
    cm5 = ConfusionMatrix(results5[:, 1], results5[:, 0])
    print("2 Layer Relu  Model with droput ")
    print(cm5)
    cm5.plot()
    plt.show()

    # print_confusion_matrix(cm, [0, 1])
