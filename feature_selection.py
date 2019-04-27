
# classifiers:
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

from lib.utils import ModelTester
from lib.utils.model_tester import generic_handler
from lib.utils import create_heatmap
from lib.db import connect
from lib.data.loader import LoanPerformanceDataset
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH

LOCAL = False
USE_LIVE_PRE_PROCESSORS = not LOCAL
CHUNK_SIZE = 100

dataset = LoanPerformanceDataset(
    chunk=CHUNK_SIZE,  # size of the query (use a large number here)
    conn=connect(local=LOCAL).connect(),  # connect to local or remote database (docker, google cloud)
    ignore_headers=['loan_id'],
    target_column='sdq',
    pre_process_pickle_path=LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if USE_LIVE_PRE_PROCESSORS else PRE_PROCESSING_ENCODERS_PICKLE_PATH,
    stage='train',
    to_tensor=False
)

X, y = dataset[0]

classifiers = [DecisionTreeClassifier, RandomForestClassifier]
classifier_args = {
    '0': dict(random_state=0, warm_start=True),
    '1': dict(random_state=0, n_estimators=30, n_jobs=4, warm_start=True),
}

tester = ModelTester(
    classifiers,
    X=X,
    y=y,
    clf_args=classifier_args,
    handlers={
        'DecisionTreeClassifier': generic_handler,
        'RandomForestClassifier': generic_handler
    }
)

results = list(tester.run_tests(default_args={}))
# for r in results:
#     print(r)

# tester.plot_fscores()
top = tester.plot_importances(top=True, std_dev_mult=1)

top_features = X[top.keys()]
sns.pairplot(top_features, size=top_features.shape[1])
plt.show()

create_heatmap(top_features)
print(top_features)
