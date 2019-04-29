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

LOCAL = True
USE_LIVE_PRE_PROCESSORS = not LOCAL
CHUNK_SIZE = 50000

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
    '0': dict(random_state=0),
    '1': dict(random_state=0, n_estimators=30, n_jobs=4),
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



results = list(tester.run_tests())
tester.plot_fscores()

top = tester.plot_importances(top=True, std_dev_mult=1)
top_features = X[list(top.keys())[:8]]

print(top_features.sample(3))
print(top_features.columns)
# ['borrower_credit_score_at_origination', 'original_upb',
#        'original_debt_to_income_ratio', 'original_loan_to_value',
#        'co_borrower_credit_score_at_origination',
#        'primary_mortgage_insurance_percent', 'first_payment_month_cos',
#        'origination_month_cos']

sns.pairplot(
    top_features,
    size=top_features.shape[1],
    hue="borrower_credit_score_at_origination",
)
plt.show()


sns.pairplot(
    top_features,
    size=top_features.shape[1],
    vars=["borrower_credit_score_at_origination","original_debt_to_income_ratio"],
    hue="borrower_credit_score_at_origination",
)
plt.show()

create_heatmap(top_features)
