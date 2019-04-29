import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functools import reduce


def important_features(columns, importances):
    indices = np.argsort(importances)[::-1]
    return pd.Series({columns[indices[f]]: importances[indices[f]] for f in range(len(columns))})


def generic_handler(clf, prfs=None, cols=np.array([]), name=None):
    importances = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else np.array([])

    pr = dict(zip(['recision', 'recall', 'fscore', 'support'], prfs)) if prfs else None
    results = {
        'name': name,
        'feature_ranks': important_features(cols, importances) if importances.any() and cols.any() else pd.Series(),
        'classifier': clf
    }
    return {**results, **pr}


class ModelTester:
    model_handlers = {
        # 'LogisticRegression': generic_handler,
        # 'DecisionTreeClassifier': generic_handler,
        # 'RandomForestClassifier': generic_handler
    }

    def __init__(self,
                 clfs=None,
                 clf_args=None,
                 X=None,
                 y=None,
                 split_data=None,
                 handlers=None,
                 x_normalizer=None,
                 y_normalizer=None):
        self.clfs = clfs
        self.clf_args = clf_args
        self.data_cols = split_data[0].columns if split_data else X.columns if not X.empty else None
        self.split_data = split_data or train_test_split(X, y, test_size=0.3, random_state=0)
        self.split_data_norm = None
        self.handlers = handlers or self.model_handlers
        self.classifiers_names_ = []
        self.results_ = []
        if x_normalizer:
            self._normalize_data(x_normalizer, y_normalizer)

    @property
    def results(self):
        if self.results_:
            return pd.DataFrame(self.results_)

    def _init_classifier_with_args(self, clf, idx, default_args=None):
        _default_args = default_args or {}
        # clf args is a dict, because it allows user to pass args for classifier at any index.
        if isinstance(self.clf_args, dict) and all(map(str.isdigit, self.clf_args.keys())):
            c_args = self.clf_args.get(str(idx), {})
        elif isinstance(self.clf_args, dict):
            c_args = self.clf_args
        else:
            c_args = {}
        _args = {**_default_args, **c_args}
        return clf(**_args)

    def _normalize_data(self, x_norm=None, y_norm=None):
        xn = x_norm() if x_norm else None
        yn = y_norm() if y_norm else None
        if xn:
            X_train, X_test, y_train, y_test = self.split_data
            self.split_data_norm = xn.fit_transform(X_train), \
                                   xn.transform(X_test), \
                                   yn.fit_transform(y_train) if yn else y_train, \
                                   yn.transform(y_test) if yn else y_test

    def run_tests(self, default_args=None, as_dict=False):
        # default_args = default_args or dict(random_state=0)
        for n, clf in enumerate(self.clfs):
            cl = self._init_classifier_with_args(clf, n, default_args)
            test = self.test(cl)
            self.results_.append(test)
            if not as_dict:
                yield test
            else:
                yield {cl.__class__.__name__: test}

    def test(self, clf, name=None, handlers=None, pr_avg='micro'):
        if self.split_data_norm or self.split_data:
            X_train, X_test, y_train, y_test = self.split_data_norm if self.split_data_norm else self.split_data
            cols = self.data_cols  # X_train.columns
            _name = name or clf.__class__.__name__
            print('Running: {} on {} features'.format(_name, len(cols)))

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pr_recall = precision_recall_fscore_support(y_test, y_pred, average=pr_avg)
            handler = handlers.get(name) if handlers else self.handlers.get(_name, generic_handler)
            if handler:
                self.classifiers_names_.append(_name)
                return handler(clf, pr_recall, cols=cols, name=_name)
            print('No handler to work with {} results'.format(_name))

        else:
            print('No data to test on.')

    def plot_importances(self):
        features = [res for res in self.results_ if not res.get('feature_ranks').empty]
        fig, axs = plt.subplots(nrows=len(features), ncols=1, figsize=(20, 20))
        if len(features) >= 2:
            plt.subplots_adjust(hspace=.5)
        for n, res in enumerate(features):
            imp = res.get('feature_ranks')
            if not imp.empty:
                ax = axs[n]
                ax.set_title(res.get('name'))
                imp.sort_values(ascending=False, inplace=True)
                imp.plot(x='Features', y='Importance', kind='bar', rot=45, fontsize=15, ax=ax)

        plt.show()

    @property
    def feature_importances(self, results=None):
        _results = results if not isinstance(results, type(None)) else self.results_
        return {res.get('name'): res.get('feature_ranks') for res in _results if not res.get('feature_ranks').empty}

    # @staticmethod
    # def _reduce_to_min_important_features(agg, imp, clf):
    #     top_imp = imp[imp >= imp.std() * 2]
    #     if not agg:
    #         print('Choosing top importances', clf)
    #         return top_imp.to_dict()
    #     if len(top_imp) < len(agg):
    #         print('Switching top importances', clf)
    #         return top_imp.to_dict()
    #     return agg
    #
    # @property
    # def minimum_top_important_features(self):
        # reduce doesnt send in key 
    #     return reduce(self._reduce_to_min_important_features, self.feature_importances, {})

    def plot_fscores(self):
        if self.results_:
            print(', '.join(
                list(map(lambda x: ' - '.join(x), zip(self.results.name, self.results.fscore.map('{:,.2f}'.format))))))
            self.results.plot(x='name', y='fscore', kind='bar', figsize=(16, 9), rot=45, fontsize=15)
            plt.show()
        else:
            print('no results to plot', self.results)
