from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt


def create_learning_curve(estimator, X, y, **kwargs):
    return learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        **{
            **dict(train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=4),
            **kwargs}
    )


def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    # plt.ylim([0.8, 1.03])
    plt.tight_layout()
    plt.show()
