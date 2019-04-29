import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmap(df, cols=None, method='pearson', **heatargs):
    _cols = cols or df.columns
    cm = df[cols].corr(method=method) if cols else df.corr(method=method)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm,
                     **{
                         **dict(cbar=True,
                                annot=True,
                                square=True,
                                fmt='.2f',
                                annot_kws={'size': 15},
                                yticklabels=_cols,
                                xticklabels=_cols),
                         **heatargs
                     }
                     )
    plt.tight_layout()
    plt.show()
