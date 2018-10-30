import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import seaborn as sns
import sys



# Plotting Features
def plotfeats(frame, feats, kind, cols=4):
    """
    :param frame:
    :param feats:
    :param kind:
    :param cols:
    :return:
    """
    rows = int(np.ceil(len(feats) / cols))
    if rows == 1 and len(feats) < cols:
        cols = len(feats)

    if kind == 'hs':
        fig, axes = plt.subplots(nrows=rows * 2, ncols=cols, figsize=(cols * 5, rows * 10))
    else:
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))

        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

    i = 0
    for f in feats:

        row = int(i / cols)
        col = i % cols

        if kind == 'hist':
            frame.plot.hist(y=f, bins=100, ax=axes[row, col])
        elif kind == 'scatter':
            frame.plot.scatter(x=f, y='SalePrice', ylim=(0, 800000), ax=axes[row, col])
        elif kind == 'hs':
            frame.plot.hist(y=f, bins=100, ax=axes[row * 2, col])
            frame.plot.scatter(x=f, y='SalePrice', ylim=(0, 800000), ax=axes[row * 2 + 1, col])
        elif kind == 'box':
            frame.plot.box(y=f, ax=axes[row, col])
        elif kind == 'boxp':
            sns.boxplot(x=f, y='SalePrice', data=frame, ax=axes[row, col])
        i += 1

    plt.show()

# Analysis of Variance
def anovaXY(data):
    """
    :param data:
    :return:
    """
    samples = []
    X = data.columns[0]
    Y = data.columns[1]
    for level in data[X].unique():
        if (type(level) == float):
            s = data[data[X].isnull()][Y].values
        else:
            s = data[data[X] == level][Y].values
        samples.append(s)
    f, p = stats.f_oneway(*samples)
    return (f,p)

# Plotting Features with Spearman metrics
def spearman(frame, features):
    """
    :param frame:
    :param features:
    :return:
    """
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.2*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    plt.show()

# Get Outlier Id
def outlierId(frame, feature, num):
    """
    :param frame:
    :param feature:
    :param num:
    :return:
    """
    return list(frame.sort_values(by=feature, ascending=False)[:num]['Id'])


def NaNRatio(frame, feats):
    """
    :param frame:
    :param feats:
    :return:
    """
    na_count = frame[feats].isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(frame)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])

    return na_data[na_data['count'] > 0]


def NARatio(frame, feats):
    """
    :param frame:
    :param feats:
    :return:
    """
    nadict = {}
    for c in feats:
        for r in frame.index:
            if 'NA' == frame.loc[r, c]:
                if nadict.get(c, 0) == 0:
                    nadict[c] = []
                nadict[c].append(r)

    return nadict


def transNaNtoNumber(frame, column, method, val=0):
    """
    :param frame:
    :param column:
    :param method:
    :param val:
    :return:
    """
    if method == 'mean':
        frame[column] = frame[column].fillna(round(frame[column].mean()))
    elif method == 'min':
        frame[column] = frame[column].fillna(round(frame[column].min()))
    elif method == 'max':
        frame[column] = frame[column].fillna(round(frame[column].max()))
    elif method == 'special':
        frame[column] = frame[column].fillna(val).round()

    return frame


def transNaNtoNA(frame, feature):
    """
    :param frame:
    :param feature:
    :return:
    """
    frame.loc[frame[feature].isnull(), feature] = 'NA'


def transNAtoNumber(frame, feat, val=0):
    """
    :param frame:
    :param feat:
    :param val:
    :return:
    """
    for r in frame[frame[feat] == 'NA'].index:
        frame.loc[r, feat] = val

    return frame


def encode(frame, feature, targetfeature='SalePrice'):
    """
    :param frame:
    :param feature:
    :param targetfeature:
    :return:
    """
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val

    ordering['price_mean'] = frame[[feature, targetfeature]].groupby(feature).mean()[targetfeature]
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['order'].to_dict()

    return ordering

def select_features(dataFrame, num, method='pearson', target_feature='SalePrice'):
    """
    :param dataFrame:
    :param num:
    :param method:
    :param target_feature:
    :return:
    """
    return dataFrame.corr(method=method).nlargest(num, target_feature).index


