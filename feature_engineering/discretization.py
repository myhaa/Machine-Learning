# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/24
File: discretization.py
Software: PyCharm
Description: 将连续特征离散化（discretization）
"""

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def k_bins_discretizer(X_fit, X_trans=None, n_bins=5, encode='onehot', strategy='uniform'):
	"""
	Bin continuous data into intervals.
	:param X_fit:
	:param X_trans:
	:param n_bins:
	:param encode:
	:param strategy:
	:return:
	"""
	x_bin = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
	x_bin.fit(X_fit)
	print("encode-{} and strategy-{} bin_edges_: ".format(encode, strategy), x_bin.bin_edges_)
	if X_trans is not None:
		res = x_bin.transform(X_trans)
	else:
		res = x_bin.transform(X_fit)
	return res


def equal_width_binning(X_fit, X_trans=None):
	x_bin = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
	x_bin.fit(X_fit)
	print("bin_edges_: ", x_bin.bin_edges_)
	if X_trans is not None:
		res = x_bin.transform(X_trans)
	else:
		res = x_bin.transform(X_fit)
	return res


def equal_freq_binning(X_fit, X_trans=None):
	x_bin = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
	x_bin.fit(X_fit)
	print("bin_edges_: ", x_bin.bin_edges_)
	if X_trans is not None:
		res = x_bin.transform(X_trans)
	else:
		res = x_bin.transform(X_fit)
	return res


def k_means_binning(X_fit, X_trans=None):
	x_bin = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
	x_bin.fit(X_fit)
	print("bin_edges_: ", x_bin.bin_edges_)
	if X_trans is not None:
		res = x_bin.transform(X_trans)
	else:
		res = x_bin.transform(X_fit)
	return res


class ChiMerge():
	"""
	todo 使用ChiMerge方法进行离散化
	有监督的分级自下而上（合并）方法，该方法在本地利用卡方标准来确定两个相邻区间是否足够相似以进行合并
	"""
	pass


class DecisionTreeDiscretization():
	"""
	todo 使用决策树方法进行离散化
	使用决策树来确定确定箱子的最佳分割点
	"""
	pass


if __name__ == '__main__':
	from machine_learning.data_input.load_data import data_loader
	
	train = data_loader()
	print(train.head())
	
	v1, v2 = 'Survived', 'Fare'
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(train, train[v1], test_size=0.3, random_state=0)
	print(X_train.shape, X_test.shape)
	
	# X_test_binning = equal_width_binning(X_train[[v2]], X_train[[v2]])
	# print(X_test_binning[:5])
	#
	# X_test_binning = equal_freq_binning(X_train[[v2]], X_train[[v2]])
	# print(X_test_binning[:5])
	#
	# X_test_binning = k_means_binning(X_train[[v2]], X_train[[v2]])
	# print(X_test_binning[:5])
	#
	X_test_binning = k_bins_discretizer(X_train[[v2]], X_train[[v2]])
	print(X_test_binning.shape)
	print(X_test_binning.toarray())  # sparse matrix to array
	
	X_test_binning = k_bins_discretizer(X_train[[v2]], encode='ordinal')
	print(X_test_binning[:10])
