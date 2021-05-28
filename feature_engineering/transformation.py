# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/24
File: transformation.py
Software: PyCharm
Description: 特征变换，feature transformation
"""
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import numpy as np
import pandas as pd


def log_transform(df, cols=None):
	"""
	log transform
	:param df:
	:param cols:
	:return:
	"""
	if not cols:
		cols = []
	df_copy = df.copy(deep=True)
	for i in cols:
		i_name = i + '_log'
		df_copy[i_name] = np.log(df_copy[i] + 1)
		print('Var ' + i_name + ' Q-Q plot')
		qqplot(df_copy, i_name)
	return df_copy


def reciprocal_transform(df, cols=None):
	"""
	倒数变换
	:param df:
	:param cols:
	:return:
	"""
	if not cols:
		cols = []
	df_copy = df.copy(deep=True)
	for i in cols:
		i_name = i + '_reciprocal'
		df_copy[i_name] = 1 / (df_copy[i] + 1e-8)
		print('Var ' + i_name + ' Q-Q plot')
		qqplot(df_copy, i_name)
	return df_copy


def sqrt_transform(df, cols=None):
	"""
	根号变换
	:param df:
	:param cols:
	:return:
	"""
	if not cols:
		cols = []
	df_copy = df.copy(deep=True)
	for i in cols:
		i_name = i + '_sqrt'
		df_copy[i_name] = df_copy[i] ** 0.5
		print('Var ' + i_name + ' Q-Q plot')
		qqplot(df_copy, i_name)
	return df_copy


def exp_transform(df, coef, cols=None):
	"""
	指数变换
	:param df:
	:param cols:
	:return:
	"""
	if not cols:
		cols = []
	df_copy = df.copy(deep=True)
	for i in cols:
		i_name = i + '_exp'
		df_copy[i_name] = df_copy[i] ** coef
		print('Var ' + i_name + ' Q-Q plot')
		qqplot(df_copy, i_name)
	return df_copy


def box_cox_transform(X_fit, X_trans=None):
	"""
	box cox 变换
	:param X_fit:
	:param X_trans:
	:return:
	"""
	from sklearn.preprocessing import PowerTransformer
	pt = PowerTransformer(method='box-cox')
	pt.fit(X_fit)
	if X_trans is not None:
		res = pt.transform(X_trans)
	else:
		res = pt.transform(X_fit)
	
	i_name = 'box_cox'
	print('Var ' + i_name + ' Q-Q plot')
	df_copy = pd.DataFrame(res, columns=[i_name])
	qqplot(df_copy, i_name)
	return res


def quantile_transform(X_fit, X_trans=None):
	"""
	box cox 变换
	:param X_fit:
	:param X_trans:
	:return:
	"""
	from sklearn.preprocessing import QuantileTransformer
	pt = QuantileTransformer(output_distribution='normal')
	pt.fit(X_fit)
	if X_trans is not None:
		res = pt.transform(X_trans)
	else:
		res = pt.transform(X_fit)
	
	i_name = 'QuantileTransformer'
	print('Var ' + i_name + ' Q-Q plot')
	df_copy = pd.DataFrame(res, columns=[i_name])
	qqplot(df_copy, i_name)
	return res


def qqplot(df, variable):
	"""
	qq plot
	:param df:
	:param variable:
	:return:
	"""
	plt.figure(figsize=(15, 6))
	plt.subplot(121)
	df[variable].hist()
	plt.subplot(122)
	stats.probplot(df[variable], dist='norm', plot=pylab)
	plt.show()


if __name__ == '__main__':
	from machine_learning.data_input.load_data import data_loader
	
	train = data_loader()
	print(train.head())
	
	v1, v2, v3 = 'Survived', 'Fare', 'Age'
	from sklearn.model_selection import train_test_split
	
	X_train, X_test, y_train, y_test = train_test_split(train, train[v1], test_size=0.3, random_state=0)
	print(X_train.shape, X_test.shape)
	
	X_train_trans = log_transform(X_train, cols=[v2])
	print(X_train_trans.head())

	# X_train_trans = reciprocal_transform(X_train, cols=[v2])
	# print(X_train_trans.head())
	#
	# X_train_trans = sqrt_transform(X_train, cols=[v2])
	# print(X_train_trans.head())
	#
	# X_train_trans = exp_transform(X_train, coef=0.2, cols=[v2])
	# print(X_train_trans.head())
	#
	# X_train_copy = X_train[X_train.Fare != 0]
	# X_train_trans = box_cox_transform(X_train_copy[[v2]])
	# print(X_train_trans[:10])
	#
	# X_train_copy = X_train[X_train.Fare != 0]
	# X_train_trans = quantile_transform(X_train_copy[[v2]])
	# print(X_train_trans[:10])
