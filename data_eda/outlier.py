# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/17
File: outlier.py
Software: PyCharm
Description:异常值通用函数合集
"""


# loading modules
import numpy as np
import pandas as pd
from warnings import warn


def outlier_detect_boundary(df, col, lower, upper):
	"""
	检测异常值，指定异常值上下界
	:param df:
	:param col: 列名
	:param lower: 下界
	:param upper: 上界
	:return:
	"""
	tmp = pd.concat([pd.Series(df[col]<lower), pd.Series(df[col]>upper)], axis=1)
	outlier_index = pd.Series(tmp.any(axis=1))
	try:
		print('num of outlier detected:', outlier_index.value_counts()[1])
		print('ratio of outlier detected:', outlier_index.value_counts()[1] / len(outlier_index))
	except:
		warn('Column {} has no outlier'.format(col))
	return outlier_index, (lower, upper)


def outlier_detect_IQR(df, col, threshold=3):
	"""
	检测异常值，IQR方法
	:param df:
	:param col:
	:param threshold: 阈值
	:return:
	"""
	IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
	lower = df[col].quantile(0.25) - (IQR * threshold)
	upper = df[col].quantile(0.75) + (IQR * threshold)
	tmp = pd.concat([pd.Series(df[col]<lower), pd.Series(df[col]>upper)], axis=1)
	outlier_index = pd.Series(tmp.any(axis=1))
	try:
		print('num of outlier detected:', outlier_index.value_counts()[1])
		print('ratio of outlier detected:', outlier_index.value_counts()[1] / len(outlier_index))
	except:
		warn('Column {} has no outlier'.format(col))
	return outlier_index, (lower, upper)


def outlier_detect_mean_std(df, col, threshold=3):
	"""
	检测异常值，mean std
	:param df:
	:param col:
	:param threshold: 阈值
	:return:
	"""
	lower = df[col].mean() - df[col].std() * threshold
	upper = df[col].mean() + df[col].std() * threshold
	tmp = pd.concat([pd.Series(df[col]<lower), pd.Series(df[col]>upper)], axis=1)
	outlier_index = pd.Series(tmp.any(axis=1))
	try:
		print('num of outlier detected:', outlier_index.value_counts()[1])
		print('ratio of outlier detected:', outlier_index.value_counts()[1] / len(outlier_index))
	except:
		warn('Column {} has no outlier'.format(col))
	return outlier_index, (lower, upper)


def outlier_detect_mad(df, col, threshold=3):
	"""
	检测异常值，mad Median and Median Absolute Deviation Method
	比mean std方法更有效，但是更激进
	:param df:
	:param col:
	:param threshold: 阈值
	:return:
	"""
	median = df[col].median()
	median_absolute_deviation = np.median([np.abs(y - median) for y in df[col]])
	modified_z_scores = pd.Series([0.6745 * (y - median) / (median_absolute_deviation + 1e-8) for y in df[col]])
	outlier_index = np.abs(modified_z_scores) > threshold
	try:
		print('num of outlier detected:', outlier_index.value_counts()[1])
		print('ratio of outlier detected:', outlier_index.value_counts()[1] / len(outlier_index))
	except:
		warn('Column {} has no outlier'.format(col))
	return outlier_index


def drop_outlier(df, outlier_index):
	"""
	删除异常值所在行
	:param df:
	:param outlier_index:
	:return:
	"""
	return df[~outlier_index]


def impute_outlier_with_arbitrary(df, outlier_index, impute_value, col=None):
	"""
	替换异常值，用impute_value替换
	:param df:
	:param outlier_index:
	:param impute_value:
	:param col:
	:return:
	"""
	if col is None:
		col = []
	df_copy = df.copy(deep=True)
	for i in col:
		name = i + '_outlier_impute_' + str(impute_value)
		df_copy[name] = df_copy[i]
		df_copy.loc[outlier_index, name] = impute_value
	return df_copy


def impute_outlier_with_method(df, outlier_index, method='mean', col=None):
	"""
	填补异常值，用mean, median, mode
	:param df:
	:param outlier_index:
	:param method:
	:param col:
	:return:
	"""
	if col is None:
		col = []
	df_copy = df.copy(deep=True)
	for i in col:
		if method == 'mean':
			name = i + '_outlier_impute_mean'
			df_copy[name] = df_copy[i]
			df_copy.loc[outlier_index, name] = df_copy[name].mean()
		elif method == 'median':
			name = i + '_outlier_impute_median'
			df_copy[name] = df_copy[i]
			df_copy.loc[outlier_index, name] = df_copy[name].median()
		elif method == 'mode':
			name = i + '_outlier_impute_mode'
			df_copy[name] = df_copy[i]
			df_copy.loc[outlier_index, name] = df_copy[name].mode()
	return df_copy


def impute_outlier_with_wind(df, col, lower, upper, method='both'):
	"""
	将数据范围截断，指定lower,upper
	:param df:
	:param col:
	:param lower: 最低值
	:param upper: 最高值
	:param method: both, lower, upper
	:return:
	"""
	df_copy = df.copy(deep=True)
	name = col+'_outlier_impute_wind'
	df_copy[name] = df_copy[col]
	if method == 'both':
		df_copy.loc[df_copy[name] < lower, name] = lower
		df_copy.loc[df_copy[name] > upper, name] = upper
	elif method == 'lower':
		df_copy.loc[df_copy[name] < lower, name] = lower
	elif method == 'upper':
		df_copy.loc[df_copy[name] > upper, name] = upper
	else:
		raise ValueError('method set error, please set lower, upper or both]')
	return df_copy
	


if __name__ == '__main__':
	import pandas as pd
	# from io import StringIO
	#
	# data = "col1,col2,col3,col4\na,b,1,5\na,b,2,6\nc,d,2,NA"
	# df = pd.read_csv(StringIO(data))
	# print(df.head())
	
	from machine_learning.data_input.load_data import data_loader
	df = data_loader()
	print(df.head())
	
	# v1 = 'Fare'
	# print(outlier_detect_boundary(df, col=v1, lower=1, upper=100))
	# print(outlier_detect_IQR(df, col=v1, threshold=3))
	# print(outlier_detect_mean_std(df, col=v1))
	# print(outlier_detect_mad(df, col=v1))
	# outlier_index, para = outlier_detect_boundary(df, col=v1, lower=1, upper=70)
	# print(impute_outlier_with_arbitrary(df, outlier_index, 10, [v1]))
	# print(impute_outlier_with_method(df, outlier_index, method='median', col=[v1]))
	# print(impute_outlier_with_wind(df, col=v1, lower=1, upper=70, method='both'))
