# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/17
File: missing_data.py
Software: PyCharm
Description: 缺失数据通用函数合集
"""


# loading modules
import os

from warnings import warn


def check_missing(df, save_path=None):
	"""
	统计各特征缺失数和缺失比例
	:param df: pandas Dataframe
	:param save_path:  保存路径
	:return:
	"""
	res = pd.concat([df.isnull().sum(), df.isnull().mean()], axis=1)
	res = res.rename(index=str, columns={0: 'total missing', 1: 'ratio'})
	if save_path:
		save_path = os.path.join(save_path, 'missing.csv')
		res.to_csv(save_path)
		print('missing result saved at: ', save_path)
	return res


def drop_missing(df, axis=0):
	"""
	删除NA所在行或者列
	:param df:
	:param axis: 同dropna的axis
	:return:
	"""
	df_copy = df.copy(deep=True)
	df_copy = df_copy.dropna(axis=axis, inplace=False)
	return df_copy


def impute_NA_with_arbitrary(df, impute_value, NA_col=None):
	"""
	填补缺失值，用指定值填补
	:param df:
	:param impute_value: 填补值
	:param NA_col: 需要填补的特征list
	:return:
	"""
	if NA_col is None:
		NA_col = []
	df_copy = df.copy(deep=True)
	for i in NA_col:
		if df_copy[i].isnull().sum() > 0:
			df_copy[i+'_NA_impute_'+str(impute_value)] = df_copy[i].fillna(impute_value)
		else:
			warn("Column {} has no missing".format(i))
	return df_copy


def impute_NA_with_method(df, method='mean', NA_col=None):
	"""
	填补缺失值，用均值、中位数、众数等方法
	:param df:
	:param method: 指定方法
	:param NA_col: 需要填补的特征list
	:return:
	"""
	if NA_col is None:
		NA_col = []
	df_copy = df.copy(deep=True)
	for i in NA_col:
		if df_copy[i].isnull().sum()>0:
			if method == 'mean':
				df_copy[i+'_NA_impute_mean'] = df_copy[i].fillna(df[i].mean())
			elif method == 'median':
				df_copy[i + '_NA_impute_median'] = df_copy[i].fillna(df[i].median())
			elif method == 'mode':
				df_copy[i + '_NA_impute_mode'] = df_copy[i].fillna(df[i].mode()[0])
		else:
			warn("Column {} has no missing".format(i))
	return df_copy


def impute_NA_with_distribution(df, NA_col=None):
	"""
	填补缺失值 at the far end of the distribution of that variable calculated by
	mean + 3*std
	:param df:
	:param NA_col: 需要填补的特征list
	:return:
	"""
	if NA_col is None:
		NA_col = []
	df_copy = df.copy(deep=True)
	for i in NA_col:
		if df_copy[i].isnull().sum()>0:
			df_copy[i + '_NA_impute_distribution'] = df_copy[i].fillna(df[i].mean()+3*df[i].std())
		else:
			warn("Column {} has no missing".format(i))
	return df_copy


def impute_NA_with_random_sampling(df, NA_col=None, random_state=0):
	"""
	填补缺失值，从样本中随机抽样填补
	:param df:
	:param NA_col:
	:param random_state:
	:return:
	"""
	if NA_col is None:
		NA_col = []
	df_copy = df.copy(deep=True)
	for i in NA_col:
		if df_copy[i].isnull().sum()>0:
			df_copy[i+'_NA_impute_random_sampling'] = df_copy[i]
			random_sampling = df_copy[i].dropna().sample(df_copy[i].isnull().sum(), random_state=random_state)
			random_sampling.index = df_copy[df_copy[i].isnull()].index
			df_copy.loc[df_copy[i].isnull(), str(i)+'_NA_impute_random_sampling'] = random_sampling
		else:
			warn("Column {} has no missing".format(i))
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
	
	print(check_missing(df))
	# print(drop_missing(df, axis=0))
	v1 = 'Age'
	print(impute_NA_with_arbitrary(df, 10, NA_col=[v1]))
	print(impute_NA_with_method(df, method='median', NA_col=[v1]))
	print(impute_NA_with_distribution(df, NA_col=[v1]))
	print(impute_NA_with_random_sampling(df, NA_col=[v1], random_state=0))
