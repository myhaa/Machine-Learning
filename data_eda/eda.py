# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/13
File: eda.py
Software: PyCharm
Description: 数据探索通用函数合集
"""

# loading modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-colorblind')


def get_dtypes(df, drop_col=None):
	"""
	获取pandas DataFrame中的离散类型特征、连续类型特征、特征列表
	:param df: pandas DataFrame
	:param drop_col: 想要删除的特征list
	:return:
	"""
	if drop_col is None:
		drop_col = []
	columns_list = list(df.columns)
	continuous_var_list = []
	discrete_var_list = columns_list.copy()
	
	for feature in columns_list:
		if df[feature].dtypes in (int, np.int32, np.int64, np.uint,
								  float, np.float32, np.float64, np.double):
			discrete_var_list.remove(feature)
			continuous_var_list.append(feature)
	
	for feature in drop_col:
		if feature in discrete_var_list:
			discrete_var_list.remove(feature)
		if feature in continuous_var_list:
			continuous_var_list.remove(feature)
	
	return discrete_var_list, continuous_var_list, columns_list


def describe(df, save_path=None):
	"""
	describe the pandas DataFrame
	:param df: pandas DataFrame
	:param save_path: 结果保存目录
	:return:
	"""
	result = df.describe(include='all')
	if save_path:
		save_path = os.path.join(save_path, 'describe.csv')
		result.to_csv(save_path)
		print('result saved at: ', str(save_path))
	return result


def discrete_var_countplot(x, df, save_path=None):
	"""
	Show the counts of observations in each categorical bin using bars.
	:param x: 要绘图的特征
	:param df: pandas DataFrame
	:param save_path: 图片保存路径
	:return:
	"""
	plt.figure(figsize=(15,10))
	sns.countplot(x=x, data=df)
	plt.title('countplot_'+str(x))
	if save_path:
		save_path = os.path.join(save_path, 'countplot_' + str(x) + '.png')
		plt.savefig(save_path)
		print('Image saved at', str(save_path))
	plt.show()


def discrete_var_pointplot(x, y, df, save_path=None):
	"""
	Show point estimates and confidence intervals using scatter plot glyphs.
	:param x: 要绘图的特征
	:param y: target
	:param df: pandas DataFrame
	:param save_path: 图片保存路径
	:return:
	"""
	plt.figure(figsize=(15,10))
	sns.pointplot(x=x, y=y, data=df)
	plt.title('pointplot_'+str(x)+'_'+str(y))
	if save_path:
		save_path = os.path.join(save_path, 'pointplot_' + str(x) +'_'+str(y) + '.png')
		plt.savefig(save_path)
		print('Image saved at', str(save_path))
	plt.show()


def discrete_var_barplot(x, y, df, save_path=None):
	"""
	Show point estimates and confidence intervals as rectangular bars.
	:param x: 要绘图的特征
	:param y: target
	:param df: pandas DataFrame
	:param save_path: 图片保存路径
	:return:
	"""
	plt.figure(figsize=(15,10))
	sns.barplot(x=x, y=y, data=df)
	plt.title('barplot_'+str(x)+'_'+str(y))
	if save_path:
		save_path = os.path.join(save_path, 'barplot_' + str(x) +'_'+str(y) + '.png')
		plt.savefig(save_path)
		print('Image saved at', str(save_path))
	plt.show()


def discrete_var_boxplot(df, save_path=None):
	"""
	Draw a boxplot for each numeric variable in a DataFrame
	:param df: pandas DataFrame
	:param save_path: 图片保存路径
	:return:
	"""
	plt.figure(figsize=(15,10))
	sns.boxplot(data=df)
	plt.title('boxplot_all')
	if save_path:
		save_path = os.path.join(save_path, 'barplot_all.png')
		plt.savefig(save_path)
		print('Image saved at', str(save_path))
	plt.show()


def discrete_var_histplot(x, hue, df, save_path=None):
	"""
	Plot univariate or bivariate histograms to show distributions of datasets
	:param df: pandas DataFrame
	:param save_path: 图片保存路径
	:return:
	"""
	plt.figure(figsize=(15,10))
	sns.histplot(x=x, hue=hue, data=df)
	plt.title('histplot_' + str(x) + '_' + str(hue))
	if save_path:
		save_path = os.path.join(save_path, 'histplot_' + str(x) + '_' + str(hue)+'.png')
		plt.savefig(save_path)
		print('Image saved at', str(save_path))
	plt.show()


def discrete_var_corrplot(df, save_path=None):
	"""
	Plot corr
	:param df: pandas DataFrame
	:param save_path: 图片保存路径
	:return:
	"""
	plt.figure(figsize=(15,10))
	corr = df.corr()
	sns.heatmap(data=corr, cmap="YlGnBu")
	plt.title('corrplot')
	if save_path:
		save_path = os.path.join(save_path, 'corrplot.png')
		plt.savefig(save_path)
		print('Image saved at', str(save_path))
	plt.show()


if __name__ == '__main__':
	# import pandas as pd
	# from io import StringIO
	#
	# data = "col1,col2,col3,col4\na,b,1,5\na,b,2,6\nc,d,2,7"
	# df = pd.read_csv(StringIO(data))
	# print(df.head())
	
	from machine_learning.data_input.load_data import data_loader
	df = data_loader()
	print(df.head())
	
	print(get_dtypes(df))
	print(describe(df))
	
	v1, v2 = 'Survived', 'Fare'
	# discrete_var_countplot(v1, df)
	# discrete_var_pointplot(x=v1, y=v2, df=df)
	# discrete_var_barplot(v1, v2, df)
	# discrete_var_boxplot(df[[v2]])
	# discrete_var_histplot(x=v2, hue=v1, df=df)
	# discrete_var_histplot(x=v2, hue=None, df=df)
	discrete_var_corrplot(df=df[[v1, v2]])
