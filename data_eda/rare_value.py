# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/24
File: rare_value.py
Software: PyCharm
Description: 离散变量处理函数，将频次低的类别作为rare value
"""


import pandas as pd


def get_rare_values(X_fit, cols, X_trans=None, threshold=0.01):
	"""
	将显示稀有标签的观察结果分组为一个独特的类别（“稀有”）
	:return:
	"""
	gp = GroupingRareValues(cols, threshold=0.01)
	gp.fit(X_fit)
	print("gp_mapping: ", gp.mapping)
	if X_trans is not None:
		res = gp.transform(X_trans)
	else:
		res = gp.transform(X_fit)
	return res


class GroupingRareValues(object):
	"""
	将显示稀有标签的观察结果分组为一个独特的类别（“稀有”）
	"""

	def __init__(self, cols=None, threshold=0.01):
		self.mapping = None
		self.cols = cols
		self.threshold = threshold
		self.X_dim = None
	
	def fit(self, X):
		self.X_dim = X.shape[1]
		_, categories = self.grouping(X)
		self.mapping = categories
		return self
	
	def transform(self, X):
		if not self.X_dim:
			raise ValueError('Must train encoder before it can be used to transform data.')
		
		#  make sure that it is the right size
		if X.shape[1] != self.X_dim:
			raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self.X_dim,))
		
		X, _ = self.grouping(X)
		
		return X
	
	
	def grouping(self, X):
		x = X.copy(deep=True)
		
		if self.mapping:
			mapping_out = self.mapping
			for i in self.mapping:
				column = i.get('col')
				x[column] = x[column].map(i['mapping'])
		else:
			mapping_out = []
			for col in self.cols:
				temp_df = pd.Series(x[col].value_counts() / len(x))
				mapping = {k: ('rare' if k not in temp_df[temp_df >= self.threshold].index else k) for k in temp_df.index}
				
				mapping = pd.Series(mapping)
				mapping_out.append({'col': col, 'mapping': mapping, 'data_type': x[col].dtype}, )
		
		return x, mapping_out


if __name__ == '__main__':
	from machine_learning.data_input.load_data import data_loader
	train = data_loader()
	print(train.head())
	
	v1, v2 = 'Pclass', 'SibSp'
	
	# print(train[v2].value_counts())
	# gp = GroupingRareValues(cols=[v1, v2], threshold=0.01)
	# gp.fit(train)
	# print(gp.mapping)
	# train_2 = gp.transform(train)
	# print(train_2.head())
	# print(train_2[v2].value_counts())
	#
	xt = get_rare_values(X_fit=train, cols=[v1, v2])
	print(xt[v2].value_counts())
