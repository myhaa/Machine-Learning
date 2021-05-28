# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/24
File: encoding.py
Software: PyCharm
Description: 特征编码（feature encoding）
"""

import pandas as pd


def onehot_encoding(df, drop_first=False):
	"""
	用不同的布尔变量（0/1）替换类别变量，以指示该观察的某些标签是否为真
	:param df:
	:param drop_first:
	:return:
	"""
	return pd.get_dummies(df, drop_first=drop_first)


def category_encoders_funcs(X_fit, cols, X_trans=None):
	"""
	* [category_encoders](https://contrib.scikit-learn.org/category_encoders/#)
	
	import category_encoders as ce
	encoder = ce.BackwardDifferenceEncoder(cols=[...])
	encoder = ce.BaseNEncoder(cols=[...])
	encoder = ce.BinaryEncoder(cols=[...])
	encoder = ce.CatBoostEncoder(cols=[...])
	encoder = ce.CountEncoder(cols=[...])
	encoder = ce.GLMMEncoder(cols=[...])
	encoder = ce.HashingEncoder(cols=[...])
	encoder = ce.HelmertEncoder(cols=[...])
	encoder = ce.JamesSteinEncoder(cols=[...])
	encoder = ce.LeaveOneOutEncoder(cols=[...])
	encoder = ce.MEstimateEncoder(cols=[...])
	encoder = ce.OneHotEncoder(cols=[...])
	encoder = ce.OrdinalEncoder(cols=[...])
	encoder = ce.SumEncoder(cols=[...])
	encoder = ce.PolynomialEncoder(cols=[...])
	encoder = ce.TargetEncoder(cols=[...])
	encoder = ce.WOEEncoder(cols=[...])
	
	encoder.fit(X, y)
	X_cleaned = encoder.transform(X_dirty)
	:param X_fit:
	:param cols:
	:param X_trans:
	:return:
	"""
	import category_encoders as ce
	encoder = ce.OneHotEncoder(cols=cols, use_cat_names=True)
	encoder.fit(X_fit)
	if X_trans is not None:
		res = encoder.transform(X_trans)
	else:
		res = encoder.transform(X_fit)
	return res


if __name__ == '__main__':
	from machine_learning.data_input.load_data import data_loader
	
	train = data_loader()
	print(train.head())
	
	v1, v2 = 'Survived', 'Sex'
	from sklearn.model_selection import train_test_split
	
	X_train, X_test, y_train, y_test = train_test_split(train, train[v1], test_size=0.3, random_state=0)
	print(X_train.shape, X_test.shape)
	
	x_onehot = onehot_encoding(X_train[v2])
	print(x_onehot.head())
	
	x_onehot = onehot_encoding(X_train)
	print(x_onehot.head())
	
	x_onehot = category_encoders_funcs(X_train, cols=[v2], X_trans=X_test)
	print(x_onehot.head())
	
	x_onehot = category_encoders_funcs(X_train, cols=[v2])
	print(x_onehot.head())
