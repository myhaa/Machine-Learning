# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/24
File: scaling.py
Software: PyCharm
Description: 特征归一化函数合集
"""

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def feature_scaling(X_fit, X_trans=None, method='Standard'):
	if method == 'Standard':
		xs = StandardScaler()
	elif method == 'MinMax':
		xs = MinMaxScaler()
	elif method == 'Robust':
		xs = RobustScaler()
	else:
		raise ValueError('set method error: Standard, MinMax or Robust!')
	print("%s Scaler" % method)
	xs.fit(X_fit)
	if X_trans is not None:
		res = xs.transform(X_trans)
	else:
		res = xs.transform(X_fit)
	return res


def z_score_scaling(X_fit, X_trans=None):
	"""
	正态化
	z = (X - X.mean) / std
	:param X_fit: 用该数据计算均值，方差
	:param X_trans: 利用计算的均值，方差，将该数据进行归一化
	:return:
	"""
	ss = StandardScaler()
	ss.fit(X_fit)
	print('mean: {}, std: {}'.format(ss.mean_, ss.scale_))
	if X_trans is not None:
		res = ss.transform(X_trans)
	else:
		res = ss.transform(X_fit)
	return res


def min_max_scaling(X_fit, X_trans=None):
	"""
	最小最大归一化
	X_scaled = (X - X.min / (X.max - X.min)
	:param X_fit:
	:param X_trans:
	:return:
	"""
	mms = MinMaxScaler()
	mms.fit(X_fit)
	print('min: {}, max: {}'.format(mms.data_min_, mms.data_max_))
	if X_trans is not None:
		res = mms.transform(X_trans)
	else:
		res = mms.transform(X_fit)
	return res


def robust_scaling(X_fit, X_trans=None):
	"""
	X_scaled = (X-X_median) / IQR
	:param X_fit:
	:param X_trans:
	:return:
	"""
	rs = RobustScaler()
	rs.fit(X_fit)
	if X_trans is not None:
		res = rs.transform(X_trans)
	else:
		res = rs.transform(X_fit)
	return res


if __name__ == '__main__':
	from machine_learning.data_input.load_data import data_loader
	
	train = data_loader()
	print(train.head())
	
	v1, v2, v3 = 'Survived', 'Fare', 'Age'
	from sklearn.model_selection import train_test_split
	
	X_train, X_test, y_train, y_test = train_test_split(train, train[v1], test_size=0.3, random_state=0)
	print(X_train.shape, X_test.shape)
	
	
	# X_train_copy = X_train.copy(deep=True)
	# x_train_norm = z_score_scaling(X_train[[v2, v3]], X_trans=X_test[[v2, v3]])
	# print(x_train_norm[:5])
	#
	# x_test_minmax = min_max_scaling(X_train[[v2, v3]], X_trans=X_test[[v2, v3]])
	# print(x_test_minmax[:5])
	#
	# x_test_robust = robust_scaling(X_train[[v2, v3]], X_trans=X_test[[v2, v3]])
	# print(x_test_robust[:5])
	#
	
	x_scale = feature_scaling(X_train[[v2, v3]], method='Standard')
	print(x_scale[:5])
	
	x_scale = feature_scaling(X_train[[v2, v3]], X_trans=X_test[[v2, v3]], method='Standard')
	print(x_scale[:5])
