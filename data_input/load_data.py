# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/24
File: load_data.py
Software: PyCharm
Description:
"""

import pandas as pd
import os


def data_loader():
	# 获取当前文件的目录
	cur_path = os.path.abspath(os.path.dirname(__file__))
	# print(cur_path)
	
	use_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']
	train_fpath = cur_path + '/dataset/titanic/train.csv'
	train = pd.read_csv(train_fpath, usecols=use_cols)
	return train


if __name__ == '__main__':
	df = data_loader()
	print(df.head())
