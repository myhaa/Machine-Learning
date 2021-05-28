# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: meiyunhe@corp.netease.com
Date: 2021/05/25
File: selection_method.py
Software: PyCharm
Description: feature selection method
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression


def detect_constant_feature(df, threshold=0.9):
	"""
	检测大部分值都是一样的特征
	:param df:
	:param threshold:
	:return:
	"""
	df_copy = df.copy(deep=True)
	constant_feature_list = []
	for feature in df_copy.columns:
		ratio = (df_copy[feature].value_counts() / float(df_copy.shape[0])).sort_values(ascending=False).values[0]
		# print(ratio)
		if ratio >= threshold:
			constant_feature_list.append(feature)
	print(len(constant_feature_list), ' features are found to be almost constant')
	return constant_feature_list


def detect_correlation_feature(df, threshold=0.7):
	"""
	检测相关性较强的特征
	:param df:
	:param threshold:
	:return:
	"""
	corr_matrix = df.corr().abs().unstack().sort_values(ascending=False)
	corr_matrix = corr_matrix[corr_matrix >= threshold]
	corr_matrix = corr_matrix[corr_matrix < 1]
	corr_matrix = pd.DataFrame(corr_matrix).reset_index()
	corr_matrix.columns = ['f1', 'f2', 'corr']
	
	# print(corr_matrix)
	
	visited = []
	df_corr_final = pd.DataFrame()

	for feature in corr_matrix['f1'].unique():
		if feature not in visited:
			df_corr_block = corr_matrix[corr_matrix['f1'] == feature]
			visited = visited + list(df_corr_block['f2'].unique()) + [feature]

			df_corr_final = pd.concat([df_corr_final, df_corr_block], axis=0)
	return df_corr_final


def select_feature_with_chi2_or_mutual(X, y, score_func=mutual_info_classif, k=10):
	"""
	Select features according to the k highest scores
	Estimate mutual information for a discrete target variable
	:param X:
	:param y:
	:param k:
	:return:
	
	See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    Generic Univariate Select : Univariate feature selector with configurable mode.
	"""
	if k >= 1:
		k_best = SelectKBest(score_func=score_func, k=k)
		k_best.fit(X, y)
		col = X.columns[k_best.get_support()]
	elif 0< k < 1:
		k_best = SelectPercentile(score_func=score_func, percentile=k*100)
		k_best.fit(X, y)
		col = X.columns[k_best.get_support()]
	else:
		raise ValueError('param k must be positive!')
	return col


def select_feature_with_decision_tree(X_train, y_train, X_test, y_test, threshold, method='roc_auc'):
	"""
	使用决策树方法来筛选特征
	:param X_train:
	:param y_train:
	:param X_test:
	:param y_test:
	:param threshold:
	:param method:
	:return:
	"""
	method_values = []
	if method == 'roc_auc':
		for feature in X_train.columns:
			clf = DecisionTreeClassifier()
			clf.fit(X_train[feature].to_frame(), y_train)
			y_scored = clf.predict_proba(X_test[feature].to_frame())
			method_values.append(roc_auc_score(y_test, y_scored[:, 1]))
	elif method == 'mse':
		for feature in X_train.columns:
			clf = DecisionTreeRegressor()
			clf.fit(X_train[feature].to_frame(), y_train)
			y_scored = clf.predict(X_test[feature].to_frame())
			method_values.append(mean_squared_error(y_test, y_scored))
	else:
		raise ValueError('param method must be roc_auc or mse!')
	method_values = pd.Series(method_values)
	method_values.index = X_train.columns
	print(method_values.sort_values(ascending=False))
	keep_col = method_values[method_values > threshold]
	print(len(keep_col), 'out of the %s features are kept' % X_train.shape[1])
	return keep_col


def select_feature_with_mlxtend(X_train, y_train, estimator=RandomForestClassifier(n_jobs=-1, n_estimators=10), k=5, forward=True, selector='Sequential'):
	"""
	Sequential or Exhaustive Feature Selection for Classification and Regression
	:param X_train:
	:param y_train:
	:param estimator: scikit-learn classifier or regressor
	:param k:
	:param forward:
	:param selector:
	:return:
	"""
	from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector
	if selector == 'Sequential':
		ss = SequentialFeatureSelector(estimator=estimator,
									   k_features=k,
									   forward=forward,
									   floating=False,
									   verbose=1,
									   scoring='roc_auc',
									   cv=3)
		ss.fit(np.array(X_train), y_train)
		keep_col = X_train.columns[list(ss.k_feature_idx_)]
	elif selector == 'Exhaustive':
		ss = ExhaustiveFeatureSelector(estimator=estimator,
									   min_features=1,
									   max_features=6,
									   scoring='roc_auc',
									   print_progress=True,
									   cv=3)
		ss.fit(np.array(X_train), y_train)
		keep_col = X_train.columns[list(ss.best_idx_)]
	else:
		raise ValueError('param selector must be Sequential or Exhaustive!')
	return keep_col


def select_feature_with_lasso(X_train, y_train):
	"""
	通过lasso进行特征选择
	:param X_train:
	:param y_train:
	:return:
	"""
	select = SelectFromModel(Lasso())
	select.fit(X_train, y_train)
	keep_col = X_train.columns[select.get_support()]
	print(len(keep_col), ' features out of %s total features' % X_train.shape[1])
	return keep_col


def select_feature_with_rf_importance(X_train, y_train):
	"""
	通过随机森林进行特征选择
	:param X_train:
	:param y_train:
	:return:
	"""
	rf = RandomForestClassifier(random_state=0)
	rf.fit(X_train, y_train)
	# importance = rf.feature_importances_
	# indices = np.argsort(importance)[::-1]
	# columns = X_train.columns
	# # std = np.std([tree.feature_importance_ for tree in rf.estimators_], axis=0)
	# print("feature ranking: ")
	# for f in range(len(columns)):
	# 	print("%d. feature no:%d feature name:%s (%f)" % (f + 1, indices[f], columns[indices[f]], importance[indices[f]]))
	#
	# plt.figure()
	# plt.title('rf feature importance')
	# plt.bar(range(len(indices)), importance[indices], yerr=std[indices], align='center')
	# plt.xticks(range(len(indices)), indices)
	# plt.xlim([-1, len(indices)])
	# plt.show()
	select = SelectFromModel(rf, threshold=0.05, prefit=True)
	keep_col = X_train.columns[select.get_support()]
	return keep_col


def select_feature_with_gbdt_importance(X_train, y_train):
	"""
	通过gbdt进行特征选择
	:param X_train:
	:param y_train:
	:return:
	"""
	gbdt = GradientBoostingClassifier(random_state=0)
	gbdt.fit(X_train, y_train)
	select = SelectFromModel(gbdt, threshold=0.05, prefit=True)
	keep_col = X_train.columns[select.get_support()]
	return keep_col


def select_feature_with_shuffle(X_train, y_train, estimator):
	"""
	排列每个特征的值，一次一个，并测量排列降低了多少准确率，或roc auc，或机器学习模型的mse。如果变量很重要，那么它们的值的随机排列将显著减少这些指标。
	:param X_train:
	:param y_train:
	:param estimator:
	:return:
	"""
	estimator.fit(X_train, y_train)
	train_auc = roc_auc_score(y_train, estimator.predict_proba(X_train)[:, 1])
	feature_dict = {}
	
	for feature in X_train.columns:
		X_train_copy = X_train.copy(deep=True).reset_index(drop=True)
		y_train_copy = y_train.copy(deep=True).reset_index(drop=True)
		
		X_train_copy[feature] = X_train_copy[feature].sample(frac=1, random_state=0).reset_index(drop=True)
		
		shuffle_auc = roc_auc_score(y_train_copy, estimator.predict_proba(X_train_copy)[:,1])
		feature_dict[feature] = train_auc - shuffle_auc
		
	auc_drop = pd.Series(feature_dict).reset_index()
	auc_drop.columns = ['feature', 'auc_drop']
	auc_drop.sort_values(by='auc_drop', ascending=False, inplace=True)
	keep_col = auc_drop[auc_drop['auc_drop'] > 0]['feature']
	return auc_drop, keep_col


def select_feature_with_recursive_drop(X_train, y_train, X_test, y_test, threshold=0.001):
	"""
	后向算法，每次丢掉一个特征，看其auc损失程度
	:param X_train:
	:param y_train:
	:param X_test:
	:param y_test:
	:param threshold:
	:return:
	"""
	estimator = RandomForestClassifier()
	estimator.fit(X_train, y_train)
	y_pred_test = estimator.predict_proba(X_test)[:, 1]
	test_auc = roc_auc_score(y_test, y_pred_test)
	
	feature_to_remove = list()
	count = 1
	for feature in X_train.columns:
		print('test feature: {}, which is {} out of {} features'.format(feature, count, X_train.shape[1]))
		count += 1
		
		estimator_ = RandomForestClassifier()
		estimator_.fit(X_train.drop(feature_to_remove+[feature], axis=1), y_train)
		y_pred_test = estimator_.predict_proba(X_test.drop(feature_to_remove+[feature], axis=1))[:, 1]
		test_auc_ = roc_auc_score(y_test, y_pred_test)
		if test_auc - test_auc_ > threshold:
			print('keep')
		else:
			print('remove, diff auc: {}'.format(test_auc-test_auc_))
			test_auc = test_auc_
			feature_to_remove.append(feature)
		
	print('done')
	print('feature to remove {}: {}'.format(len(feature_to_remove), feature_to_remove))
	keep_col = [x for x in X_train.columns if x not in feature_to_remove]
	return keep_col


def select_feature_with_recursive_add(X_train, y_train, estimator):
	pass


if __name__ == '__main__':
	from sklearn.datasets import load_breast_cancer
	
	data = load_breast_cancer()
	data = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(data['feature_names'], ['target']))
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['target'], axis=1), data.target, test_size=0.2, random_state=0)
	print(X_train.shape, X_test.shape)

	# print(detect_constant_feature(X_train, threshold=0.4))
	# print(detect_correlation_feature(X_train))
	# print(select_feature_with_chi2_or_mutual(X_train, y_train, k=3))
	# print(select_feature_with_chi2_or_mutual(X_train, y_train, k=0.2))
	#
	# print(select_feature_with_chi2_or_mutual(X_train, y_train, k=3, score_func=chi2))
	# print(select_feature_with_chi2_or_mutual(X_train, y_train, k=0.2, score_func=chi2))
	#
	# print(select_feature_with_decision_tree(X_train, y_train, X_test, y_test, threshold=0.8))
	# print(select_feature_with_decision_tree(X_train, y_train, X_test, y_test, threshold=0.4, method='mse'))
	#
	# print(select_feature_with_mlxtend(X_train, y_train))
	# print(select_feature_with_mlxtend(X_train, y_train, forward=False))
	# print(select_k_features_with_mlxtend(X_train, y_train, selector='Exhaustive'))
	# print(select_feature_with_lasso(X_train, y_train))
	# print(select_feature_with_rf_importance(X_train, y_train))
	# print(select_feature_with_gbdt_importance(X_train, y_train))
	# print(select_feature_with_shuffle(X_train, y_train, RandomForestClassifier(random_state=0)))
	print(select_feature_with_recursive_drop(X_train, y_train, X_test, y_test))
