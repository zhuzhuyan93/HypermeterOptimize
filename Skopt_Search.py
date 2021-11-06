import pandas as pd 
import numpy as np 

from sklearn import ensemble 
from sklearn import metrics 
from sklearn import model_selection 
from sklearn import preprocessing, decomposition, pipeline  

from functools import partial  
from skopt import space, gp_minimize

def optimize(params, param_names, x, y):  
	params = dict(zip(param_names, params))  
	print(params) 
	model = ensemble.RandomForestClassifier(**params)  
	kf = model_selection.StratifiedKFold(n_splits=5)   
	accuracies = []  
	for idx in kf.split(X=x, y=y):  
		train_idx, test_idx = idx[0], idx[1]   
		xtrain, ytrain = x[train_idx], y[train_idx] 
		xtest, ytest = x[test_idx], y[test_idx]   
		# print(xtrain, '#'*30, xtest)
		model.fit(xtrain, ytrain)  
		preds = model.predict(xtest)  
		fold_acc = metrics.accuracy_score(ytest, preds)  
		accuracies.append(fold_acc)  

	return np.mean(accuracies) * (-1)  

if __name__ == '__main__': 
	df = pd.read_csv('train.csv')  
	X = df.drop(columns=['price_range']).values 
	y = df.price_range.values  

	param_space = [
		space.Integer(3, 15, name='max_depth'),  
		space.Integer(100, 600, name='n_estimators'),  
		space.Categorical(['gini', 'entropy'], name='criterion'), 
		space.Real(0.01, 1, prior='uniform',name='max_features'),  
	]  
	param_names = [
		"max_depth",  
		'n_estimators',  
		'criterion',  
		'max_features',  
	]
	optimization_function = partial(optimize, param_names=param_names, x=X, y=y)  

	result = gp_minimize(
		optimization_function,  
		dimensions=param_space,  
		n_calls=15,  
		n_random_starts=10,  
		verbose=10,   
		)
	print(dict(zip(param_names, result.x))) 
 
