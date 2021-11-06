import pandas as pd 
import numpy as np 

from sklearn import ensemble 
from sklearn import metrics 
from sklearn import model_selection 
from sklearn import preprocessing, decomposition, pipeline  

from functools import partial  
from skopt import space, gp_minimize  
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope  

def optimize(params, x, y):  
	model = ensemble.RandomForestClassifier(**params)  
	kf = model_selection.StratifiedKFold(n_splits=5)   
	accuracies = []  
	for idx in kf.split(X=x, y=y):  
		train_idx, test_idx = idx[0], idx[1]   
		xtrain, ytrain = x[train_idx], y[train_idx] 
		xtest, ytest = x[test_idx], y[test_idx]   
		model.fit(xtrain, ytrain)  
		preds = model.predict(xtest)  
		fold_acc = metrics.accuracy_score(ytest, preds)  
		accuracies.append(fold_acc)  

	return np.mean(accuracies) * (-1)  

if __name__ == '__main__': 
	df = pd.read_csv('train.csv')  
	X = df.drop(columns=['price_range']).values 
	y = df.price_range.values  


	param_space = {
		"max_depth": hp.quniform("max_depth", 3, 15, 1), 
		"n_estimators":scope.int(hp.quniform("n_estimators", 100, 600, 1)),   
		"criterion":hp.choice('criterion', ['gini', 'entropy']),  
		"max_features":hp.uniform('max_features', 0.01, 1),   
	}
	
	optimization_function = partial(optimize, x=X, y=y)  

	trials = Trials()  

	result = fmin(
		fn = optimization_function,  
		space=param_space,  
		max_evals=15,
		trials=trials, 
		algo=tpe.suggest,
		)
	print(result) 
 
