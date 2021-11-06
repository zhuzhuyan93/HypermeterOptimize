import pandas as pd 
import numpy as np 

from sklearn import ensemble 
from sklearn import metrics 
from sklearn import model_selection 
from sklearn import preprocessing, decomposition, pipeline 


def optimize(params, param_names, x, y):  
	params = dict(zip(param_names, params))  
	model = ensemble.RandomForestClassifier(**params)  
	kf = model_selection.StratifiedKFold(n_splits=5)   
	accuracies = []  
	for idx in kf.split(X=x, y=x):  
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

	scl = preprocessing.StandardScaler()  
	pca = decomposition.PCA()  
	rf = ensemble.RandomForestClassifier(n_jobs=-1)   


	classifier = pipeline.Pipeline([("scaling", scl), ("pca", pca), ("rf", rf)])
	param_grid = {
		"pca__n_components":np.arange(5, 10), 
		"rf__n_estimators" : np.arange(100, 1500, 100),
		"rf__max_depth" : np.arange(1, 20, 1), 
		"rf__criterion" : ['gini', 'entropy'],
	}  

	model = model_selection.RandomizedSearchCV(
		estimator=classifier, 
		param_distributions= param_grid, 
		n_iter=10,
		scoring='accuracy', 
		verbose=10, 
		n_jobs=1, 
		cv=5, 
		)

	model.fit(X, y)  

	print(model.best_score_)  
	print(model.best_estimator_.get_params()) 