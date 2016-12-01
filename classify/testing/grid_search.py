from sklearn import svm, grid_search
import numpy as np

data = np.loadtxt("DB/fv_all.list", unpack=True)
target = np.loadtxt("DB/ev.score", unpack=True)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 32000], 'gamma':[0, 10]}
svr = svm.SVR()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(data.T, target.T)

print clf.get_params()
print clf.score(data.T, target.T)
print clf.best_params_
