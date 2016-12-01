import sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

def generate_report(clf, X_test, Y_test):
    
    print clf.get_params()
    print clf.score(X_test, Y_test)
    print clf.best_params_
    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))


def run_grid_search(X, Y):
    print("# Tuning hyper-parameters") 

    C_range = 2. ** np.array([-5,-3,-1, 1, 3, 5, 7, 9, 11, 13, 15 ,17])
    gamma_range = 2. ** np.array([-15 ,-13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9])

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.1, random_state=0)

    parameters = {'kernel':['rbf'], 'C':C_range, 'gamma':gamma_range}

    svr = svm.SVC(tol=0.00000001)
    clf = grid_search.GridSearchCV(svr, parameters, cv=cv, n_jobs=-1)
    clf.fit(X, Y)
    
    return clf


def scale_data(data):

    feature_scalar = preprocessing.StandardScaler()
    # feature_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = feature_scalar.fit_transform(data)
    # np.savetxt("fv_1.list", X, fmt='%0.6f')
    
    # X = np.round(data)
    X = preprocessing.normalize(X, norm='l2')
    # np.savetxt("fv_2.list", X, fmt='%0.6f')

    return X, feature_scalar


def run_cross_validation(X, Y):

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=0.2, random_state=0)
    # regressor = svm.SVC(C=8, gamma=8)
    regressor = svm.SVC(kernel= 'rbf', C=32, gamma=8)
    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv)

    print "Score : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


def remove_average_photographs(data, target):
    x, y = data.shape

    X = []
    Y = []

    for i in range(x):
        if target[i] < 0.30:
            Y.append(0)
            X.append(data[i])

        elif target[i] > 0.70:
            Y.append(1)
            X.append(data[i])

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y

def learn_composition(f_file, v_file):
    # load features
    data = np.loadtxt(f_file)
    target = np.loadtxt(v_file)
    # target = target*5
    # Y = np.matrix.round(target)

    X, Y = remove_average_photographs(data, target)
    print X.shape
    
    # X, feature_scalar = scale_data(X)
    # X = data

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

    # clf = run_grid_search(X_train, Y_train)

    # generate_report(clf, X_test, Y_test)

    run_cross_validation(X_train, Y_train)

    regressor = svm.SVC(kernel= 'rbf', C=2, gamma=1, probability=True)
    regressor.fit(X_train, Y_train)
    print regressor.score(X_train, Y_train)
    # print regressor.predict_proba(X_train)
    print regressor.score(X_test, Y_test)
    Pr = regressor.predict(X_test)
    print 'Accuracy : {0}'.format(accuracy_score(Y_test, Pr))
    print 'Average Precision : {0}'.format(average_precision_score(Y_test, Pr))
    print 'F1 Score : {0}'.format(f1_score(Y_test, Pr))
    print 'Precision : {0}'.format(precision_score(Y_test, Pr))
    print 'Recall : {0}'.format(recall_score(Y_test, Pr))
    print 'ROC : {0}'.format(roc_auc_score(Y_test, Pr))
    print regressor.predict_proba(X_test)
    result =  regressor.predict_proba(data)
    np.savetxt("prob.list", result, fmt='%.6f')

    print precision_recall_fscore_support(Y_test, Pr)

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm="SAMME", n_estimators=200)
    # # clf = AdaBoostClassifier(svm.SVC(kernel= 'rbf', C=32, gamma=32), algorithm="SAMME", n_estimators=200)
    # scores = cross_validation.cross_val_score(clf, X, Y)
    # print scores
    # print scores.mean()

    # # clf = RandomForestClassifier(n_estimators=50)
    # clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    # scores = cross_validation.cross_val_score(clf, X, Y)
    # print scores
    # print scores.mean()


if __name__ == "__main__":

    feature_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/composition.fv"
    # feature_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/dump_ImageDB_640/composition_full_avg.fv"
    a_score_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/aesthetic.scores"

    learn_composition(feature_file, a_score_file)

