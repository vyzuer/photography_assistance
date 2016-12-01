import sys, os
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import math

def generate_report(clf, X_test, Y_test):
    
    print clf.get_params()
    print clf.score(X_test, Y_test)
    print clf.best_params_
    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    # print("Grid scores on development set:")
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() / 2, params))


def run_grid_search(X, Y):
    print("# Tuning hyper-parameters") 

    C_range = 2. ** np.array([-5,-3,-1, 1, 3, 5, 7, 9, 11, 13, 15 ,17])
    gamma_range = 2. ** np.array([-15 ,-13, -11, -9, -7, -5, -3, -1, 1, 3, 5])
    epsilon_range= [0, 0.01, 0.1, 0.5, 1, 2, 4]

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.2, random_state=0)

    parameters = {'kernel':['rbf'], 'C':C_range, 'gamma':gamma_range, 'epsilon':epsilon_range}

    svr = svm.SVR(tol=0.00000001)
    clf = grid_search.GridSearchCV(svr, parameters, cv=cv, scoring='r2', n_jobs=-1)
    clf.fit(X, Y)
    
    return clf


def scale_data(data, target):

    feature_scalar = preprocessing.StandardScaler()
    # feature_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = feature_scalar.fit_transform(data)
    # np.savetxt("fv_1.list", X, fmt='%0.6f')
    
    # X = data
    X = preprocessing.normalize(X, norm='l2')
    np.savetxt("fv_2.list", X, fmt='%0.6f')

    # target_scalar = preprocessing.StandardScaler()
    target_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    Y_temp = np.reshape(target, (-1, 1))
    Y = target_scalar.fit_transform(Y_temp)
    Y = np.squeeze(np.asarray(Y))
    np.savetxt("a_score.list", Y, fmt='%0.6f')
    
    return X, Y, feature_scalar, target_scalar


def run_cross_validation(X, Y):

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.3, random_state=0)
    regressor = svm.SVR()
    # regressor = svm.SVR(C=8, gamma=32, epsilon =0.01, tol=0.000001)
    # regressor = svm.SVR(C=8, gamma=8, epsilon =0.01, tol=0.000001)
    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='mean_squared_error')

    print "Mean Square Error : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='r2')

    print "R2 Score : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


def read_features(db, location, dump_dir, comp=True, view=True, geo=True, time=False):
    # db = "/home/yogesh/Project/Flickr-YsR/"
    db_path = db + location
    dump_path = db_path + dump_dir

    f_comp = dump_path + "/feature.fv"

    f_view = dump_path + "/view.fv"

    f_geo = db_path + "/geo.info"

    f_time = db_path + "/weather.info"

    f_target = db_path + "/aesthetic.scores"

    X = np.loadtxt(f_comp)

    if view == True :
        X1 = np.loadtxt(f_view)
        X = np.hstack([X, X1])

    if geo == True :
        X1 = np.loadtxt(f_geo)
        X = np.hstack([X, X1])

    target = np.loadtxt(f_target)


    return X, target


def learn_composition(db, location, dump_dir, grid_search=False):
    # load features
    data, target = read_features(db, location, dump_dir)
    
    X, Y, feature_scalar, target_scalar = scale_data(data, target)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

    # clf = run_grid_search(X_train, Y_train)

    # generate_report(clf, X_test, Y_test)

    run_cross_validation(X_train, Y_train)

    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    # regressor = svm.SVR(C=8, gamma=8, epsilon =0.01, tol=0.000001)
    regressor = svm.SVR()
    regressor.fit(X_train, Y_train)
    print regressor.score(X_train, Y_train)
    print regressor.score(X_test, Y_test)


if __name__ == "__main__":

    # feature_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/composition.fv"
    # a_score_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/aesthetic.scores"

    db = "/home/yogesh/Project/Flickr-YsR/"
    # location = "/esplanade/"
    location = "/floatMarina/"
    # location = "/merlionImages/"
    dump_dir = "/dump_ImageDB_640/"
    # dump_dir = "/dump_Image_640/"

    learn_composition(db, location, dump_dir, grid_search=True)

