import sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import math

LUMINANCE = False
ROUND_OFF = False

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
    gamma_range = 2. ** np.array([-15 ,-13, -11, -9, -7, -5, -3, -1, 1, 3, 5])
    epsilon_range= [0, 0.01, 0.1, 0.5, 1, 2, 4]

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.1, random_state=0)

    parameters = {'C':C_range, 'gamma':gamma_range, 'epsilon':epsilon_range}

    svr = svm.SVR(tol=0.00000001)
    clf = grid_search.GridSearchCV(svr, parameters, cv=cv, scoring='r2', n_jobs=-1)
    clf.fit(X, Y)
    
    return clf

def scale_data_testing(data, target):

    # scales = np.array([7, 7, 7, 5, 4, 4, 2, 5, 3, 1, 1, 1, 1])
    scales = np.array([5, 9, 5, 5, 1, 1, 4, 7, 3, 3, 1, 1, 3])
    # scales = np.ones(13)
    # scaling different features differently to add weight to features
    # feature_scalar = preprocessing.StandardScaler()
    # feature_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #X = feature_scalar.fit_transform(data)
    #np.savetxt("fv_1.list", X, fmt='%0.6f')
    
    print data.shape
    X = []
    feature_scalar = []
    flag = 0
    for i in range(0,13):
        X0 = data[:, i:i+1]
        # my_scaler = preprocessing.MinMaxScaler(feature_range=(-scales[i], scales[i]))
        # target_scalar = preprocessing.StandardScaler()
        my_scaler = preprocessing.MinMaxScaler(feature_range=(-scales[i], scales[i]))
        feature_scalar.append(my_scaler)
        X1 = my_scaler.fit_transform(X0)
        if flag==0 :
            X = X1
            flag = 1
        else:
            X = np.hstack([X, X1])

    print X.shape
    
    X = preprocessing.normalize(X, norm='l2')
    np.savetxt("fv_2.list", X, fmt='%0.6f')

    # target_scalar = preprocessing.StandardScaler()
    target_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    Y_temp = np.reshape(target, (-1, 1))
    Y = target_scalar.fit_transform(Y_temp)
    Y = np.squeeze(np.asarray(Y))
    np.savetxt("cam_scale.list", Y, fmt='%0.6f')
    
    return X, Y, feature_scalar, target_scalar


def scale_data(data, target):

    # scaling different features differently to add weight to features
    # feature_scalar = preprocessing.StandardScaler()
    # feature_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #X = feature_scalar.fit_transform(data)
    #np.savetxt("fv_1.list", X, fmt='%0.6f')
    
    print data.shape
    X1 = data[:, 0:3]
    X2 = data[:, 3:6]
    X3 = data[:, 6:9]
    X4 = data[:, 9:13]

    feature_scalar = []
    feature_scalar1 = preprocessing.MinMaxScaler(feature_range=(-6, 6))
    feature_scalar2 = preprocessing.MinMaxScaler(feature_range=(-3, 3))
    feature_scalar3 = preprocessing.MinMaxScaler(feature_range=(-3, 3))
    feature_scalar4 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    feature_scalar.append(feature_scalar1)
    feature_scalar.append(feature_scalar2)
    feature_scalar.append(feature_scalar3)
    feature_scalar.append(feature_scalar4)

    X1 = feature_scalar1.fit_transform(X1)
    X2 = feature_scalar2.fit_transform(X2)
    X3 = feature_scalar3.fit_transform(X3)
    X4 = feature_scalar4.fit_transform(X4)

    X = np.hstack([X1, X2, X3, X4])
    print X.shape
    
    X = preprocessing.normalize(X, norm='l2')
    np.savetxt("fv_2.list", X, fmt='%0.6f')

    # target_scalar = preprocessing.StandardScaler()
    target_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    Y_temp = np.reshape(target, (-1, 1))
    Y = target_scalar.fit_transform(Y_temp)
    Y = np.squeeze(np.asarray(Y))
    np.savetxt("cam_scale.list", Y, fmt='%0.6f')
    
    return X, Y, feature_scalar, target_scalar


def run_cross_validation(X, Y):

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=0.1, random_state=0)
    regressor = svm.SVR(C=2, gamma=32, epsilon =0.0, tol=0.000001)
    # regressor = svm.SVR(C=8, gamma=8, epsilon =0.01, tol=0.000001)
    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='mean_squared_error')

    print "Mean Square Error : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='r2')

    print "R2 Score : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


def ev_to_luminance(Y):
    n_samples = len(Y)
    
    L = np.zeros(n_samples)

    for i in range(n_samples):
        L[i] = math.pow(2, Y[i] - 3)

    np.savetxt("luminance.list", L, fmt='%.6f')

    return L

def luminance_to_ev(L):    
    ev_score = math.log(L, 2) + 3

    return ev_score


def wrapper_luminance_to_ev(L):
    n_samples = len(L)

    EV = np.zeros(n_samples)

    for i in range(n_samples):
        print L[i]
        EV[i] = math.log(L[i], 2) + 3

    return EV

def scale_sample(data, scalar):

    X = []
    flag = 0
    for i in range(0,13):
        X0 = data[i:i+1]
        X1 = scalar[i].transform(X0)
        if flag==0 :
            X = X1
            flag = 1
        else:
            X = np.hstack([X, X1])

    return X

def learn_camera_parameters(f_file, v_file):
    # load features
    data = np.loadtxt(f_file, unpack=True)
    target = np.loadtxt(v_file, unpack=True)
    
    # convert to integer ev values
    if ROUND_OFF == True:
        target = np.matrix.round(target)

    # convert ev score to luminance
    if LUMINANCE == True :
        target = ev_to_luminance(target)

    X, Y, feature_scalar, target_scalar = scale_data_testing(data.T, target)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)

    # clf = run_grid_search(X_train, Y_train)

    # generate_report(clf, X_test, Y_test)

    run_cross_validation(X_train, Y_train)

    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    # regressor = svm.SVR(C=8, gamma=8, epsilon =0.01, tol=0.000001)
    regressor = svm.SVR(C=2, gamma=32, epsilon =0.0, tol=0.000001)
    regressor.fit(X_train, Y_train)
    print regressor.score(X_test, Y_test)

    Y_pred = regressor.predict(X_test)
    
    print "Variance Score : ", explained_variance_score(Y_test, Y_pred)
    print "Mean Absolute Error : ", mean_absolute_error(Y_test, Y_pred)
    print "Mean Squared Error : ", mean_squared_error(Y_test, Y_pred)
    print "R2 Score: ", r2_score(Y_test, Y_pred)

    Y_0 = target_scalar.inverse_transform(Y_test)
    Y_1 = target_scalar.inverse_transform(Y_pred)
    print ['{:.6f}' .format(i) for i in Y_0]
    print ['{:.6f}' .format(i) for i in Y_1]

    print "R2 Score: ", r2_score(Y_0, Y_1)
    print "Mean Squared Error : ", mean_squared_error(Y_0, Y_1)

    if LUMINANCE == True:
        Y_0 = wrapper_luminance_to_ev(Y_0)
        Y_1 = wrapper_luminance_to_ev(Y_1)

        print "R2 Score: ", r2_score(Y_0, Y_1)
        print "Mean Squared Error : ", mean_squared_error(Y_0, Y_1)

    Y_01 = np.matrix.round(Y_0)
    Y_11 = np.matrix.round(Y_1)

    print "R2 Score: ", r2_score(Y_01, Y_11)
    print "Mean Squared Error : ", mean_squared_error(Y_01, Y_11)

    np.savetxt("y0.list", Y_0, fmt='%.6f')
    np.savetxt("y01.list", Y_01, fmt='%.6f')
    np.savetxt("y1.list", Y_1, fmt='%.6f')
    np.savetxt("y11.list", Y_11, fmt='%.6f')

    # testing
    # sample = np.array([4, -7.5, -2, 6, 0, 0, 0, 0, 0, .88, .77, 0, .77])
    sample = np.array([4.38333, -7.6833, -1.65, 6.2, 0, 0, 0, 0, 5, 0.878, 0.77, 0, 0.7])
    sample = scale_sample(sample, feature_scalar)
    sample_pred = regressor.predict(sample)
    sample_pred = target_scalar.inverse_transform(sample_pred)
    print sample_pred
    print luminance_to_ev(sample_pred)


learn_camera_parameters(sys.argv[1], sys.argv[2])

