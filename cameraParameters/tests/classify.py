import sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search


def learnCameraParameters(f_file, v_file):
    # load features
    X = np.loadtxt(f_file, unpack=True)
    X = np.transpose(X)
    print X.shape

    # min_max_scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # X = min_max_scaler1.fit_transform(X)
    # np.savetxt("fv_scale.list", X, fmt='%0.6f')

    scaler1 = preprocessing.StandardScaler()
    X = scaler1.fit_transform(X)
    np.savetxt("fv_1.list", X, fmt='%0.6f')

    X = preprocessing.normalize(X, norm='l2')
    np.savetxt("fv_2.list", X, fmt='%0.6f')

    Y = np.loadtxt(v_file, unpack=True)
    # print Y.shape
        

    # min_max_scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # Y_temp = np.reshape(Y, (-1, 1))
    # Y = min_max_scaler2.fit_transform(Y_temp)
    # Y = np.squeeze(np.asarray(Y))

    scaler2 = preprocessing.StandardScaler()
    Y = scaler2.fit_transform(Y)
    np.savetxt("cam_scale.list", Y, fmt='%0.6f')

    X_new = X[:, 0:13]
    # X_new = X
    # print X_new
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_new, Y, test_size=0.05, random_state=0)

    # C_range = 2. ** np.array([-5,-3,-1, 1, 3, 5, 7, 9, 11, 13, 15 ,17])
    # gamma_range = 2. ** np.array([-15 ,-13, -11, -9, -7, -5, -3, -1, 1, 3, 5])
    # epsilon_range= [0, 0.01, 0.1, 0.5, 1, 2, 4]

    # parameters = {'C':C_range, 'gamma':gamma_range, 'epsilon':epsilon_range}
    # svr = svm.SVR()
    # clf = grid_search.GridSearchCV(svr, parameters)
    # clf.fit(X, Y)
    # 
    # print clf.get_params()
    # print clf.score(X_test, Y_test)
    # print clf.best_params_

    regressor = svm.SVR(C=32, gamma=0, epsilon =.3, tol=.000001)
    # regressor = svm.SVR()
    scores = cross_validation.cross_val_score(regressor, X_new, Y, cv=10, scoring='mean_squared_error')

    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    scores = cross_validation.cross_val_score(regressor, X_new, Y, cv=10, scoring='r2')

    print scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    regressor = svm.SVR(C=32, gamma=0, epsilon =.9, tol=.000001)
    # regressor = svm.SVR(C=.125, gamma=0.001953125, epsilon =0.50, tol=.000001)
    # regressor = svm.SVR()
    regressor.fit(X_train, Y_train)
    print regressor.score(X_test, Y_test)
    print regressor.predict(X_test)
    print Y_test

    #y = scaler4.inverse_transform(yy)
    #y = np.transpose(y)
    #y = scaler3.inverse_transform(y)
    #y = np.transpose(y)
    ## y = min_max_scaler2.inverse_transform(y)
    #np.savetxt("cam_back.list", y, fmt='%0.6f')


learnCameraParameters(sys.argv[1], sys.argv[2])

