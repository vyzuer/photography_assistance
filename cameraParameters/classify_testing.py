import sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


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


def scale_data_testing(data, target, scales):

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
    feature_scalar2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    feature_scalar3 = preprocessing.MinMaxScaler(feature_range=(-9, 9))
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
    regressor = svm.SVR(C=8, gamma=32, epsilon =0.01, tol=0.000001)
    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='mean_squared_error')

    print "Mean Square Error : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='r2')

    print "R2 Score : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)



def learn_camera_parameters(f_file, v_file, scales):
    # load features
    data = np.loadtxt(f_file, unpack=True)
    target = np.loadtxt(v_file, unpack=True)

    X, Y, feature_scalar, target_scalar = scale_data_testing(data.T, target, scales)
    # X, Y, feature_scalar, target_scalar = scale_data(data.T, target)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)

    # clf = run_grid_search(X_train, Y_train)

    # generate_report(clf, X_test, Y_test)

    run_cross_validation(X_train, Y_train)

    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    regressor = svm.SVR(C=8, gamma=32, epsilon =0.01, tol=0.000001)
    regressor.fit(X_train, Y_train)
    print regressor.score(X_test, Y_test)

    Y_pred = regressor.predict(X_test)
    
    print "Variance Score : ", explained_variance_score(Y_test, Y_pred)
    print "Mean Absolute Error : ", mean_absolute_error(Y_test, Y_pred)
    print "Mean Squared Error : ", mean_squared_error(Y_test, Y_pred)
    print "R2 Score: ", r2_score(Y_test, Y_pred)

    #y = scaler4.inverse_transform(yy)
    #y = np.transpose(y)
    #y = scaler3.inverse_transform(y)
    #y = np.transpose(y)
    ## y = min_max_scaler2.inverse_transform(y)
    #np.savetxt("cam_back.list", y, fmt='%0.6f')


for i in range(5, 11, 2):
    for j in range(5, 11, 2):
        for k in range(5, 11, 2):
            for l in range(1, 6, 2):
                for o in range(1, 11, 3):
                    for p in range(1, 11, 3):
                        scales = []
                        for x in [i,j,k,l,1,1,o,p,3,3,1,1,3]:
                            scales.append(x)
                        print np.asarray(scales)
                        learn_camera_parameters(sys.argv[1], sys.argv[2], np.asarray(scales))

