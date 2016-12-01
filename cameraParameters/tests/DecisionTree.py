import sys
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import cross_validation

def learnCameraParameters(f_file, v_file):
    # load features
    X = np.loadtxt(f_file, unpack=True)
    X = np.transpose(X)
    print X.shape
    min_max_scaler1 = preprocessing.MinMaxScaler()
    X = min_max_scaler1.fit_transform(X)
    np.savetxt("fv_scale.list", X, fmt='%0.6f')

    X = np.transpose(X)
    scaler1 = preprocessing.StandardScaler()
    X = scaler1.fit_transform(X)
    X = np.transpose(X)
    scaler2 = preprocessing.StandardScaler()
    X = scaler2.fit_transform(X)

    y = np.loadtxt(v_file, unpack=True)
    # y = np.transpose(y)
    # min_max_scaler2 = preprocessing.MinMaxScaler()
    # y = min_max_scaler2.fit_transform(y)
    # np.savetxt("cam_scale.list", y, fmt='%0.6f')

    # y = np.transpose(y)
    scaler3 = preprocessing.StandardScaler()
    y = scaler3.fit_transform(y)
    y = np.transpose(y)

    scaler4 = preprocessing.StandardScaler()
    y = scaler4.fit_transform(y)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)
    regressor = DecisionTreeRegressor()   
    # regressor = DecisionTreeRegressor(max_features='log2', min_samples_split=20, min_samples_leaf=20, random_state=71)   
    # scores = cross_validation.cross_val_score(regressor, X, y, cv=4)
    regressor.fit(X_train, y_train)
    X_new = regressor.fit_transform(X_train, y_train)
    regressor2 = DecisionTreeRegressor()   
    regressor2.fit(X_new, y_train)
    print regressor2.score(X_new, y_train)
    X_new = regressor.transform(X_test)
    print regressor2.score(X_new, y_test)
    yy = regressor.predict(X)
    # print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    y = scaler4.inverse_transform(yy)
    y = np.transpose(y)
    y = scaler3.inverse_transform(y)
    y = np.transpose(y)
    # y = min_max_scaler2.inverse_transform(y)
    np.savetxt("cam_back.list", y, fmt='%0.6f')


learnCameraParameters(sys.argv[1], sys.argv[2])

