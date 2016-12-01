import sys, os
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import math
import time
from sklearn.externals import joblib

LUMINANCE = False
ROUND_OFF = False

def generate_report(clf, X_test, Y_test, fp):
    
    print >> fp, clf.get_params()
    print >> fp, clf.score(X_test, Y_test)
    print >> fp, clf.best_params_
    print >> fp, ("Best parameters set found on development set:")
    print >> fp, (clf.best_estimator_)
    print >> fp, ("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print >> fp, ("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

    return clf.best_params_['C'], clf.best_params_['gamma'], clf.best_params_['epsilon']


def run_grid_search(X, Y):
    print("# Tuning hyper-parameters") 

    # C_range = 2. ** np.array([-3, -2, 0, 1])
    # gamma_range = 2. ** np.array([0, 1, 2, 3])
    # # epsilon_range = 2. ** np.array([-15, -13, -11 ])
    # epsilon_range= [0.0001, 0.005, 0.01, 0.1]

    C_range = [0.001, 0.01, 0.25, 0.5, 0.75, 1, 1.5, 2]
    gamma_range = [0.1, 0.5, 1, 1.5, 2, 2.5, 3]
    epsilon_range= [0.000001, 0.00001, 0.0001, 0.01, 0.1, 0.5, 1]

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.2, random_state=777)

    parameters = {'C':C_range, 'gamma':gamma_range, 'epsilon':epsilon_range}

    svr = svm.SVR(kernel='rbf', tol=0.0000000001)
    clf = grid_search.GridSearchCV(svr, parameters, cv=cv, scoring='r2', n_jobs=5)
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

    feature_scalar = preprocessing.StandardScaler()
    # feature_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = feature_scalar.fit_transform(data)
    # np.savetxt("fv_1.list", X, fmt='%0.6f')
    
    # X = np.round(data)
    X = preprocessing.normalize(X, norm='l2')
    # np.savetxt("fv_2.list", X, fmt='%0.6f')

    target = np.log2(target)
    # target_scalar = preprocessing.StandardScaler()
    target_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    Y_temp = np.reshape(target, (-1, 1))
    Y = target_scalar.fit_transform(Y_temp)
    Y = np.squeeze(np.asarray(Y))
    
    return X, Y, feature_scalar, target_scalar


def scale_data_0(data, target):

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


def run_cross_validation(X, Y, C, gamma, epsilon, fp):

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.2, random_state=777)
    regressor = svm.SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon, tol=0.000001)
    # regressor = svm.SVR(C=8, gamma=8, epsilon =0.01, tol=0.000001)
    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='mean_squared_error')

    print "Mean Square Error : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    print >> fp, "Mean Square Error : ", scores
    print >> fp, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv, scoring='r2')

    print "R2 Score : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    print >> fp, "R2 Score : ", scores
    print >> fp, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


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

def read_features(db_path, dump_dir, env=True, view=False, geo=False):
    # db = "/home/yogesh/Project/Flickr-YsR/"
    dump_path = db_path + dump_dir

    f_view = dump_path + "/view_1.list"

    f_face = dump_path + "/face_1.list"

    f_geo = db_path + "/geo_1.info"

    f_env = db_path + "/weather_1.info"

    f_score = db_path + "/aesthetic_1.scores"

    f_ev = db_path + "/ev.score"

    f_target = db_path + "/camera_1.settings"

    X = np.loadtxt(f_env, delimiter=' ', dtype=np.float)

    # pca1 = PCA(n_components=10)
    # X = pca1.fit_transform(X)

    X_0 = np.loadtxt(f_face, delimiter=' ', dtype=np.float)

    X = np.hstack([X, X_0])

    X_1 = np.loadtxt(f_ev)
    X_1 = np.reshape(X_1, (-1, 1))

    X = np.hstack([X, X_1])

    pca = PCA(n_components=20)
    if view == True :
        X1 = np.loadtxt(f_view)

        X2 = pca.fit_transform(X1)
        
        X = np.hstack([X, X2])

    if geo == True :
        X1 = np.loadtxt(f_geo)
        X = np.hstack([X, X1])

    target = np.loadtxt(f_target)[:,1:2]

    a_score = np.loadtxt(f_score)

    return X, target, a_score, pca

    
def remove_bad_photographs(data, target, a_score):
    x, y = data.shape

    X = []
    Y = []

    for i in range(x):
        if a_score[i] > 0.20:
            Y.append(target[i])
            X.append(data[i])

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y

def read_params(f_name):
    params = [32.0, 8.0, 0.01]

    if os.path.exists(f_name):
        params = np.loadtxt(f_name)
    
    return params[0], params[1], params[2]
    

def dump_params(f_name, C, gamma, epsilon):
    fp = open(f_name, 'w')
    fp.write('{0:0.8f} {1:0.8f} {2:0.8f}'.format(C, gamma, epsilon))
    fp.close()


def learn_camera_parameters(db, dump_dir, grid_search=False):
    results_dir = db + "results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    f_result = results_dir + "/res_ap." + str(time.time())

    params_dir = db + "params/"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    f_name = params_dir + "/.ap.params"

    model_dir = db + "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    comp_model_dir = model_dir + "ap/"
    if not os.path.exists(comp_model_dir):
        os.makedirs(comp_model_dir)

    dump_f_scalar = comp_model_dir + "/f_scalar_ap.pkl"
    dump_t_scalar = comp_model_dir + "/t_scalar_ap.pkl"
    dump_pca = comp_model_dir + "/pca_ap.pkl"
    dump_svr = comp_model_dir + "/svr_ap.pkl"

    print f_result
    fp = open(f_result, 'w')
    # load features
    data, target, a_score, pca = read_features(db, dump_dir, geo=True, view=True)
    
    X, Y = remove_bad_photographs(data, target, a_score)
    print X.shape

    # convert to integer ev values
    if ROUND_OFF == True:
        target = np.matrix.round(target)

    # convert ev score to luminance
    if LUMINANCE == True :
        target = ev_to_luminance(target)

    X, Y, feature_scalar, target_scalar = scale_data(X, Y)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=777)

    # C = 32
    # gamma = 8
    # epsilon = 0.01
    if grid_search == False:
        C, gamma, epsilon = read_params(f_name)
        # C, gamma, epsilon = 1, 4, 0.1
    else:
        clf = run_grid_search(X_train, Y_train)
        C, gamma, epsilon = generate_report(clf, X_test, Y_test, fp)
        dump_params(f_name, C, gamma, epsilon)

    run_cross_validation(X_train, Y_train, C, gamma, epsilon, fp)

    # regressor = svm.SVR(C=32, gamma=0, epsilon =0.1, tol=0.000001)
    # regressor = svm.SVR(C=8, gamma=8, epsilon =0.01, tol=0.000001)
    regressor = svm.SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon, tol=0.000001)
    regressor.fit(X_train, Y_train)
    print regressor.score(X_test, Y_test)

    Y_pred = regressor.predict(X_test)
    
    print "Variance Score : ", explained_variance_score(Y_test, Y_pred)
    print "Mean Absolute Error : ", mean_absolute_error(Y_test, Y_pred)
    print "Mean Squared Error : ", mean_squared_error(Y_test, Y_pred)
    print "R2 Score: ", r2_score(Y_test, Y_pred)

    print >> fp, "Variance Score : ", explained_variance_score(Y_test, Y_pred)
    print >> fp, "Mean Absolute Error : ", mean_absolute_error(Y_test, Y_pred)
    print >> fp, "Mean Squared Error : ", mean_squared_error(Y_test, Y_pred)
    print >> fp, "R2 Score: ", r2_score(Y_test, Y_pred)
    print >> fp, 'C = {0}, gamma = {1}, epsilon = {2} '.format(C, gamma, epsilon)
    # Y_0 = target_scalar.inverse_transform(Y_test)
    # Y_1 = target_scalar.inverse_transform(Y_pred)
    # print ['{:.6f}' .format(i) for i in Y_0]
    # print ['{:.6f}' .format(i) for i in Y_1]

    # print "R2 Score: ", r2_score(Y_0, Y_1)
    # print "Mean Squared Error : ", mean_squared_error(Y_0, Y_1)

    # if LUMINANCE == True:
    #     Y_0 = wrapper_luminance_to_ev(Y_0)
    #     Y_1 = wrapper_luminance_to_ev(Y_1)

    #     print "R2 Score: ", r2_score(Y_0, Y_1)
    #     print "Mean Squared Error : ", mean_squared_error(Y_0, Y_1)

    # Y_01 = np.matrix.round(Y_0)
    # Y_11 = np.matrix.round(Y_1)

    # print "R2 Score: ", r2_score(Y_01, Y_11)
    # print "Mean Squared Error : ", mean_squared_error(Y_01, Y_11)

    # np.savetxt("y0.list", Y_0, fmt='%.6f')
    # np.savetxt("y01.list", Y_01, fmt='%.6f')
    # np.savetxt("y1.list", Y_1, fmt='%.6f')
    # np.savetxt("y11.list", Y_11, fmt='%.6f')

    # # testing
    # # sample = np.array([4, -7.5, -2, 6, 0, 0, 0, 0, 0, .88, .77, 0, .77])
    # sample = np.array([4.38333, -7.6833, -1.65, 6.2, 0, 0, 0, 0, 5, 0.878, 0.77, 0, 0.7])
    # sample = scale_sample(sample, feature_scalar)
    # sample_pred = regressor.predict(sample)
    # sample_pred = target_scalar.inverse_transform(sample_pred)
    # print sample_pred
    # print luminance_to_ev(sample_pred)

    fp.close()

    joblib.dump(regressor, dump_svr)
    joblib.dump(feature_scalar, dump_f_scalar)
    joblib.dump(target_scalar, dump_t_scalar)
    joblib.dump(pca, dump_pca)

if __name__ == "__main__":

    #db_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/"
    # db_path = "/home/yogesh/Project/Flickr-YsR/esplanade/"
    #feature_file = db_path + "dump_ImageDB_640/comp_map.fv"
    # feature_file = db_path + "dump_ImageDB_640/feature.fv"
    # feature_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/dump_ImageDB_640/composition_full_avg.fv"
    #a_score_file = db_path + "aesthetic.scores"

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump"
        sys.exit(0)

    db = sys.argv[1]
    dump_dir = sys.argv[2]

    # db = "/home/yogesh/Project/Flickr-YsR/"
    # db = "/mnt/windows/DataSet-YsR/"
    # dump_dir = "/dump_ImageDB_640/"
    # dump_dir = "/dump_Image_640/"

    # learn_camera_parameters(db, dump_dir, grid_search=True)    
    learn_camera_parameters(db, dump_dir, grid_search=False)    


