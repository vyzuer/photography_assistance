import sys, os
import numpy as np
from sklearn import preprocessing, cross_validation, svm, grid_search
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import time
from sklearn.externals import joblib

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

    return clf.best_params_['C'], clf.best_params_['gamma']

def run_grid_search(X, Y):
    print("# Tuning hyper-parameters") 

    # C_range = 2. ** np.array([-5,-3,-1, 1, 3, 5, 7, 9, 11, 13, 15 ,17])
    # gamma_range = 2. ** np.array([-15 ,-13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9])

    C_range = 2. ** np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # C_range = [1, 1.3, 1.7, 2, 2.3, 2.7, 3, 4, 5]
    gamma_range = [0.001, 0.01, 0.1, 1, 1.4, 1.8, 2, 2.5, 3, 6]

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.2, random_state=777)

    parameters = {'kernel':['rbf'], 'C':C_range, 'gamma':gamma_range}

    svr = svm.SVC(tol=0.0000000001)
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


def run_cross_validation(X, Y, C, gamma, fp):

    n_samples = X.shape[0]
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=0.2, random_state=777)
    # regressor = svm.SVC(C=8, gamma=8)
    regressor = svm.SVC(kernel= 'rbf', C=C, gamma=gamma)
    scores = cross_validation.cross_val_score(regressor, X, Y, cv=cv)

    print "Score : ", scores
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

    print >> fp, "Score : ", scores
    print >> fp, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)


def remove_average_photographs(data, target):
    x, y = data.shape

    X = []
    Y = []

    for i in range(x):
        if target[i] < 0.25:
            Y.append(0)
            X.append(data[i])

        elif target[i] > 0.75:
            Y.append(1)
            X.append(data[i])

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y

def read_features(db_path, dump_dir, comp=True, view=True, geo=False, time=False):
    # db = "/home/yogesh/Project/Flickr-YsR/"
    dump_path = db_path + "feature_dumps/"

    f_comp = dump_path + "/feature.list"

    f_view = dump_path + "/view.list"

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

def read_params(f_name):
    params = [2.0, 2.0]

    if os.path.exists(f_name):
        params = np.loadtxt(f_name)
    
    return params[0], params[1]


def dump_params(f_name, C, gamma):
    fp = open(f_name, 'w')
    fp.write('{0:0.8f} {1:0.8f}'.format(C, gamma))
    fp.close()


def learn_composition(db, dump_dir, grid_search=False):
    results_dir = db + "results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    f_result = results_dir + "/res_comp." + str(time.time())

    params_dir = db + "params/"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    f_name = params_dir + "/.comp.params"

    model_dir = db + "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    comp_model_dir = model_dir + "comp/"
    if not os.path.exists(comp_model_dir):
        os.makedirs(comp_model_dir)

    dump_scalar = comp_model_dir + "/scalar.pkl"
    dump_pca = comp_model_dir + "/pca.pkl"
    dump_svm = comp_model_dir + "/svm.pkl"

    print f_result
    fp = open(f_result, 'w')
    # load features
    data, target = read_features(db, dump_dir)

    print data.shape
    print target.shape
    # target = target*5
    # Y = np.matrix.round(target)

    X, Y = remove_average_photographs(data, target)
    print X.shape
    
    X, feature_scalar = scale_data(X)
    
    pca = PCA(n_components=250)
    X = pca.fit_transform(X)

    # X = data

    # C = 32
    # gamma = 2

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=777)

    if grid_search == False:
        C, gamma = read_params(f_name)
        # C, gamma = 2, 2
    else:
        clf = run_grid_search(X_train, Y_train)
        C, gamma = generate_report(clf, X_test, Y_test, fp)
        dump_params(f_name, C, gamma)

    run_cross_validation(X_train, Y_train, C, gamma, fp)

    regressor = svm.SVC(kernel= 'rbf', C=C, gamma=gamma, probability=True)
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
    # print regressor.predict_proba(X_test)
    # result =  regressor.predict_proba(data)
    # np.savetxt("prob.list", result, fmt='%.6f')
    print precision_recall_fscore_support(Y_test, Pr)


    Pr = regressor.predict(X_test)
    print >> fp, regressor.score(X_train, Y_train)
    print >> fp, regressor.score(X_test, Y_test)
    print >> fp, 'Accuracy : {0}'.format(accuracy_score(Y_test, Pr))
    print >> fp, 'Average Precision : {0}'.format(average_precision_score(Y_test, Pr))
    print >> fp, 'F1 Score : {0}'.format(f1_score(Y_test, Pr))
    print >> fp, 'Precision : {0}'.format(precision_score(Y_test, Pr))
    print >> fp, 'Recall : {0}'.format(recall_score(Y_test, Pr))
    print >> fp, 'ROC : {0}'.format(roc_auc_score(Y_test, Pr))
    print >> fp, precision_recall_fscore_support(Y_test, Pr)
    print >> fp, 'C = {0}, gamma = {1} '.format(C, gamma)

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), algorithm="SAMME", n_estimators=200)
    # # clf = AdaBoostClassifier(svm.SVC(kernel= 'rbf', C=32, gamma=32), algorithm="SAMME", n_estimators=200)
    # scores = cross_validation.cross_val_score(clf, X, Y)
    # print scores
    # print scores.mean()

    # # clf = RandomForestClassifier(n_estimators=50)
    # clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=777)
    # scores = cross_validation.cross_val_score(clf, X, Y)
    # print scores
    # print scores.mean()

    fp.close()
    
    joblib.dump(regressor, dump_svm)
    joblib.dump(feature_scalar, dump_scalar)
    joblib.dump(pca, dump_pca)


if __name__ == "__main__":

    #db_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/"
    # db_path = "/home/yogesh/Project/Flickr-YsR/esplanade/"
    #feature_file = db_path + "dump_ImageDB_640/comp_map.fv"
    # feature_file = db_path + "dump_ImageDB_640/feature.fv"
    # feature_file = "/home/yogesh/Project/Flickr-YsR/merlionImages/dump_ImageDB_640/composition_full_avg.fv"
    #a_score_file = db_path + "aesthetic.scores"

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    db = sys.argv[1]
    dump_dir = sys.argv[2]

    # db = "/home/yogesh/Project/Flickr-YsR/"
    # db = "/mnt/windows/DataSet-YsR/"
    # dump_dir = "/dump_ImageDB_640/"
    # dump_dir = "/dump_Image_640/"

    # learn_composition(db, dump_dir, grid_search=False)
    learn_composition(db, dump_dir, grid_search=True)

