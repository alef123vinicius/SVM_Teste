import numpy as np
import pandas as pd
import random as r
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

RS = 1993
np.random.seed(RS)
K = 10
H = 0.7
print "Started MNIST Radial"

df_dataset = pd.read_csv('../dataset/200pcs-train.csv')

class ClassifierWrapper(object):
    def __init__(self, classifier, params=None):
        self.classifier = classifier(**params)

    def train(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)

    def predict(self, x):
    	return self.classifier.predict(x)

    def getClassifier(self):
    	return self.classifier

def cross_validation(classifier, X_train, Y_train):
    kf = KFold(X_train.shape[0], n_folds=K, shuffle=False, random_state=RS)
    
    cv_acc = []
    
    for i, (train_index, valid_index) in enumerate(kf):
        x_train = X_train[train_index]
        y_train = Y_train[train_index]
        
        x_valid = X_train[valid_index]
        y_valid = Y_train[valid_index]
            
        classifier.train(x_train, y_train)
        predicts = classifier.predict(x_valid)
        acc = accuracy_score(y_valid, predicts)
        print "Accuracy in k={} is {}".format(i, acc)

        cv_acc.append(acc)

    # accuracy
    return np.mean(cv_acc)

def holdout(classifier, x_dataset, y_dataset):
    train_x, valid_x, train_y, valid_y = train_test_split(x_dataset, y_dataset, train_size=H)
    classifier.train(train_x, train_y)
    predicts = classifier.predict(valid_x)
    acc = accuracy_score(valid_y, predicts)
    return acc

def run(radial_params_list, x_dataset, y_dataset, validation='hd'):
    """
    validation: hd for holdout and cv for cross validation k fold
    """
    classifiers = []
    for params in radial_params_list:
        classifiers.append(ClassifierWrapper(classifier=SVC, params=params))

    print "# of classifiers: {}".format(len(classifiers))
    results = radial_params_list
    for i, classifier in enumerate(classifiers):
        print '# {} with params: {}'.format(i, radial_params_list[i])    
        if validation == 'cv':
            acc = cross_validation(classifier, x_dataset, y_dataset)
            print "Acc cross validation with {}-fold: {}".format(K, acc)
            results[i]['acc'] = acc
        elif validation == 'hd':
            acc = holdout(classifier, x_dataset, y_dataset)
            print "Acc holdout with {}% for train: {}".format(H * 100, acc)
            results[i]['acc'] = acc
        else:
            print 'error: set validation type'

    results = pd.DataFrame(data=results)
    return results

def main():
    #nclass = len(df_dataset.label.unique())
    y_dataset = df_dataset.label.values
    #le = LabelEncoder()
    #y_dataset = le.fit_transform(y_dataset)
    x_dataset = df_dataset.drop('label', axis=1).values

    # set grid params
    radial_params = {
        'kernel':['rbf'],
        'gamma':[0.1, 0.3, 0.5, 0.7, 0.9],
        'C':[20, 200]
    }

    radial_params_list = list(ParameterGrid(radial_params))
    results = run(radial_params_list, x_dataset, y_dataset)
    results.to_csv('results_radial_svm_200pcs.csv', index=False)
    
main()
print "Done"