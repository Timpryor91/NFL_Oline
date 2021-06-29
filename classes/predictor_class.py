# -*- coding: utf-8 -*-
"""

@author: timpr

Some inspiration taken from the following article by Michael Galarnyk
https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

and the following post from Stack Overflow
https://stackoverflow.com/questions/39163354/evaluating-logistic-regression-with-cross-validation

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class Predictor(object):
    """
    A predictor object represents a model used to predict whether an
    NFL offensive lineman draft prospect will be successful in the league or not
    
    """
    def __init__(self, data_filename, target_filename, prediction_filename):
        """
        Initialize a predictor object
        
        Parameters:
            data_filename (string): the file name of a csv containing the training/validation data
            target_filename (string): the file name of a csv containing the target values for the data
        
        """
        # Initiate numpy arrays from the input data
        self.data_filename = data_filename
        self.data_df = pd.read_csv(data_filename)
        self.data = self.data_df.to_numpy()
                
        self.target_filename = target_filename
        self.target_df = pd.read_csv(target_filename)
        self.target = self.target_df.to_numpy()
        
        self.prediction_filename = prediction_filename
        self.prediction_df = pd.read_csv(prediction_filename)
        self.prediction = self.prediction_df.to_numpy()
                                
    def logisitic_regression_predictor(self):
        """
        Generates a logistic regression model to predict success, utilizing
        cross validation

        Returns:
            logi_accuracy_score (float): the accuracy of the model assessed on the data
            logi_predicted_targets (np array): the prediction targets from the unknown data
        
        """
        
        self.log_regression_model = LogisticRegressionCV(cv=10).fit(self.data, self.target.ravel()) 
        self.logi_accuracy_score = self.log_regression_model.score(self.data, self.target.ravel())
        self.logi_predicted_targets = self.log_regression_model.predict(self.prediction)
        
        return(self.logi_accuracy_score, self.logi_predicted_targets)

    def decision_tree_predictor(self):
        """
        Generates a model using a decision tree method to predict success, utilizing
        cross validation
        
        Returns:
            dt_accuracy_score (float): the accuracy of the model assessed on the data
            dt_predicted_targets (np array): the prediction targets from the unknown data        
        
        """
        
        self.dt_model = tree.DecisionTreeClassifier().fit(self.data, self.target.ravel())
        self.dt_accuracy_score = np.mean(cross_val_score(self.dt_model,
                                                         self.data,
                                                         self.target.ravel(),
                                                         cv = 10))
        self.dt_predicted_targets = self.dt_model.predict(self.prediction)        
        
        return(self.dt_accuracy_score, self.dt_predicted_targets)
    
    def knn_predictor(self):
        """
        Generates a model using a k nearest neighbors method to predict success, utilizing
        cross validation
        
        Returns:
            knn_accuracy_score (float): the accuracy of the model assessed on the data
            knn_predicted_targets (np array): the prediction targets from the unknown data        
                
        """
        self.knn_model = KNeighborsClassifier().fit(self.data, self.target.ravel())
        self.knn_accuracy_score = np.mean(cross_val_score(self.knn_model,
                                                          self.data,
                                                          self.target.ravel(),
                                                          cv = 10))
        self.knn_predicted_targets = self.knn_model.predict(self.prediction)          
        
        return(self.knn_accuracy_score, self.knn_predicted_targets)

    def gnb_predictor(self):
        """
        Generates a model using a Gaussian Naive Bayes method to predict success, utilizing
        cross validation
        
        Returns:
            gnb_accuracy_score (float): the accuracy of the model assessed on the data
            gnb_predicted_targets (np array): the prediction targets from the unknown data        
                
        """
        self.gnb_model = GaussianNB().fit(self.data, self.target.ravel())
        self.gnb_accuracy_score = np.mean(cross_val_score(self.gnb_model,
                                                          self.data,
                                                          self.target.ravel(),
                                                          cv = 10))
        self.gnb_predicted_targets = self.gnb_model.predict(self.prediction)          
        
        return(self.gnb_accuracy_score, self.gnb_predicted_targets)

    def svm_predictor(self):
        """
        Generates a model using a Support Vector Machine method to predict success, utilizing
        cross validation
        
        Returns:
            svm_accuracy_score (float): the accuracy of the model assessed on the data
            svm_predicted_targets (np array): the prediction targets from the unknown data        
                
        """
        self.svm_model = svm.SVC().fit(self.data, self.target.ravel())
        self.svm_accuracy_score = np.mean(cross_val_score(self.svm_model,
                                                          self.data,
                                                          self.target.ravel(),
                                                          cv = 10))
        self.svm_predicted_targets = self.svm_model.predict(self.prediction)          
        
        return(self.svm_accuracy_score, self.svm_predicted_targets)