# -*- coding: utf-8 -*-
"""

@author: timpr
"""
from classes.predictor_class import Predictor

if __name__ == "__main__":
    
    data_file = "nfl_data_full.csv" 
    target_file = "nfl_targets_full.csv" 
    prediction_file = "nfl_2021_players.csv"
    
    # Create a predictor model to implement different ML methods
    predictor = Predictor(data_file, target_file, prediction_file)
    
    # Implement logisic regression
    log_score, log_targets = predictor.logisitic_regression_predictor()
    print("Logistic Regression Accuracy: " + str(round(log_score,3)))
    print("Logistic Regression Predictions:")
    print(log_targets)
    print("")
    
    # Implement decision tree
    dt_score, dt_targets = predictor.decision_tree_predictor()
    print("Decision Tree Accuracy: " + str(round(dt_score,3)))
    print("Decision Tree Predictions:")
    print(dt_targets)
    print("")
    
    # Implement k-nearest neighbors
    knn_score, knn_targets = predictor.knn_predictor()
    print("K Nearest Neighbors Accuracy: " + str(round(knn_score,3)))
    print("K Nearest Neighbors Predictions:")
    print(knn_targets)
    print("")
    
    # Implement Gaussian Naive Bayes
    gnb_score, gnb_targets = predictor.gnb_predictor()
    print("Gaussian Naive Bayes Accuracy: " + str(round(gnb_score,3)))
    print("Gaussian Naive Bayes Predictions:")
    print(gnb_targets)
    print("")
    
    # Implement support vector machine
    svm_score, svm_targets = predictor.svm_predictor()
    print("Support Vector Machine Accuracy: " + str(round(svm_score,3)))
    print("Support Vector Machine Predictions:")
    print(svm_targets)
    print("")