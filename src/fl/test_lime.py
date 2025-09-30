#!/usr/bin/env python3

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import lime
import lime.lime_tabular


INPUT_SHAPE = 78

def build_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(INPUT_SHAPE,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def load_cicids_data():
    try:
        df = pd.read_csv("../../data/cicids/CICIDS2017_alpha.csv")
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Convert labels to binary (0 = BENIGN, 1 = ATTACK)
        y_binary = (y != 'BENIGN').astype(int)
        
        return X, y_binary, df.columns[:-1].tolist()
    
    except Exception as e:
        return None, None, None

def test_pure_lime():
    X, y, feature_names = load_cicids_data()
    if X is None:
        return
    
    df = pd.read_csv("../../data/cicids/CICIDS2017_alpha.csv")
    labels = df.iloc[:, -1].values
 
    portscan_indices = np.where(labels == 'PortScan')[0]
    benign_indices = np.where(labels == 'BENIGN')[0]
    
    if len(portscan_indices) == 0:
        print("No PortScan attacks found")
        return
    
    model = build_model()
    X_background = X[benign_indices[:1000]]
    portscan_sample_idx = portscan_indices[0]
    X_test = X[portscan_sample_idx:portscan_sample_idx+1]
    
    try:
        # define prediction function for LIME
        def predict_proba(X):
            predictions = model.predict(X, verbose=0)
            return np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        
        # create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_background,
            feature_names=feature_names,
            class_names=['Normal', 'PortScan'],
            mode='classification',
            discretize_continuous=True
        )
        
        # explain the test instance
        explanation = explainer.explain_instance(
            X_test[0], 
            predict_proba, 
            num_features=15
        )
        
        # raw LIME output
        exp_list = explanation.as_list(label=1)
        
        for feature, weight in exp_list:
            print(f"{feature}: {weight:.6f}")
            
    except Exception as e:
        print(f"Error: {e}")
    try:
        # define prediction function for LIME
        def predict_proba(X):
            predictions = model.predict(X, verbose=0)
            return np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_background,
            feature_names=feature_names,
            class_names=['Normal', 'PortScan'],
            mode='classification',
            discretize_continuous=True
        )
        
        # explain the test instance
        explanation = explainer.explain_instance(
            X_test[0], 
            predict_proba, 
            num_features=15
        )
        
        # raw LIME output
        exp_list = explanation.as_list(label=1)
        
        for feature, weight in exp_list:
            print(f"{feature}: {weight:.6f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pure_lime()