#!/usr/bin/env python3
"""
train_intent.py — Trains ML Models for Driver Intent Prediction
Reads from dataset/adas_dataset.csv and trains Random Forest models 
for each intent target. Saves to models/ directory.
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), '../dataset/adas_dataset.csv')
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print("Dataset not found! Run dataset generator first.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows.")
    
    if len(df) < 100:
        print("Dataset too small. Generate more data.")
        return

    # Prepare features and targets
    feature_cols = ['speed', 'acceleration', 'lateral_deviation', 'yaw', 
                    'steering', 'throttle', 'brake', 'obstacle_distance', 'ttc']
    target_cols = ['intent_lane_change', 'intent_brake', 'intent_overtake', 'intent_offroad']
    
    # Cap TTC to 999.0
    if 'ttc' in df.columns:
        df['ttc'] = df['ttc'].clip(upper=999.0)
        
    # Drop rows with NaNs
    df = df.dropna(subset=feature_cols + target_cols)
    print(f"Valid rows for training: {len(df)}")
    
    X = df[feature_cols].values
    
    # Train a separate model for each intent
    for target in target_cols:
        print(f"\n--- Training Model for: {target} ---")
        y = df[target].values
        
        # Check class distribution
        pos_count = np.sum(y)
        if pos_count == 0 or pos_count == len(y):
            print(f"WARNING: No variation in target {target}. All values are {y[0]}. Skipping training.")
            continue
            
        print(f"Class distribution - Positive: {pos_count} ({(pos_count/len(y))*100:.1f}%), Negative: {len(y)-pos_count}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest
        # Class weight balanced helps with rare events like offroad/collision intent
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(models_dir, f"{target}_rf.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Saved to: {model_path}")
        
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
