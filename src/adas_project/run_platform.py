#!/usr/bin/env python3
import os
import subprocess
import time

def run_pipeline():
    print("========================================")
    print(" ADAS GAZEBO NATIVE AUTONOMOUS PIPELINE ")
    print("========================================")
    
    workspace = '/home/hsiraa/adas_ws/src/adas_project'
    
    # Step 3: Dataset Generation (Native Gazebo)
    print("\\n[1/3] Generating Driver Dataset in Gazebo...")
    res = subprocess.run(['python3', 'run_dataset_gen.py'], cwd=f'{workspace}/scripts')
    if res.returncode != 0:
        print("Dataset generation failed.")
        return
        
    # Step 4: ML Training Loop
    print("\\n[2/3] Training Intent Prediction Models...")
    res = subprocess.run(['python3', 'train_intent.py'], cwd=f'{workspace}/ml')
    if res.returncode != 0:
        print("ML Training failed.")
        return
        
    # Step 7-11: Validate, Sim, Vis & Dashboard (Native Gazebo)
    print("\\n[3/3] Running Gazebo Validation Loop & Live Dashboard...")
    res = subprocess.run(['python3', 'run_validation.py'], cwd=f'{workspace}/scripts')
    if res.returncode != 0:
        print("Validation Loop failed.")
        return
        
    print("========================================")
    print(" PIPELINE COMPLETED AUTONOMOUSLY ")
    print("========================================")

if __name__ == "__main__":
    run_pipeline()
