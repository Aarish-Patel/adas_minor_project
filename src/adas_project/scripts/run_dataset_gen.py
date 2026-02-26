#!/usr/bin/env python3
import subprocess
import time
import os
import signal

def run_dataset_generation():
    print("========================================")
    print(" ADAS GAZEBO DATASET GENERATOR ")
    print("========================================")
    
    workspace = '/home/hsiraa/adas_ws'
    
    # 1. Start Gazebo Core Simulation
    print("[1/4] Booting Gazebo Engine...")
    gz_proc = subprocess.Popen(
        f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && ros2 launch adas_project sim.launch.py",
        shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        
    time.sleep(10) # Wait for Gazebo to load the world and spawn adas_vehicle
    
    # 2. Start Traffic Scenario Controller
    print("[2/4] Injecting Traffic Actors...")
    scen_proc = subprocess.Popen(
        f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/scenario_controller.py",
        shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        
    time.sleep(2) # Let actors settle
    
    drivers = ['AGGRESSIVE', 'NORMAL', 'DEFENSIVE']
    
    for drv in drivers:
        print(f"\\n--- Generating {drv} dataset ---")
        # 3. Start Driver Bot
        drv_proc = subprocess.Popen(
            f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/behavior_generator.py --mode auto --driver {drv}",
            shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
            
        time.sleep(2)
        
        # 4. Start Orchestrator
        orch_proc = subprocess.Popen(
            f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/dataset_orchestrator.py --ros-args -p driver_type:={drv}",
            shell=True, executable='/bin/bash', preexec_fn=os.setsid)
            
        # Wait for orchestrator to finish (300 episodes)
        orch_proc.wait()
        
        # Slay driver bot
        os.killpg(os.getpgid(drv_proc.pid), signal.SIGINT)
        time.sleep(2)
        
    # Shutdown everything
    print("\\n[4/4] Dataset Complete. Cleaning up Gazebo...")
    os.killpg(os.getpgid(scen_proc.pid), signal.SIGINT)
    os.killpg(os.getpgid(gz_proc.pid), signal.SIGINT)
    
    print("FINISHED.")

if __name__ == "__main__":
    run_dataset_generation()
