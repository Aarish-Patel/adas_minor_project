#!/usr/bin/env python3
import subprocess
import time
import os
import signal
import json

def run_validation_loop():
    print("========================================")
    print(" ADAS GAZEBO VALIDATION LOOP ")
    print("========================================")
    
    workspace = '/home/hsiraa/adas_ws'
    
    # 1. Start Pygame Dashboard
    print("[1/5] Booting Live Dashboard...")
    dash_proc = subprocess.Popen(['python3', 'dashboard.py'], cwd=f'{workspace}/src/adas_project/scripts')
    
    # Init Stats
    with open('../results/live_stats.json', 'w') as f:
        json.dump({"current_phase": "BOOTING SIMULATOR..."}, f)
        
    # 2. Start Gazebo Core Simulation
    print("[2/5] Booting Gazebo Engine...")
    gz_proc = subprocess.Popen(
        f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && ros2 launch adas_project sim.launch.py",
        shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        
    time.sleep(15) 
    
    # 3. Start Traffic
    print("[3/5] Injecting Traffic Actors...")
    scen_proc = subprocess.Popen(
        f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/scenario_controller.py",
        shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        
    time.sleep(5) 
    
    drivers = ['AGGRESSIVE', 'NORMAL', 'DEFENSIVE']
    levels = [0, 1, 2, 3, 4]
    systems = ['FIXED', 'ML']
    
    print("\\n[4/5] Running 6000 Episodes...")
    for drv in drivers:
        for lvl in levels:
            for sys_type in systems:
            
                print(f"\\n--- {drv} | L{lvl} | {sys_type} ---")
                
                # Update phase
                with open('../results/live_stats.json', 'r') as f:
                    stats = json.load(f)
                    stats['current_phase'] = "SIMULATING..."
                with open('../results/live_stats.json', 'w') as f:
                    json.dump(stats, f)
                
                # Start Driver Node
                drv_proc = subprocess.Popen(
                    f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/behavior_generator.py --mode auto --driver {drv}",
                    shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
                    
                # Start ADAS Node
                adas_file = "fixed_adas.py" if sys_type == 'FIXED' else "ml_adas.py"
                adas_proc = subprocess.Popen(
                    f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/{adas_file} --ros-args -p adas_level:={lvl}",
                    shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
                    
                time.sleep(1)
                
                # Start Orchestrator
                orch_proc = subprocess.Popen(
                    f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash && python3 {workspace}/src/adas_project/scripts/validation_orchestrator.py --ros-args -p driver_type:={drv} -p level:={lvl} -p system:={sys_type}",
                    shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
                    
                # Wait for 200 episodes
                orch_proc.wait()
                
                # Slay driver and adas for next permutation
                os.killpg(os.getpgid(adas_proc.pid), signal.SIGINT)
                os.killpg(os.getpgid(drv_proc.pid), signal.SIGINT)
                time.sleep(1.5)
                
    # Shutdown Gazebo
    print("\\n[5/5] Generating Graphs & Cleanup...")
    with open('../results/live_stats.json', 'r') as f:
        stats = json.load(f)
        stats['current_phase'] = "GENERATING GRAPHS..."
    with open('../results/live_stats.json', 'w') as f:
        json.dump(stats, f)
    
    os.killpg(os.getpgid(scen_proc.pid), signal.SIGINT)
    os.killpg(os.getpgid(gz_proc.pid), signal.SIGINT)
    
    # Generate graphs
    subprocess.run(['python3', 'generate_graphs.py'], cwd=f'{workspace}/src/adas_project/scripts')
    
    # Kill Dashboard
    dash_proc.kill()
    print("VALIDATION COMPLETE.")

if __name__ == "__main__":
    run_validation_loop()
