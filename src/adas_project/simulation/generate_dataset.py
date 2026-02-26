import sys
import os
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulation.fast_sim import FastSim
from drivers.driver_models import Driver

def generate_dataset(num_episodes=300, max_steps=500, dt=0.1):
    os.makedirs('../dataset', exist_ok=True)
    filename = '../dataset/adas_dataset.csv'
    
    headers = ['speed', 'acceleration', 'lateral_deviation', 'yaw', 'steering', 
               'throttle', 'brake', 'obstacle_distance', 'ttc', 'offroad', 'collision', 
               'driver_type', 'intent_lane_change', 'intent_brake', 'intent_overtake', 'intent_offroad']
               
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for driver_type in ['AGGRESSIVE', 'NORMAL', 'DEFENSIVE']:
            print(f'Generating {num_episodes} episodes for {driver_type} driver...')
            driver = Driver(driver_type)
            for ep in range(num_episodes):
                sim = FastSim(dt)
                driver.target_lane = 0.0
                state = sim.get_track_state()
                
                for step in range(max_steps):
                    steer, throttle, brake, il, ib, io, iof = driver.get_action(state, dt)
                    
                    row = {
                        'speed': state['speed'],
                        'acceleration': state['acceleration'],
                        'lateral_deviation': state['lateral_deviation'],
                        'yaw': 0.0, # Simplified in 2D track-relative
                        'steering': steer,
                        'throttle': throttle,
                        'brake': brake,
                        'obstacle_distance': state['obstacle_distance'],
                        'ttc': state['ttc'],
                        'offroad': int(state['offroad']),
                        'collision': int(state['collision']),
                        'driver_type': driver_type,
                        'intent_lane_change': il,
                        'intent_brake': ib,
                        'intent_overtake': io,
                        'intent_offroad': iof
                    }
                    writer.writerow(row)
                    
                    state = sim.step(steer, throttle, brake)
                    
                    if state['collision'] or state['offroad']:
                        # Save final state and terminate episode
                        row['collision'] = int(state['collision'])
                        row['offroad'] = int(state['offroad'])
                        writer.writerow(row)
                        break

if __name__ == '__main__':
    generate_dataset()
