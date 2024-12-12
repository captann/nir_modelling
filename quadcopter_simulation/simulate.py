import numpy as np
import json
from quadcopter import Quadcopter
from typing import Dict, List

class QuadcopterSimulation:
    def __init__(self, simulation_time=10.0, dt=0.01, initial_position=(0.0, 0.0, 2.0)):
        # Time parameters
        self.t = 0
        self.dt = dt
        self.simulation_time = simulation_time
        self.time_points = np.arange(0, simulation_time, dt)
        
        # Initialize quadcopter at specified position
        self.quadcopter = Quadcopter(dt=dt, initial_position=initial_position)
        
        # Storage for simulation results
        self.states: List[Dict] = []
    
    def run_simulation(self, wind_disturbance_start=3.0, 
                      wind_force=np.array([1.0, 0.5, 0.3])):
        """Run simulation with wind disturbance"""
        for t in self.time_points:
            # Apply wind disturbance after specified time
            current_wind = wind_force if t >= wind_disturbance_start else np.zeros(3)
            
            # Update quadcopter state
            self.quadcopter.update(self.dt, current_wind)
            
            # Store state
            self.states.append({
                "time": t,
                **self.quadcopter.get_state_dict()
            })
            
    def save_results(self, filename="simulation_results.json"):
        """Save simulation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.states, f, indent=2)

def main():
    # Create and run simulation with quadcopter starting at height of 2 meters
    sim = QuadcopterSimulation(simulation_time=15.0, initial_position=(0.0, 0.0, 2.0))
    # Added moderate wind disturbance
    sim.run_simulation(wind_disturbance_start=3.0, 
                      wind_force=np.array([0.5, 0.3, 0.1]))
    sim.save_results()

if __name__ == "__main__":
    main()
