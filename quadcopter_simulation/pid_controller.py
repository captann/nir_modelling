import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, dt, min_output=float('-inf'), max_output=float('inf'), max_rate=float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.min_output = min_output
        self.max_output = max_output
        self.max_rate = max_rate
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_output = 0.0
    
    def update(self, measurement):
        # Calculate error
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        d_term = -self.kd * (measurement - self.prev_error) / self.dt
        self.prev_error = measurement
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply rate limiting
        if self.max_rate < float('inf'):
            output_change = output - self.prev_output
            output_change = max(min(output_change, self.max_rate * self.dt), -self.max_rate * self.dt)
            output = self.prev_output + output_change
        
        # Apply output limits
        output = max(min(output, self.max_output), self.min_output)
        
        # Store previous output for rate limiting
        self.prev_output = output
        
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_output = 0.0
