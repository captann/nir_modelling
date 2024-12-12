import numpy as np
from dataclasses import dataclass
from pid_controller import PIDController

@dataclass
class QuadcopterState:
    # Position
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # Velocity
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # Attitude angles
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    # Angular rates
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def get_dict(self):
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'vx': self.vx, 'vy': self.vy, 'vz': self.vz,
            'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw,
            'wx': self.wx, 'wy': self.wy, 'wz': self.wz
        }

class Quadcopter:
    def __init__(self, dt=0.01, initial_position=(0.0, 0.0, 0.0)):
        # Physical parameters
        self.mass = 1.0  # kg
        self.arm_length = 0.2  # meters
        self.inertia = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.6]
        ]) * 0.1  # kg*m^2
        
        # Motor parameters
        self.MIN_PWM = 1000  # Минимальное значение PWM
        self.MAX_PWM = 2000  # Максимальное значение PWM
        self.PWM_RANGE = self.MAX_PWM - self.MIN_PWM
        
        # Коэффициенты для модели двигателя
        self.KT = 7.5e-8  # Коэффициент тяги [N/(PWM^2)]
        self.KD = 1.5e-9  # Коэффициент сопротивления [Nm/(PWM^2)]
        
        self.dt = dt
        self.g = 9.81  # m/s^2
        
        # Initialize state with given position
        self.state = QuadcopterState()
        self.state.x, self.state.y, self.state.z = initial_position
        
        # Calculate hover PWM
        hover_thrust = self.mass * self.g / 4  # Делим на 4, так как у нас 4 мотора
        self.HOVER_PWM = np.sqrt(hover_thrust / self.KT)
        
        # Control parameters
        self.MAX_TILT = np.deg2rad(45)
        
        # Initialize controllers with target at initial position
        self._init_controllers(initial_position)
        
        # Store motor PWM values
        self.motor_pwm = np.ones(4) * self.HOVER_PWM

    def _init_controllers(self, target_position):
        # Position controllers - increased proportional and derivative gains, reduced integral
        self.x_pid = PIDController(kp=3.0, ki=0.1, kd=4.0,
                                setpoint=target_position[0], dt=self.dt,
                                min_output=-self.MAX_TILT,
                                max_output=self.MAX_TILT)
        
        self.y_pid = PIDController(kp=3.0, ki=0.1, kd=4.0,
                                setpoint=target_position[1], dt=self.dt,
                                min_output=-self.MAX_TILT,
                                max_output=self.MAX_TILT)
        
        # Z controller - adjusted gains for better height control
        self.z_pid = PIDController(kp=100, ki=0.05, kd=50,
                                setpoint=target_position[2], dt=self.dt,
                                min_output=-400,  # Increased range for more authority
                                max_output=400)

        # Attitude stabilization - increased gains for faster response
        self.attitude_kp = 12.0
        self.attitude_kd = 4.0

    def _calculate_motor_pwm(self, thrust_offset, roll_torque, pitch_torque, yaw_torque):
        """Calculate PWM values for motors based on desired thrust and torques"""
        # Base PWM for all motors (hover PWM + thrust offset from Z controller)
        base_pwm = self.HOVER_PWM + thrust_offset
        
        # Convert torques to PWM differences
        roll_pwm = roll_torque * 400  # Scale factor for roll
        pitch_pwm = pitch_torque * 400  # Scale factor for pitch
        yaw_pwm = yaw_torque * 400  # Scale factor for yaw
        
        # Calculate individual motor PWM values
        m1_pwm = base_pwm - pitch_pwm - roll_pwm - yaw_pwm
        m2_pwm = base_pwm - pitch_pwm + roll_pwm + yaw_pwm
        m3_pwm = base_pwm + pitch_pwm + roll_pwm - yaw_pwm
        m4_pwm = base_pwm + pitch_pwm - roll_pwm + yaw_pwm
        
        # Clip PWM values to valid range
        pwm_values = np.clip([m1_pwm, m2_pwm, m3_pwm, m4_pwm], 
                           self.MIN_PWM, self.MAX_PWM)
        
        return pwm_values

    def _pwm_to_forces(self, pwm_values):
        """Convert PWM values to forces and torques"""
        # Calculate forces based on PWM (quadratic relationship)
        forces = self.KT * pwm_values ** 2
        return forces

    def update(self, dt, wind_force=np.zeros(3)):
        # Get desired angles from position control
        desired_pitch = -self.x_pid.update(self.state.x)
        desired_roll = self.y_pid.update(self.state.y)
        thrust_pwm_offset = self.z_pid.update(self.state.z)
        
        # Attitude control
        roll_torque = -self.attitude_kp * (self.state.roll - desired_roll) - self.attitude_kd * self.state.wx
        pitch_torque = -self.attitude_kp * (self.state.pitch - desired_pitch) - self.attitude_kd * self.state.wy
        yaw_torque = -self.attitude_kp * self.state.yaw - self.attitude_kd * self.state.wz
        
        # Calculate motor PWM values
        self.motor_pwm = self._calculate_motor_pwm(thrust_pwm_offset, roll_torque, pitch_torque, yaw_torque)
        
        # Convert PWM to forces
        motor_forces = self._pwm_to_forces(self.motor_pwm)
        
        # Update physics
        self._update_physics(dt, motor_forces, wind_force)

    def _update_physics(self, dt, motor_forces, wind_force):
        # Total thrust in body frame
        thrust = np.array([0, 0, np.sum(motor_forces)])
        
        # Rotation matrix from body to inertial frame
        R = self._rotation_matrix()
        
        # Forces in inertial frame
        thrust_inertial = R @ thrust
        gravity = np.array([0, 0, -self.mass * self.g])
        total_force = thrust_inertial + gravity + wind_force
        
        # Linear acceleration, velocity and position
        acceleration = total_force / self.mass
        self.state.vx += acceleration[0] * dt
        self.state.vy += acceleration[1] * dt
        self.state.vz += acceleration[2] * dt
        
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        self.state.z += self.state.vz * dt
        
        # Torques (assuming small angles)
        roll_torque = self.arm_length * (motor_forces[2] + motor_forces[3] - motor_forces[0] - motor_forces[1])
        pitch_torque = self.arm_length * (motor_forces[2] - motor_forces[3] - motor_forces[0] + motor_forces[1])
        yaw_torque = 0.1 * (motor_forces[1] + motor_forces[3] - motor_forces[0] - motor_forces[2])
        torques = np.array([roll_torque, pitch_torque, yaw_torque])
        
        # Angular acceleration, velocity and position
        angular_acc = np.linalg.inv(self.inertia) @ (torques - np.cross(np.array([self.state.wx, self.state.wy, self.state.wz]), self.inertia @ np.array([self.state.wx, self.state.wy, self.state.wz])))
        
        self.state.wx += angular_acc[0] * dt
        self.state.wy += angular_acc[1] * dt
        self.state.wz += angular_acc[2] * dt
        
        self.state.roll += self.state.wx * dt
        self.state.pitch += self.state.wy * dt
        self.state.yaw += self.state.wz * dt

    def _rotation_matrix(self):
        """Returns rotation matrix from body to inertial frame"""
        cr, cp, cy = np.cos(self.state.roll), np.cos(self.state.pitch), np.cos(self.state.yaw)
        sr, sp, sy = np.sin(self.state.roll), np.sin(self.state.pitch), np.sin(self.state.yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def get_state_dict(self):
        state_dict = self.state.get_dict()
        state_dict['motor_pwm'] = self.motor_pwm.tolist()
        return state_dict
