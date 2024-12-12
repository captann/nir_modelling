import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

class QuadcopterVisualizer:
    def __init__(self, data_file: str):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
    
    def extract_data(self, key: str) -> List[float]:
        """Extract data series from simulation results"""
        return [state[key] for state in self.data]
    
    def plot_state_response(self):
        """Plot complete state response"""
        fig = plt.figure(figsize=(15, 15))
        
        # Create subplots with specific height ratios and width ratios
        gs = plt.GridSpec(3, 2, height_ratios=[2, 2, 1])
        
        # First column: time series plots
        # Position subplot
        ax_pos = fig.add_subplot(gs[0, 0])
        self.plot_positions(ax_pos)
        
        # Angles subplot
        ax_ang = fig.add_subplot(gs[0, 1])
        self.plot_angles(ax_ang)
        
        # Linear velocities subplot
        ax_vel = fig.add_subplot(gs[1, 0])
        self.plot_velocities(ax_vel)
        
        # Angular velocities subplot
        ax_avel = fig.add_subplot(gs[1, 1])
        self.plot_angular_velocities(ax_avel)
        
        # Motor forces subplot (spans both columns)
        ax_motors = fig.add_subplot(gs[2, :])
        self.plot_motor_forces(ax_motors)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_trajectory(self):
        """Plot 3D trajectory of the quadcopter"""
        fig = plt.figure(figsize=(15, 15))
        
        # Create subplots with specific height ratios
        gs = plt.GridSpec(3, 2, height_ratios=[2, 2, 1])
        
        # First column: time series plots
        # Position subplot
        ax_pos = fig.add_subplot(gs[0, 0])
        self.plot_positions(ax_pos)
        
        # Angles subplot
        ax_ang = fig.add_subplot(gs[0, 1])
        self.plot_angles(ax_ang)
        
        # Linear velocities subplot
        ax_vel = fig.add_subplot(gs[1, 0])
        self.plot_velocities(ax_vel)
        
        # Angular velocities subplot
        ax_avel = fig.add_subplot(gs[1, 1])
        self.plot_angular_velocities(ax_avel)
        
        # Motor forces subplot (spans both columns)
        ax_motors = fig.add_subplot(gs[2, :])
        self.plot_motor_forces(ax_motors)
        
        # Add 3D trajectory plot in a new figure
        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # Extract position data
        x = self.extract_data('x')
        y = self.extract_data('y')
        z = self.extract_data('z')
        
        # Plot trajectory
        ax_3d.plot(x, y, z, 'b-', label='Траектория')
        
        # Plot start and end points
        ax_3d.scatter(x[0], y[0], z[0], c='g', marker='o', s=100, label='Старт')
        ax_3d.scatter(x[-1], y[-1], z[-1], c='r', marker='o', s=100, label='Финиш')
        
        # Set labels and title
        ax_3d.set_xlabel('X (м)')
        ax_3d.set_ylabel('Y (м)')
        ax_3d.set_zlabel('Z (м)')
        ax_3d.set_title('3D Траектория квадрокоптера')
        
        # Add legend
        ax_3d.legend()
        
        return fig, fig_3d
    
    def plot_positions(self, ax):
        """Plot position data"""
        time = self.extract_data('time')
        x = self.extract_data('x')
        y = self.extract_data('y')
        z = self.extract_data('z')
        
        ax.plot(time, x, label='X')
        ax.plot(time, y, label='Y')
        ax.plot(time, z, label='Z')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('Положение (м)')
        ax.set_title('Положение')
    
    def plot_angles(self, ax):
        """Plot angle data"""
        time = self.extract_data('time')
        roll = np.rad2deg(self.extract_data('roll'))
        pitch = np.rad2deg(self.extract_data('pitch'))
        yaw = np.rad2deg(self.extract_data('yaw'))
        
        ax.plot(time, roll, label='Крен')
        ax.plot(time, pitch, label='Тангаж')
        ax.plot(time, yaw, label='Рысканье')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('Угол (град)')
        ax.set_title('Углы')
    
    def plot_velocities(self, ax):
        """Plot velocity data"""
        time = self.extract_data('time')
        vx = self.extract_data('vx')
        vy = self.extract_data('vy')
        vz = self.extract_data('vz')
        
        ax.plot(time, vx, label='Vx')
        ax.plot(time, vy, label='Vy')
        ax.plot(time, vz, label='Vz')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('Скорость (м/с)')
        ax.set_title('Линейные скорости')

    def plot_angular_velocities(self, ax):
        """Plot angular velocity data"""
        time = self.extract_data('time')
        wx = self.extract_data('wx')
        wy = self.extract_data('wy')
        wz = self.extract_data('wz')
        
        ax.plot(time, wx, label='ωX')
        ax.plot(time, wy, label='ωY')
        ax.plot(time, wz, label='ωZ')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('Угловая скорость (град/с)')
        ax.set_title('Угловые скорости')

    def plot_motor_forces(self, ax):
        """Plot motor forces"""
        time = self.extract_data('time')
        pwm = np.array([state['motor_pwm'] for state in self.data])
        
        ax.plot(time, pwm[:, 0], label='Мотор 1')
        ax.plot(time, pwm[:, 1], label='Мотор 2')
        ax.plot(time, pwm[:, 2], label='Мотор 3')
        ax.plot(time, pwm[:, 3], label='Мотор 4')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('Сила (Н)')
        ax.set_xlabel('Время (с)')
        ax.set_title('Силы моторов')

def main():
    # Use non-interactive backend
    plt.switch_backend('agg')
    
    # Create visualizations
    visualizer = QuadcopterVisualizer('simulation_results.json')
    
    # Create figure with subplots and 3D trajectory
    fig, fig_3d = visualizer.plot_3d_trajectory()
    
    # Save both figures to the same file
    plt.figure(fig.number)
    plt.savefig('quadcopter_response_temp1.png', dpi=300, bbox_inches='tight')
    plt.figure(fig_3d.number)
    plt.savefig('quadcopter_response_temp2.png', dpi=300, bbox_inches='tight')
    
    # Combine images horizontally
    from PIL import Image
    
    # Open images
    img1 = Image.open('quadcopter_response_temp1.png')
    img2 = Image.open('quadcopter_response_temp2.png')
    
    # Create a new image with the combined width and maximum height
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    combined_img = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste images
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))
    
    # Save combined image
    combined_img.save('quadcopter_response.png', quality=95)
    
    # Clean up temporary files
    import os
    os.remove('quadcopter_response_temp1.png')
    os.remove('quadcopter_response_temp2.png')

if __name__ == "__main__":
    main()
