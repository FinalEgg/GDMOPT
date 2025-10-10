import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import os
import glob
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from env.cellfree.config import X, Y, H, M, N, P
from script.simulation.sim_config import LAUNCH_INTERVAL, DRONE_SPEED, TIME_STEP, NUM_CHARGING_STATIONS
from script.simulation.drone_path import drone_path
from script.simulation.model_prediction import model_prediction, load_model
from env.cellfree.env import CellFreeEnv
from model.actor import Actor
from model.sac import Actor as SACActor
from model.diffusion import MLP
from policy import DDPG, SAC
import threading
import time

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell-free UAV Network Simulation")
        self.root.geometry("1200x800")

        # Simulation parameters from sim_config
        self.launch_interval = LAUNCH_INTERVAL
        self.drone_speed = DRONE_SPEED
        self.time_step = TIME_STEP

        # Initialize positions
        self.start_pos = np.array([0.0, 0.0, 0.0])
        self.end_pos = np.array([float(X), float(Y), 0.0])
        self.bs_positions = np.random.uniform(0, [X, Y, 0], (M, 3)).astype(float)
        self.charging_stations = np.random.uniform(0, [X, Y, 0], (NUM_CHARGING_STATIONS, 3)).astype(float)

        # Drones
        self.drones = []  # List of drone dicts: {'pos': np.array, 'path': list, 'path_index': int, 'launched': bool, 'arrived': bool, 'color': str}

        # Colors for drones
        self.drone_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # Model
        self.current_model = None
        self.model_type = None

        self.create_widgets()
        self.setup_plot()
        self.running = False

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                       values=['ddpg', 'sac', 'diffusion'], state='readonly')
        self.model_combo.grid(row=0, column=1, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

        # Weight selection
        ttk.Label(control_frame, text="Weights:").grid(row=0, column=2, padx=5)
        self.weight_var = tk.StringVar()
        self.weight_combo = ttk.Combobox(control_frame, textvariable=self.weight_var, state='readonly')
        self.weight_combo.grid(row=0, column=3, padx=5)

        # Start/Pause button
        self.start_button = ttk.Button(control_frame, text="Start Simulation", command=self.toggle_simulation)
        self.start_button.grid(row=0, column=4, padx=20)

        # Plot frame
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot()

    def update_plot(self):
        self.ax.clear()

        # Set grid
        self.ax.set_xlim(0, X)
        self.ax.set_ylim(0, Y)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Plot start and end
        self.ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=15, label='Start')
        self.ax.plot(self.end_pos[0], self.end_pos[1], 'ro', markersize=15, label='End')

        # Plot base stations
        for i, bs in enumerate(self.bs_positions):
            self.ax.plot(bs[0], bs[1], 'b^', markersize=12)
            self.ax.text(bs[0], bs[1], f'BS{i}', ha='center', va='bottom')

        # Plot charging stations
        for i, cs in enumerate(self.charging_stations):
            self.ax.plot(cs[0], cs[1], 'ys', markersize=10)
            self.ax.text(cs[0], cs[1], f'CS{i}', ha='center', va='bottom')

        # Plot drones and connections
        for drone in self.drones:
            if drone['launched']:
                # Plot path
                path = drone['path']
                if len(path) > 1:
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    self.ax.plot(path_x, path_y, color=drone['color'], linewidth=2, linestyle='--', alpha=0.7)

                if not drone['arrived']:
                    pos = drone['pos']
                    self.ax.plot(pos[0], pos[1], marker='D', markersize=8, color=drone['color'])

                    # Plot connections
                    if self.current_model is not None:
                        connections = self.get_connections(pos)
                        # connections is (N*M,) array, first M values are for the first UAV
                        for bs_idx in range(M):
                            strength = connections[bs_idx]
                            if strength > 0.1:  # Only show significant connections
                                bs = self.bs_positions[bs_idx]
                                alpha = min(strength, 1.0)
                                self.ax.plot([pos[0], bs[0]], [pos[1], bs[1]],
                                           '--', alpha=alpha, linewidth=2, color='purple')

        self.ax.legend()
        self.canvas.draw()

    def on_model_select(self, event):
        model_type = self.model_var.get()
        self.model_type = model_type

        # Find available weights
        log_path = f'log/default/{model_type}/cellfree'
        if os.path.exists(log_path):
            weight_files = glob.glob(os.path.join(log_path, '**/policy.pth'), recursive=True)
            weight_options = [os.path.relpath(f, log_path) for f in weight_files]
        else:
            weight_options = []

        self.weight_combo['values'] = weight_options
        if weight_options:
            self.weight_combo.set(weight_options[0])

    def load_model(self):
        if not self.model_var.get() or not self.weight_var.get():
            return False

        model_type = self.model_var.get()
        weight_path = f'log/default/{model_type}/cellfree/{self.weight_var.get()}'

        self.current_model = load_model(model_type, weight_path)
        return self.current_model is not None

    def get_connections(self, drone_pos):
        # Create state vector: BS positions, UAV positions (N drones), connection matrix, power matrix
        uav_positions = np.tile(self.start_pos, (N, 1))  # Default all UAVs at start
        # Set the current drone position
        uav_positions[0] = drone_pos  # Assume first drone is the active one

        state = np.concatenate([
            self.bs_positions.flatten(),
            uav_positions.flatten(),
            np.zeros(M*N),  # Connection matrix
            np.zeros(M*N)   # Power matrix
        ])

        # Use model_prediction function from model_prediction.py
        return model_prediction(self.current_model, self.model_type, state)

    def drone_path(self, start, end):
        # Use the drone_path function from drone_path.py
        return drone_path(start, end, self.charging_stations)

    def launch_drone(self):
        if len(self.drones) < N:
            path = self.drone_path(self.start_pos, self.end_pos)
            color = self.drone_colors[len(self.drones)]
            drone = {
                'pos': self.start_pos.copy().astype(float),
                'path': path,
                'path_index': 0,
                'launched': True,
                'arrived': False,
                'color': color
            }
            self.drones.append(drone)

    def update_drones(self):
        for drone in self.drones:
            if not drone['arrived']:
                path = drone['path']
                idx = drone['path_index']

                if idx < len(path) - 1:
                    target = path[idx + 1]
                    direction = target - drone['pos']
                    distance = np.linalg.norm(direction)

                    if distance > self.drone_speed * self.time_step:
                        direction = direction / distance
                        drone['pos'] += direction * self.drone_speed * self.time_step
                    else:
                        drone['pos'] = target.copy()
                        drone['path_index'] += 1

                        if idx + 1 == len(path) - 1:
                            drone['arrived'] = True

    def simulation_loop(self):
        if self.running:
            self.update_drones()

            # Launch new drones
            if len(self.drones) < N and (len(self.drones) == 0 or time.time() - self.last_launch > self.launch_interval):
                self.launch_drone()
                self.last_launch = time.time()

            self.update_plot()

            # Check if all drones arrived
            if all(d['arrived'] for d in self.drones) and len(self.drones) == N:
                self.running = False
                self.start_button.config(text="Start Simulation")

            self.root.after(int(self.time_step * 1000), self.simulation_loop)

    def toggle_simulation(self):
        if not self.running:
            if self.load_model():
                self.running = True
                self.start_button.config(text="Pause Simulation")
                self.last_launch = time.time()
                self.drones = []
                self.simulation_loop()
            else:
                tk.messagebox.showerror("Error", "Failed to load model")
        else:
            self.running = False
            self.start_button.config(text="Start Simulation")

    def on_closing(self):
        self.running = False
        self.root.quit()

def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
