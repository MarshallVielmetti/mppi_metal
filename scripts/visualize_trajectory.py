import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_trajectory.py <path_to_trajectory.csv>")
        print("Defaulting to 'trajectory.csv'")
        csv_file = "trajectory.csv"
    else:
        csv_file = sys.argv[1]

    try:
        # Format: sim_id,step,x,y,theta,v,a_cmd,omega_cmd,best_cost
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        if len(data) == 0:
            print("CSV file is empty.")
            return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    sim_ids = np.unique(data[:, 0])

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # We will plot true trajectories lightly
    colors = plt.cm.viridis(np.linspace(0, 1, len(sim_ids)))
    
    for i, sim_id in enumerate(sim_ids):
        sim_data = data[data[:, 0] == sim_id]
        
        step = sim_data[:, 1]
        x = sim_data[:, 2]
        y = sim_data[:, 3]
        theta = sim_data[:, 4]
        v = sim_data[:, 5]
        a_cmd = sim_data[:, 6]
        omega_cmd = sim_data[:, 7]
        cost = sim_data[:, 8]
        
        # Plot Trajectory
        ax1.plot(x, y, color='blue', alpha=0.15)
        
        # Plot Controls
        ax2.plot(step, a_cmd, color='red', alpha=0.1)
        ax2.plot(step, omega_cmd, color='orange', alpha=0.1)
        
        # Plot States
        ax3.plot(step, v, color='green', alpha=0.1)
        ax3.plot(step, theta, color='purple', alpha=0.1)
        
        # Plot Cost
        ax4.plot(step, cost, color='brown', alpha=0.1)
        
        if i == 0: # Highlight one simulation and add labels/quivers
            ax1.plot(x, y, color='darkblue', alpha=1.0, label='Sim 0 Trajectory')
            ax1.plot(x[0], y[0], 'go', label='Start', markersize=8)
            ax1.plot(x[-1], y[-1], 'ro', label='End', markersize=8)
            stride = max(1, len(x) // 10)
            ax1.quiver(x[::stride], y[::stride], np.cos(theta[::stride]), np.sin(theta[::stride]), 
                       color='black', alpha=0.8, width=0.005, label='Heading (Sim 0)')
                       
            ax2.plot(step, a_cmd, color='darkred', alpha=1.0, label='a_cmd (Sim 0)')
            ax2.plot(step, omega_cmd, color='darkorange', alpha=1.0, label='omega_cmd (Sim 0)')
            
            ax3.plot(step, v, color='darkgreen', alpha=1.0, label='Velocity (Sim 0)')
            ax3.plot(step, theta, color='indigo', alpha=1.0, label='Heading (Sim 0)')
            
            ax4.plot(step, cost, color='maroon', alpha=1.0, label='Best Cost (Sim 0)')

    # Labels and dressing
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Vehicle Trajectory ({len(sim_ids)} Simulations)')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')

    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Command Value')
    ax2.set_title('Control Inputs over Time')
    ax2.grid(True)
    ax2.legend()

    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('State Value')
    ax3.set_title('Velocity & Heading over Time')
    ax3.grid(True)
    ax3.legend()

    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Cost')
    ax4.set_title('Diagnostic: Best Rollout Cost')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
