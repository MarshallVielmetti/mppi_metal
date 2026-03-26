import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_wez_pursuit.py <path_to_trajectory.csv>")
        print("Defaulting to 'trajectory_batch.csv'")
        csv_file = "trajectory_batch.csv"
    else:
        csv_file = sys.argv[1]

    try:
        # Load header to find adversary columns
        with open(csv_file, 'r') as f:
            header = f.readline().strip().split(',')
        
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        if len(data) == 0:
            print("CSV file is empty.")
            return
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    # Infer number of adversaries and captured column from header
    # Header format: sim_id,step,x,y,theta,v,a_cmd,omega_cmd,captured,best_cost,x_adv_1,y_adv_1,th_adv_1,v_adv_1...
    adv_cols = [i for i, h in enumerate(header) if 'x_adv_' in h]
    num_adversaries = len(adv_cols)
    print(f"Found {num_adversaries} adversaries in data.")
    
    captured_col = header.index('captured') if 'captured' in header else -1
    if captured_col == -1:
        print("Warning: 'captured' column not found in CSV header.")

    sim_ids = np.unique(data[:, 0])
    
    # Visualization parameters
    goal = (5.0, 5.0)
    goal_radius = 0.2
    collision_radius = 0.5

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Plot goal
    goal_circle = Circle(goal, goal_radius, color='green', alpha=0.3, label='Goal Area')
    ax.add_patch(goal_circle)
    ax.plot(goal[0], goal[1], 'g*', markersize=10)

    # Plot a few simulations to avoid clutter
    num_to_plot = min(5, len(sim_ids))
    for i in range(num_to_plot):
        sim_id = sim_ids[i]
        sim_data = data[data[:, 0] == sim_id]
        
        # Ego trajectory
        x_ego = sim_data[:, 2]
        y_ego = sim_data[:, 3]
        ax.plot(x_ego, y_ego, label=f'Ego Sim {int(sim_id)}', alpha=0.8, linewidth=2)
        ax.plot(x_ego[0], y_ego[0], 'ko', markersize=4) # Start point

        # Plot captured points
        if captured_col != -1:
            captured_mask = sim_data[:, captured_col].astype(bool)
            if np.any(captured_mask):
                x_captured = x_ego[captured_mask]
                y_captured = y_ego[captured_mask]
                ax.plot(x_captured, y_captured, 'rx', markersize=8, label='Captured' if i == 0 else "")

        # Adversary trajectories
        for adv_idx in range(num_adversaries):
            # Column mapping: x_adv_i is at adv_cols[adv_idx]
            col_base = adv_cols[adv_idx]
            x_adv = sim_data[:, col_base]
            y_adv = sim_data[:, col_base + 1]
            
            line, = ax.plot(x_adv, y_adv, '--', alpha=0.6, label=f'Adv {adv_idx+1} Sim {int(sim_id)}')
            
            # Draw collision radius at start and end of adversary
            ax.add_patch(Circle((x_adv[0], y_adv[0]), collision_radius, color=line.get_color(), alpha=0.1, fill=True))
            ax.add_patch(Circle((x_adv[-1], y_adv[-1]), collision_radius, color=line.get_color(), alpha=0.2, fill=True))

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Pursuit-Evasion Game: Ego vs {num_adversaries} Adversaries')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_png = "pursuit_visualization.png"
    plt.savefig(output_png)
    print(f"Saved visualization to {output_png}")
    # plt.show() # Can't show in this environment, so saving to file

if __name__ == '__main__':
    main()
