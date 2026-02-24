#!/usr/bin/env python3
"""
Animated visualization of PRM and RRT planning algorithms.
Shows step-by-step how each algorithm builds its data structure and finds a path.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from environment import Environment
from motion_planning import PRM, RRT


class PRMVisualizer:
    """Visualize PRM algorithm step-by-step."""
    
    def __init__(self, env, start, goal):
        self.env = env
        self.start = start
        self.goal = goal
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.ion()
        
    def visualize(self, phase, progress, total, nodes, edges, current_node, visited):
        """Callback for PRM visualization."""
        self.ax.clear()
        
        # Set title based on phase
        if phase == 'sampling':
            title = f"PRM: Sampling ({progress}/{total} nodes)"
        elif phase == 'connecting':
            title = f"PRM: Connecting Neighbors ({progress}/{total} nodes)"
        else:  # searching
            title = f"PRM: A* Search (iteration {progress})"
        
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlim(self.env.x_min, self.env.x_max)
        self.ax.set_ylim(self.env.y_min, self.env.y_max)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for lm_pos in self.env.landmarks.values():
            circle = Circle((lm_pos[0], lm_pos[1]), self.env.obstacle_radius,
                          color='gray', alpha=0.4)
            self.ax.add_patch(circle)
        
        # Draw sampled nodes
        if nodes:
            nodes_array = np.array(nodes)
            self.ax.plot(nodes_array[:, 0], nodes_array[:, 1], 'c.', 
                        markersize=4, alpha=0.5, label='Sampled nodes')
        
        # Draw edges (if in connecting or searching phase)
        if phase in ['connecting', 'searching'] and edges:
            for i in range(len(nodes)):
                for j in edges[i]:
                    if j < len(nodes):  # Safety check
                        self.ax.plot([nodes[i][0], nodes[j][0]], 
                                   [nodes[i][1], nodes[j][1]],
                                   'c-', linewidth=0.3, alpha=0.2)
        
        # Highlight current node during connecting
        if phase == 'connecting' and current_node is not None and current_node < len(nodes):
            node = nodes[current_node]
            self.ax.plot(node[0], node[1], 'yo', markersize=8, 
                        label='Current node', zorder=5)
            # Highlight its edges
            for j in edges[current_node]:
                if j < len(nodes):
                    self.ax.plot([node[0], nodes[j][0]], [node[1], nodes[j][1]],
                               'y-', linewidth=1.5, alpha=0.6, zorder=4)
        
        # Highlight visited nodes during search
        if phase == 'searching' and visited:
            for idx in visited:
                if idx < len(nodes):
                    self.ax.plot(nodes[idx][0], nodes[idx][1], 'mo', 
                               markersize=4, alpha=0.6)
        
        # Highlight current search node
        if phase == 'searching' and current_node is not None and current_node < len(nodes):
            node = nodes[current_node]
            self.ax.plot(node[0], node[1], 'ro', markersize=10,
                        label='Current search', zorder=6)
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, 
                    label='Start', zorder=7)
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, 
                    label='Goal', zorder=7)
        
        # Add info box
        if phase == 'sampling':
            info = f"Sampling collision-free nodes\nProgress: {progress}/{total}"
        elif phase == 'connecting':
            edge_count = sum(len(v) for v in edges.values()) // 2
            info = f"Connecting k-nearest neighbors\nEdges created: {edge_count}"
        else:
            info = f"A* search in progress\nNodes explored: {len(visited) if visited else 0}"
        
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()


class RRTVisualizer:
    """Visualize RRT algorithm step-by-step."""
    
    def __init__(self, env, start, goal):
        self.env = env
        self.start = start
        self.goal = goal
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.ion()
        
    def visualize(self, nodes, parents, iteration, rand_point, nearest_idx, new_node):
        """Callback for RRT visualization."""
        self.ax.clear()
        self.ax.set_title(f"RRT: Growing Tree (iteration {iteration})", 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlim(self.env.x_min, self.env.x_max)
        self.ax.set_ylim(self.env.y_min, self.env.y_max)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for lm_pos in self.env.landmarks.values():
            circle = Circle((lm_pos[0], lm_pos[1]), self.env.obstacle_radius,
                          color='gray', alpha=0.4)
            self.ax.add_patch(circle)
        
        # Draw tree edges
        for i, parent_idx in parents.items():
            if parent_idx is not None:
                self.ax.plot([nodes[i][0], nodes[parent_idx][0]],
                           [nodes[i][1], nodes[parent_idx][1]],
                           'b-', linewidth=0.8, alpha=0.4)
        
        # Draw tree nodes
        if nodes:
            nodes_array = np.array(nodes)
            self.ax.plot(nodes_array[:, 0], nodes_array[:, 1], 'bo', 
                        markersize=3, alpha=0.6, label='Tree nodes')
        
        # Highlight the latest extension
        if new_node is not None and nearest_idx is not None:
            nearest_node = nodes[nearest_idx]
            # Draw random sample
            self.ax.plot(rand_point[0], rand_point[1], 'rx', markersize=10,
                        label='Random sample', alpha=0.7)
            # Draw nearest node
            self.ax.plot(nearest_node[0], nearest_node[1], 'yo', markersize=8,
                        label='Nearest node', zorder=5)
            # Draw new node
            self.ax.plot(new_node[0], new_node[1], 'go', markersize=8,
                        label='New node', zorder=5)
            # Draw extension line
            self.ax.plot([nearest_node[0], new_node[0]],
                       [nearest_node[1], new_node[1]],
                       'g-', linewidth=2, alpha=0.8, zorder=4)
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, 
                    label='Start', zorder=7)
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, 
                    label='Goal', zorder=7)
        
        # Add info box
        info = f"Tree nodes: {len(nodes)}\nIteration: {iteration}"
        if new_node is not None:
            dist_to_goal = np.linalg.norm(new_node - self.goal)
            info += f"\nDist to goal: {dist_to_goal:.2f}m"
        
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()


def demo_prm(env, start, goal):
    """Run PRM with visualization."""
    print("\n" + "="*60)
    print("PRM (Probabilistic Roadmap) Visualization")
    print("="*60)
    print("\nPhases:")
    print("  1. Sampling: Generate random collision-free nodes")
    print("  2. Connecting: Link k-nearest neighbors")
    print("  3. Searching: A* search on the roadmap")
    print("="*60 + "\n")
    
    visualizer = PRMVisualizer(env, start, goal)
    prm = PRM(env, n_samples=300, k_neighbors=10)
    
    try:
        path = prm.plan(start, goal, visualize_callback=visualizer.visualize)
        
        if path is not None:
            print(f"\n✓ Path found!")
            print(f"  Length: {sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)):.2f}m")
            print(f"  Waypoints: {len(path)}")
            
            # Show final result with path
            visualizer.visualize('final', 0, 0, prm.nodes, prm.edges, None, None)
            path_array = np.array(path)
            visualizer.ax.plot(path_array[:, 0], path_array[:, 1], 'b-', 
                             linewidth=3, label='Found path', zorder=6)
            visualizer.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            print("\nFinal roadmap displayed. Press Ctrl+C or close window to continue...")
            plt.pause(3)
        else:
            print("\n✗ No path found")
            plt.pause(2)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def demo_rrt(env, start, goal):
    """Run RRT with visualization."""
    print("\n" + "="*60)
    print("RRT (Rapidly-exploring Random Tree) Visualization")
    print("="*60)
    print("\nHow it works:")
    print("  1. Sample random point (or goal)")
    print("  2. Find nearest tree node")
    print("  3. Extend tree toward sample")
    print("  4. Repeat until goal reached")
    print("="*60 + "\n")
    
    visualizer = RRTVisualizer(env, start, goal)
    rrt = RRT(env, max_iter=2000, step_size=0.5, goal_sample_rate=0.1)
    
    try:
        path = rrt.plan(start, goal, goal_tolerance=0.5, 
                       visualize_callback=visualizer.visualize)
        
        if path is not None:
            print(f"\n✓ Path found!")
            print(f"  Length: {sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)):.2f}m")
            print(f"  Waypoints: {len(path)}")
            
            # Show final result with path
            visualizer.ax.clear()
            visualizer.ax.set_title("RRT: Final Path", fontsize=14, fontweight='bold')
            visualizer.ax.set_xlim(env.x_min, env.x_max)
            visualizer.ax.set_ylim(env.y_min, env.y_max)
            visualizer.ax.set_aspect('equal')
            visualizer.ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for lm_pos in env.landmarks.values():
                circle = Circle((lm_pos[0], lm_pos[1]), env.obstacle_radius,
                              color='gray', alpha=0.4)
                visualizer.ax.add_patch(circle)
            
            # Draw tree (faded)
            for i, parent_idx in rrt.parents.items():
                if parent_idx is not None:
                    visualizer.ax.plot([rrt.nodes[i][0], rrt.nodes[parent_idx][0]],
                                      [rrt.nodes[i][1], rrt.nodes[parent_idx][1]],
                                      'b-', linewidth=0.5, alpha=0.2)
            
            # Draw path
            path_array = np.array(path)
            visualizer.ax.plot(path_array[:, 0], path_array[:, 1], 'g-', 
                             linewidth=3, label='Found path', zorder=6)
            visualizer.ax.plot(start[0], start[1], 'go', markersize=15, 
                             label='Start', zorder=7)
            visualizer.ax.plot(goal[0], goal[1], 'r*', markersize=20, 
                             label='Goal', zorder=7)
            visualizer.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            print("\nFinal path displayed. Press Ctrl+C or close window to continue...")
            plt.pause(3)
        else:
            print("\n✗ No path found")
            plt.pause(2)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def main():
    """Run both PRM and RRT demos."""
    print("\n" + "="*60)
    print("MOTION PLANNING ALGORITHM VISUALIZATION")
    print("="*60)
    print("\nThis demo shows step-by-step visualization of:")
    print("  • PRM (Probabilistic Roadmap)")
    print("  • RRT (Rapidly-exploring Random Tree)")
    print("\nWatch how each algorithm builds its structure!")
    print("="*60)
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])
    
    print(f"\nPlanning from {start} to {goal}")
    print(f"Environment: {len(env.landmarks)} circular obstacles")
    
    # Demo PRM
    demo_prm(env, start, goal)
    
    # Demo RRT
    demo_rrt(env, start, goal)
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
