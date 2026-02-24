import numpy as np
from collections import defaultdict
import heapq


class PRM:
    """Probabilistic Roadmap planner."""
    
    def __init__(self, environment, n_samples=500, k_neighbors=10):
        """
        Args:
            environment: Environment instance
            n_samples: number of random samples
            k_neighbors: number of nearest neighbors to connect
        """
        self.env = environment
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        
        self.nodes = []
        self.edges = defaultdict(list)
        self.roadmap_built = False
        
    def build_roadmap(self, visualize_callback=None):
        """Build the probabilistic roadmap."""
        print(f"Building PRM roadmap with {self.n_samples} samples...")
        
        # Sample collision-free nodes
        attempts = 0
        max_attempts = self.n_samples * 10
        
        while len(self.nodes) < self.n_samples and attempts < max_attempts:
            x = np.random.uniform(self.env.x_min, self.env.x_max)
            y = np.random.uniform(self.env.y_min, self.env.y_max)
            point = np.array([x, y])
            
            if self.env.is_collision_free(point):
                self.nodes.append(point)
                
                # Visualization callback during sampling
                if visualize_callback and len(self.nodes) % 20 == 0:
                    visualize_callback('sampling', len(self.nodes), self.n_samples, 
                                     self.nodes, self.edges, None, None)
            
            attempts += 1
        
        print(f"Sampled {len(self.nodes)} collision-free nodes")
        
        # Connect k-nearest neighbors
        total_connections = len(self.nodes)
        for i, node in enumerate(self.nodes):
            # Find k nearest neighbors
            distances = [np.linalg.norm(node - other) for other in self.nodes]
            nearest_indices = np.argsort(distances)[1:self.k_neighbors+1]
            
            for j in nearest_indices:
                if self.env.is_path_collision_free(node, self.nodes[j]):
                    self.edges[i].append(j)
                    self.edges[j].append(i)
            
            # Visualization callback during connection
            if visualize_callback and i % 10 == 0:
                visualize_callback('connecting', i, total_connections,
                                 self.nodes, self.edges, i, None)
        
        self.roadmap_built = True
        print(f"Roadmap built with {sum(len(v) for v in self.edges.values()) // 2} edges")
    
    def plan(self, start, goal, visualize_callback=None):
        """
        Plan a path from start to goal using A*.
        
        Args:
            start: [x, y] start position
            goal: [x, y] goal position
            visualize_callback: function for visualization
            
        Returns:
            path: list of [x, y] waypoints, or None if no path found
        """
        if not self.roadmap_built:
            self.build_roadmap(visualize_callback)
        
        # Add start and goal to roadmap temporarily
        start_idx = len(self.nodes)
        goal_idx = len(self.nodes) + 1
        
        temp_nodes = self.nodes + [start, goal]
        temp_edges = defaultdict(list, self.edges)
        
        # Connect start to nearby nodes
        for i, node in enumerate(self.nodes):
            if self.env.is_path_collision_free(start, node):
                temp_edges[start_idx].append(i)
                temp_edges[i].append(start_idx)
        
        # Connect goal to nearby nodes
        for i, node in enumerate(self.nodes):
            if self.env.is_path_collision_free(goal, node):
                temp_edges[goal_idx].append(i)
                temp_edges[i].append(goal_idx)
        
        # A* search
        def heuristic(idx):
            return np.linalg.norm(temp_nodes[idx] - goal)
        
        open_set = [(heuristic(start_idx), 0, start_idx)]
        came_from = {}
        g_score = {start_idx: 0}
        closed_set = set()
        
        search_iteration = 0
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Visualization callback during search
            if visualize_callback and search_iteration % 5 == 0:
                visualize_callback('searching', search_iteration, len(temp_nodes),
                                 temp_nodes, temp_edges, current, list(closed_set))
            
            search_iteration += 1
            
            if current == goal_idx:
                # Reconstruct path
                path = [goal]
                while current in came_from:
                    current = came_from[current]
                    path.append(temp_nodes[current])
                path.reverse()
                return np.array(path)
            
            for neighbor in temp_edges[current]:
                if neighbor in closed_set:
                    continue
                    
                tentative_g = current_g + np.linalg.norm(
                    temp_nodes[current] - temp_nodes[neighbor]
                )
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
        
        return None  # No path found


class RRT:
    """Rapidly-exploring Random Tree planner."""
    
    def __init__(self, environment, max_iter=5000, step_size=0.5, goal_sample_rate=0.1):
        """
        Args:
            environment: Environment instance
            max_iter: maximum iterations
            step_size: maximum step size for tree extension
            goal_sample_rate: probability of sampling goal
        """
        self.env = environment
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        
        # For visualization
        self.nodes = []
        self.parents = {}
        self.tree_built = False
        
    def plan(self, start, goal, goal_tolerance=0.5, visualize_callback=None):
        """
        Plan a path from start to goal.
        
        Args:
            start: [x, y] start position
            goal: [x, y] goal position
            goal_tolerance: distance to goal considered success
            visualize_callback: function(nodes, parents, iteration) for visualization
            
        Returns:
            path: list of [x, y] waypoints, or None if no path found
        """
        print(f"Planning with RRT (max {self.max_iter} iterations)...")
        
        # Tree structure
        self.nodes = [start]
        self.parents = {0: None}
        
        for i in range(self.max_iter):
            # Sample random point (sometimes sample goal)
            if np.random.random() < self.goal_sample_rate:
                rand_point = goal
            else:
                x = np.random.uniform(self.env.x_min, self.env.x_max)
                y = np.random.uniform(self.env.y_min, self.env.y_max)
                rand_point = np.array([x, y])
            
            # Find nearest node in tree
            distances = [np.linalg.norm(node - rand_point) for node in self.nodes]
            nearest_idx = np.argmin(distances)
            nearest_node = self.nodes[nearest_idx]
            
            # Steer towards random point
            direction = rand_point - nearest_node
            distance = np.linalg.norm(direction)
            
            if distance > self.step_size:
                direction = direction / distance * self.step_size
            
            new_node = nearest_node + direction
            
            # Check if new node is collision-free
            if self.env.is_collision_free(new_node) and \
               self.env.is_path_collision_free(nearest_node, new_node):
                new_idx = len(self.nodes)
                self.nodes.append(new_node)
                self.parents[new_idx] = nearest_idx
                
                # Visualization callback
                if visualize_callback and i % 10 == 0:
                    visualize_callback(self.nodes, self.parents, i, rand_point, nearest_idx, new_node)
                
                # Check if goal reached
                if np.linalg.norm(new_node - goal) < goal_tolerance:
                    print(f"Goal reached in {i+1} iterations!")
                    
                    # Reconstruct path
                    path = [goal, new_node]
                    current_idx = new_idx
                    while self.parents[current_idx] is not None:
                        current_idx = self.parents[current_idx]
                        path.append(self.nodes[current_idx])
                    path.reverse()
                    
                    self.tree_built = True
                    return np.array(path)
        
        print("RRT failed to find path")
        self.tree_built = True
        return None

