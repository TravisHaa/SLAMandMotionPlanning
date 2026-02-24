import numpy as np
import matplotlib.pyplot as plt


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def motion_model(state, u, dt):
    """
    Simple differential-drive-like motion model.
    state: [x, y, theta]
    u: (v, omega)
    """
    x, y, theta = state
    v, omega = u

    dx = v * dt * np.cos(theta)
    dy = v * dt * np.sin(theta)
    dtheta = omega * dt

    x_new = x + dx
    y_new = y + dy
    theta_new = wrap_angle(theta + dtheta)

    return np.array([x_new, y_new, theta_new])


class Environment:
    """2D navigation environment with obstacle landmarks."""
    
    def __init__(self, seed=42):
        """Initialize environment with landmarks as obstacles."""
        np.random.seed(seed)
        
        # Define 15 landmarks (obstacles) in the environment
        self.landmarks = {
            0: np.array([5.0, 10.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([15.0, 15.0]),
            3: np.array([0.0, 15.0]),
            4: np.array([8.0, 9.0]),
            5: np.array([3.0, 5.0]),
            6: np.array([12.0, 12.0]),
            7: np.array([18.0, 8.0]),
            8: np.array([6.0, 18.0]),
            9: np.array([14.0, 3.0]),
            10: np.array([2.0, 2.0]),
            11: np.array([16.0, 6.0]),
            12: np.array([9.0, 14.0]),
            13: np.array([4.0, 12.0]),
            14: np.array([11.0, 5.0]),
        }
        
        # Obstacle radius (landmarks are circular obstacles)
        self.obstacle_radius = 0.8
        
        # Environment bounds
        self.x_min, self.x_max = -2.0, 20.0
        self.y_min, self.y_max = -2.0, 20.0
        
        # Sensor parameters
        self.max_range = 20.0
        self.fov = np.deg2rad(180.0)
        
    def get_measurements(self, robot_pose, measurement_noise):
        """
        Get noisy measurements to visible landmarks.
        
        Args:
            robot_pose: [x, y, theta]
            measurement_noise: 2x2 covariance matrix [range, bearing]
            
        Returns:
            List of (lm_id, range, bearing) tuples
        """
        measurements = []
        x, y, theta = robot_pose
        
        for lm_id, lm_pos in self.landmarks.items():
            dx = lm_pos[0] - x
            dy = lm_pos[1] - y
            r = np.sqrt(dx * dx + dy * dy)
            
            # Check if in range
            if r > self.max_range:
                continue
            
            # Check if in field of view
            bearing = wrap_angle(np.arctan2(dy, dx) - theta)
            if abs(bearing) > self.fov / 2:
                continue
            
            # Add measurement noise
            r_meas = r + np.random.normal(0, np.sqrt(measurement_noise[0, 0]))
            b_meas = bearing + np.random.normal(0, np.sqrt(measurement_noise[1, 1]))
            b_meas = wrap_angle(b_meas)
            
            measurements.append((lm_id, r_meas, b_meas))
        
        return measurements
    
    def is_collision_free(self, point):
        """Check if a point is collision-free."""
        x, y = point[0], point[1]
        
        # Check bounds
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        
        # Check obstacles
        for lm_pos in self.landmarks.values():
            dist = np.sqrt((x - lm_pos[0])**2 + (y - lm_pos[1])**2)
            if dist < self.obstacle_radius:
                return False
        
        return True
    
    def is_path_collision_free(self, p1, p2, step_size=0.1):
        """Check if a straight line path between two points is collision-free."""
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        num_steps = int(dist / step_size) + 1
        
        for i in range(num_steps + 1):
            alpha = i / num_steps if num_steps > 0 else 1.0
            point = p1 + alpha * (p2 - p1)
            if not self.is_collision_free(point):
                return False
        
        return True
    
    def visualize(self, ax=None, show_obstacles=True):
        """Visualize the environment."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect('equal')
        ax.grid(True)
        
        if show_obstacles:
            # Draw obstacles
            for lm_id, lm_pos in self.landmarks.items():
                circle = plt.Circle((lm_pos[0], lm_pos[1]), self.obstacle_radius, 
                                   color='gray', alpha=0.5)
                ax.add_patch(circle)
                ax.text(lm_pos[0] + 0.3, lm_pos[1] + 0.3, f"L{lm_id}", fontsize=8)
        
        return ax


class RobotSimulator:
    """Simulates a differential-drive robot in the environment."""
    
    def __init__(self, environment, init_pose, process_noise, measurement_noise):
        """
        Args:
            environment: Environment instance
            init_pose: [x, y, theta]
            process_noise: 3x3 covariance matrix
            measurement_noise: 2x2 covariance matrix
        """
        self.env = environment
        self.true_pose = np.array(init_pose, dtype=float)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dt = 0.1
        
        # History
        self.true_path = [self.true_pose.copy()]
        
    def step(self, control):
        """
        Execute one simulation step.
        
        Args:
            control: (v, omega) velocity and angular velocity
            
        Returns:
            measurements: list of (lm_id, range, bearing)
        """
        v, omega = control
        
        # Add process noise to motion
        noise = np.array([
            np.random.normal(0, np.sqrt(self.process_noise[0, 0])),
            np.random.normal(0, np.sqrt(self.process_noise[1, 1])),
            np.random.normal(0, np.sqrt(self.process_noise[2, 2]))
        ])
        
        self.true_pose = motion_model(self.true_pose, control, self.dt) + noise
        self.true_pose[2] = wrap_angle(self.true_pose[2])
        
        self.true_path.append(self.true_pose.copy())
        
        # Get measurements
        measurements = self.env.get_measurements(self.true_pose, self.measurement_noise)
        
        return measurements
    
    def get_true_path(self):
        """Return numpy array of true path."""
        return np.array(self.true_path)
