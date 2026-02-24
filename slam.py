import numpy as np
from environment import wrap_angle, motion_model
import matplotlib.patches as patches


def get_covariance_ellipse(covariance, n_std=2.0):
    """
    Calculate parameters for plotting covariance ellipse.
    
    Args:
        covariance: 2x2 covariance matrix
        n_std: number of standard deviations for ellipse
        
    Returns:
        width, height, angle (in degrees)
    """
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    # Get the index of the largest eigenvalue
    largest_eigenval_idx = np.argmax(eigenvalues)
    largest_eigenval = eigenvalues[largest_eigenval_idx]
    smallest_eigenval = eigenvalues[1 - largest_eigenval_idx]
    
    # Get the eigenvector for the largest eigenvalue
    largest_eigenvec = eigenvectors[:, largest_eigenval_idx]
    
    # Calculate angle
    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
    angle = np.degrees(angle)
    
    # Calculate width and height (2 standard deviations)
    width = 2 * n_std * np.sqrt(largest_eigenval)
    height = 2 * n_std * np.sqrt(smallest_eigenval)
    
    return width, height, angle


class EKFSLAM:
    """Extended Kalman Filter SLAM implementation."""
    
    def __init__(self, init_pose, R, Q):
        """
        Args:
            init_pose: initial robot pose [x, y, theta]
            R: 3x3 process noise covariance (robot motion)
            Q: 2x2 measurement noise covariance (range, bearing)
        """
        self.mu = np.array(init_pose, dtype=float)  # [x, y, theta, m1x, m1y, ...]
        self.P = np.eye(3) * 1e-3
        self.R = R
        self.Q = Q
        self.landmark_indices = {}  # lm_id -> index of mx in state
        
        # History
        self.est_path = [self.mu[0:3].copy()]

    def predict(self, u, dt):
        """EKF prediction step."""
        v, omega = u
        x, y, theta = self.mu[0:3]

        new_pose = motion_model(self.mu[0:3], u, dt)
        self.mu[0:3] = new_pose
        theta = new_pose[2]

        # Jacobian of motion wrt robot pose
        G_r = np.eye(3)
        G_r[0, 2] = -v * dt * np.sin(theta)
        G_r[1, 2] = v * dt * np.cos(theta)

        N = self.mu.shape[0]
        Fx = np.hstack((np.eye(3), np.zeros((3, N - 3))))

        G = np.eye(N) + Fx.T @ (G_r - np.eye(3)) @ Fx
        self.P = G @ self.P @ G.T + Fx.T @ self.R @ Fx

    def add_landmark(self, lm_id, measurement):
        """Initialize new landmark in the state vector."""
        r, bearing = measurement
        x, y, theta = self.mu[0:3]

        global_bearing = wrap_angle(theta + bearing)
        mx = x + r * np.cos(global_bearing)
        my = y + r * np.sin(global_bearing)

        N = self.mu.shape[0]
        self.mu = np.concatenate([self.mu, np.array([mx, my])])

        P_new = np.zeros((N + 2, N + 2))
        P_new[:N, :N] = self.P
        P_new[N:, N:] = np.eye(2) * 1e3  # large initial uncertainty
        self.P = P_new

        self.landmark_indices[lm_id] = N

    def update(self, measurements):
        """EKF correction for list of (lm_id, r, bearing)."""
        for lm_id, r_meas, bearing_meas in measurements:
            if lm_id not in self.landmark_indices:
                self.add_landmark(lm_id, (r_meas, bearing_meas))

            idx = self.landmark_indices[lm_id]
            mx = self.mu[idx]
            my = self.mu[idx + 1]
            x, y, theta = self.mu[0:3]

            dx = mx - x
            dy = my - y
            q = dx * dx + dy * dy
            q = max(q, 1e-9)
            sqrt_q = np.sqrt(q)

            r_pred = sqrt_q
            bearing_pred = wrap_angle(np.arctan2(dy, dx) - theta)
            z_pred = np.array([r_pred, bearing_pred])

            H = np.zeros((2, self.mu.shape[0]))

            # w.r.t robot pose
            H[0, 0] = -dx / sqrt_q
            H[0, 1] = -dy / sqrt_q
            H[0, 2] = 0.0

            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1.0

            # w.r.t landmark position
            H[0, idx] = dx / sqrt_q
            H[0, idx + 1] = dy / sqrt_q
            H[1, idx] = -dy / q
            H[1, idx + 1] = dx / q

            z = np.array([r_meas, bearing_meas])
            y_k = z - z_pred
            y_k[1] = wrap_angle(y_k[1])

            S = H @ self.P @ H.T + self.Q
            K = self.P @ H.T @ np.linalg.inv(S)

            self.mu = self.mu + K @ y_k
            self.mu[2] = wrap_angle(self.mu[2])

            I = np.eye(self.mu.shape[0])
            self.P = (I - K @ H) @ self.P
    
    def step(self, control, measurements, dt=0.1):
        """Combined predict and update step."""
        self.predict(control, dt)
        self.update(measurements)
        self.est_path.append(self.mu[0:3].copy())
    
    def get_estimated_landmarks(self):
        """Return dictionary of estimated landmark positions."""
        landmarks = {}
        for lm_id, idx in self.landmark_indices.items():
            landmarks[lm_id] = np.array([self.mu[idx], self.mu[idx + 1]])
        return landmarks
    
    def get_estimated_path(self):
        """Return numpy array of estimated path."""
        return np.array(self.est_path)
    
    def get_current_pose(self):
        """Return current estimated pose."""
        return self.mu[0:3].copy()
    
    def get_landmark_covariances(self):
        """
        Return dictionary of landmark covariances.
        
        Returns:
            dict: {lm_id: 2x2 covariance matrix}
        """
        covariances = {}
        for lm_id, idx in self.landmark_indices.items():
            # Extract 2x2 covariance for this landmark
            cov = self.P[idx:idx+2, idx:idx+2]
            covariances[lm_id] = cov
        return covariances
    
    def get_pose_covariance(self):
        """Return 2x2 position covariance of robot (x, y only)."""
        return self.P[0:2, 0:2]
