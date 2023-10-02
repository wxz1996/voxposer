import numpy as np
from gymnasium_robotics.envs.franka_kitchen import KitchenEnv
from numba import njit


class FrankaKitchenVisionEnv(KitchenEnv):
    
    def __init__(
        self,
        tasks_to_complete: str,
        terminate_on_tasks_completed: bool = True,
        remove_task_when_completed: bool = True,
        object_noise_ratio: float = 0.0005,
        **kwargs,
    ):
        super().__init__(tasks_to_complete, 
                         terminate_on_tasks_completed, remove_task_when_completed, 
                         object_noise_ratio, **kwargs)
        model = self.robot_env.model
        fovy = model.cam_fovy[self.robot_env.camera_id][0][0]
        img = self.render(render_mode='rgb_array')
        height, width, _ = img.shape
        self.K = self.compute_intrinsics(fovy, width, height)
    
    def render(self, render_mode):
        rendered_image = self.robot_env.mujoco_renderer.render(
            render_mode, 
            self.robot_env.camera_id, 
            self.robot_env.camera_name
        )
        
        if not render_mode == 'human':
            if render_mode == 'depth_array':
                rendered_image = self.z_buffer_to_depth(rendered_image)
                rendered_image = np.clip(rendered_image, 0, 3)
            return rendered_image
        
    def compute_intrinsics(self, fov_degrees, w, h):
        """
        Compute the intrinsic matrix given FOV and image dimensions.
        
        Parameters:
        - fov_degrees: Field of view in degrees.
        - w: Image width in pixels.
        - h: Image height in pixels.
        
        Returns:
        - K: 3x3 intrinsic matrix.
        """
        # Convert FOV from degrees to radians
        fov_radians = np.deg2rad(fov_degrees)
        
        # Estimate focal length using the pinhole camera model
        fx = fy = w / (2 * np.tan(fov_radians / 2))
        
        # Principal point (usually the image center)
        cx, cy = w / 2, h / 2
        
        # Intrinsic matrix
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        
        return K
        
    def z_buffer_to_depth(self, z_buffer):
        # https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L734
        model = self.robot_env.model
        extent = model.stat.extent
        near = model.vis.map.znear * extent
        far = model.vis.map.zfar * extent
        # Convert from [0 1] to depth in meters, see links below:
        # http://stackoverflow.com/a/6657284/1461210
        # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
        image = near / (1 - z_buffer * (1 - near / far))
        return image
    
    @staticmethod
    @njit(cache=True)
    def project_rgbd_to_pointcloud(depth_image, rgb_image, K_inv):
        height, width = depth_image.shape[:2]
        
        # Initialize arrays
        u = np.empty((height, width), dtype=np.int32)
        v = np.empty((height, width), dtype=np.int32)
        uv_homogeneous = np.empty((height, width, 3))
        
        # Create meshgrid of image coordinates using loops
        for i in range(height):
            for j in range(width):
                u[i, j] = j
                v[i, j] = i
                uv_homogeneous[i, j, 0] = j
                uv_homogeneous[i, j, 1] = i
                uv_homogeneous[i, j, 2] = 1
        
        # Project to 3D
        xyz = np.empty((height, width, 3))
        for i in range(height):
            for j in range(width):
                sum0 = sum1 = sum2 = 0.0
                for k in range(3):
                    sum0 += K_inv[0, k] * uv_homogeneous[i, j, k]
                    sum1 += K_inv[1, k] * uv_homogeneous[i, j, k]
                    sum2 += K_inv[2, k] * uv_homogeneous[i, j, k]
                xyz[i, j, 0] = sum0 * depth_image[i, j]
                xyz[i, j, 1] = sum1 * depth_image[i, j]
                xyz[i, j, 2] = sum2 * depth_image[i, j]
        
        # Flatten and stack with RGB values
        valid_count = np.sum(depth_image > 0)
        result = np.empty((valid_count, 6))
        idx = 0
        for i in range(height):
            for j in range(width):
                if depth_image[i, j] > 0:
                    result[idx, 0:3] = xyz[i, j]
                    result[idx, 3:6] = rgb_image[i, j]
                    idx += 1
        
        return result