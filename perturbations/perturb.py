import numpy as np

class VIMAPerturb:
    def __init__(self):
        # self.bounding_box=np.array([-0.5,0.5],[0.25,0.75])
        self.x_bounds = np.array([0.25, 0.75], dtype=np.float32)
        self.y_bounds = np.array([-0.5, 0.5], dtype=np.float32)
        
    def get_random_bounded_number_with_delta(range_bounds, target_coord, delta):
        while True:
            # Generate random x and y within the range
            random_x = np.random.uniform(range_bounds[0][0], range_bounds[0][1], 1)[0]
            random_y = np.random.uniform(range_bounds[1][0], range_bounds[1][1], 1)[0]
            
            # Check if the random point is outside the delta region
            if not (target_coord[0] - delta <= random_x <= target_coord[0] + delta and
                    target_coord[1] - delta <= random_y <= target_coord[1] + delta):
                return random_x, random_y
        
    def perturb_reflection(self, place_pose):
        return place_pose[::-1]

    def random_drop(self, ):
        x_coord = np.random.uniform(self.x_bounds[0], self.x_bounds[1], 1)[0]
        y_coord = np.random.uniform(self.y_bounds[0], self.y_bounds[1], 1)[0]
        return np.array([x_coord, y_coord], dtype=np.float32)
        

