import cv2
import numpy as np

def resize_to_square(img: np.ndarray) -> np.ndarray:

    size = max(img.shape[:2])
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def create_fisheye_mapping(h, w, strength=0.6):
    
    center = (w // 2, h // 2)
    R_max = min(center)
    out_size = 2 * R_max
    new_center = (out_size // 2, out_size // 2)
    
    map_x = np.zeros((out_size, out_size), dtype=np.float32)
    map_y = np.zeros((out_size, out_size), dtype=np.float32)

    for y in range(out_size):
        for x in range(out_size):
            dx, dy = x - new_center[0], y - new_center[1]
            r = np.sqrt(dx**2 + dy**2)
            
            if r > R_max:
                map_x[y, x] = -1
                map_y[y, x] = -1
                continue

            theta = np.arctan2(dy, dx)
            r_dist = np.tan((r / R_max) * (np.pi / 2) * strength) * R_max / np.tan((np.pi / 2) * strength)

            src_x = center[0] + r_dist * np.cos(theta)
            src_y = center[1] + r_dist * np.sin(theta)

            if 0 <= src_x < w and 0 <= src_y < h:
                map_x[y, x] = src_x
                map_y[y, x] = src_y
            else:
                map_x[y, x] = -1
                map_y[y, x] = -1
    
    return map_x, map_y

def apply_fisheye(image, map_x, map_y, interpolation=cv2.INTER_LINEAR):
   
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

def create_LUT_table(image, distortion_strength=0.6):

    h, w = image.shape[:2]
    map_x, map_y = create_fisheye_mapping(h, w,distortion_strength)
    return map_x, map_y
