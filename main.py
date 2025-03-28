
from process_yolo_dataset import process_yolo_subset, update_yaml, get_LUT
import os
import time
if __name__ == "__main__":
    dataset_dir = "./top_view_person_dataset"
    output_dir = "./fisheye_top_view_person_dataset"
    subsets = ['train',  'test', 'valid']
    strength = 0.65
    start_time = time.perf_counter()
    map_x, map_y = get_LUT(dataset_dir, strength)
    for subset in subsets:
        images_dir = os.path.join(dataset_dir, subset, 'images')
        labels_dir = os.path.join(dataset_dir, subset, 'labels')
        output_images_dir = os.path.join(output_dir, subset, 'images')
        output_labels_dir = os.path.join(output_dir, subset, 'labels')

        process_yolo_subset(images_dir, labels_dir, output_images_dir, output_labels_dir, map_x, map_y)

    input_yaml = os.path.join(dataset_dir, 'data.yaml')
    output_yaml = os.path.join(output_dir, 'data.yaml')
    update_yaml(input_yaml, output_yaml, output_dir)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")