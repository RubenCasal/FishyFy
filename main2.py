import os
import cv2
import yaml
from fisheye_transformation import resize_to_square, create_LUT_table, apply_fisheye
from generate_new_bboxes import (
    yolo_to_absolute,
    generate_bbox_mask,
    find_fisheye_yolo_bbox,
    load_yolo_bboxes,
)

def process_yolo_subset(images_dir, labels_dir, output_images_dir, output_labels_dir, strength=0.65):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"❌ No se encontraron imágenes en {images_dir}.")
        return

    first_image_path = os.path.join(images_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"❌ Error: no se pudo cargar la imagen {first_image_path}")
        return

    first_image_square = resize_to_square(first_image)
    target_size = first_image_square.shape[:2]
    map_x, map_y = create_LUT_table(first_image_square, distortion_strength=strength)

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"⚠ El archivo de etiquetas {label_path} no existe. Se omite la imagen {image_file}.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: no se pudo cargar la imagen {image_path}")
            continue

        image_square = resize_to_square(image)
        image_resized = cv2.resize(image_square, (target_size[1], target_size[0]))
        h, w = image_resized.shape[:2]

        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        bboxes_yolo = load_yolo_bboxes(label_path)
        if not bboxes_yolo:
            print(f"❌ No se encontraron etiquetas válidas en el archivo {label_path}")
            continue

        bboxes_absolute = yolo_to_absolute(bboxes_yolo, w, h)

        masks = generate_bbox_mask(image_resized.shape, bboxes_absolute)

        circle_center = (w // 2, h // 2)
        circle_radius = min(circle_center)

        fisheye_image = apply_fisheye(image_gray, map_x, map_y)

        new_bboxes = []
        for i, mask in enumerate(masks):
            cls_id = int(bboxes_absolute[i][0])
            distorted_mask = apply_fisheye(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
            bbox = find_fisheye_yolo_bbox(distorted_mask, w, h, circle_center, circle_radius)
            if bbox:
                new_bboxes.append([cls_id] + bbox)
            else:
                print(f"⚠ BBox {i} descartada tras la transformación.")

        output_image_path = os.path.join(output_images_dir, image_file)
        cv2.imwrite(output_image_path, fisheye_image)

        output_label_path = os.path.join(output_labels_dir, label_file)
        with open(output_label_path, "w") as f:
            for bbox in new_bboxes:
                f.write(" ".join(f"{val:.6f}" for val in bbox) + "\n")

        print(f"✅ Procesado {image_file}: imagen y etiquetas guardadas.")

def update_yaml(input_yaml, output_yaml, output_dir):
    with open(input_yaml, "r") as f:
        data = yaml.safe_load(f)

    data['train'] = os.path.join(output_dir, "train")
    data['val'] = os.path.join(output_dir, "val")
    if 'test' in data:
        data['test'] = os.path.join(output_dir, "test")

    with open(output_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("✅ Archivo data.yaml actualizado.")

if __name__ == "__main__":
    dataset_dir = "./top_view_person_dataset"
    output_dir = "./dataset_transformado"
    subsets = ['train',  'test']

    for subset in subsets:
        images_dir = os.path.join(dataset_dir, subset, 'images')
        labels_dir = os.path.join(dataset_dir, subset, 'labels')
        output_images_dir = os.path.join(output_dir, subset, 'images')
        output_labels_dir = os.path.join(output_dir, subset, 'labels')

        process_yolo_subset(images_dir, labels_dir, output_images_dir, output_labels_dir, strength=0.65)

    input_yaml = os.path.join(dataset_dir, 'data.yaml')
    output_yaml = os.path.join(output_dir, 'data.yaml')
    update_yaml(input_yaml, output_yaml, output_dir)
