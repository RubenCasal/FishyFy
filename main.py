import cv2
import numpy as np
import os
from fisheye_transformation import resize_to_square, create_LUT_table, apply_fisheye
from generate_new_bboxes import yolo_to_absolute, generate_bbox_mask, find_fisheye_yolo_bbox, load_yolo_bboxes

def process_yolo_dataset(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir, strength=0.65):
    # Crear directorios de salida si no existen
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Obtener lista de archivos de imágenes
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print("❌ No se encontraron imágenes en el directorio de entrada.")
        return

    # Leer y redimensionar la primera imagen para determinar el tamaño objetivo y calcular la LUT
    first_image_path = os.path.join(input_images_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"❌ Error: no se pudo cargar la imagen {first_image_path}")
        return

    first_image_square = resize_to_square(first_image)
    target_size = first_image_square.shape[:2]  # (alto, ancho)
    map_x, map_y = create_LUT_table(first_image_square, distortion_strength=strength)

    for image_file in image_files:
        image_path = os.path.join(input_images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(input_labels_dir, label_file)

        # Verificar si el archivo de etiquetas existe
        if not os.path.exists(label_path):
            print(f"⚠ El archivo de etiquetas {label_path} no existe. Se omite la imagen {image_file}.")
            continue

        # Leer imagen y etiquetas
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: no se pudo cargar la imagen {image_path}")
            continue

        # Redimensionar la imagen a cuadrado y luego al tamaño objetivo
        image_square = resize_to_square(image)
        image_resized = cv2.resize(image_square, (target_size[1], target_size[0]))
        h, w = image_resized.shape[:2]

        # Convertir la imagen a blanco y negro
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        bboxes_yolo = load_yolo_bboxes(label_path)
        if not bboxes_yolo:
            print(f"❌ No se encontraron etiquetas válidas en el archivo {label_path}")
            continue

        # Convertir a coordenadas absolutas
        bboxes_absolute = yolo_to_absolute(bboxes_yolo, w, h)

        # Generar máscaras
        masks = generate_bbox_mask(image_resized.shape, bboxes_absolute)

        # Calcular centro y radio visibles
        circle_center = (w // 2, h // 2)
        circle_radius = min(circle_center)

        # Distorsionar imagen
        fisheye_image = apply_fisheye(image_gray, map_x, map_y)

        # Calcular nuevas bboxes
        new_bboxes = []
        for i, mask in enumerate(masks):
            cls_id = int(bboxes_absolute[i][0])
            distorted_mask = apply_fisheye(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
            bbox = find_fisheye_yolo_bbox(distorted_mask, w, h, circle_center, circle_radius)
            if bbox:
                new_bboxes.append([cls_id] + bbox)
            else:
                print(f"⚠ BBox {i} descartada tras la transformación.")

        # Guardar imagen transformada sin bounding boxes
        output_image_path = os.path.join(output_images_dir, image_file)
        cv2.imwrite(output_image_path, fisheye_image)

        # Guardar nuevas bboxes en archivo
        output_label_path = os.path.join(output_labels_dir, label_file)
        with open(output_label_path, "w") as f:
            for bbox in new_bboxes:
                f.write(" ".join(f"{val:.6f}" for val in bbox) + "\n")

        print(f"✅ Procesado {image_file}: imagen y etiquetas guardadas.")

# Ejemplo de uso
if __name__ == "__main__":
    input_images_dir = "./top_view_person_dataset/train/images"
    input_labels_dir = "./top_view_person_dataset/train/labels"
    output_images_dir = "./corregida/images"
    output_labels_dir = "./corregida/labels"
    process_yolo_dataset(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir, strength=0.65)