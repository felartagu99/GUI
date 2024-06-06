import os
import shutil
import sys
from PIL import Image
import cv2
import numpy as np
import json
import supervision as sv
import torch
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QMessageBox, QComboBox, QFileDialog, QInputDialog, QDialog, QProgressBar,
                             QRadioButton, QButtonGroup,QGridLayout, QSlider, QDialogButtonBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from segment_anything import SamPredictor, sam_model_registry
import xml.etree.ElementTree as ET
from xml.dom import minidom

from groundingdino.util.inference import load_model, load_image, predict, annotate
from GroundingDINO.groundingdino.util import box_ops

from welcome_interface import WelcomeInterface
from dataset_generation_interface import DatasetGenerationInterface 

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_next_id(directory, base_name, extension):
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    if not existing_files:
        return 1
    existing_ids = [int(f[len(base_name)+1:-len(extension)]) for f in existing_files if f[len(base_name)+1:-len(extension)].isdigit()]
    return max(existing_ids, default=0) + 1

def export_to_pascal_voc_annotations(image_path, boxes, phrases, output_dir, labels):
    image = cv2.imread(image_path)
    height, width, depth = image.shape

    # Crear el elemento raíz del XML
    annotation = ET.Element('annotation')

    # Sub-elementos básicos
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(output_dir)

    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)

    path = ET.SubElement(annotation, 'path')
    path.text = image_path

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width_el = ET.SubElement(size, 'width')
    width_el.text = str(width)
    height_el = ET.SubElement(size, 'height')
    height_el.text = str(height)
    depth_el = ET.SubElement(size, 'depth')
    depth_el.text = str(depth)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    # Sub-elementos para cada objeto
    for box, phrase in zip(boxes, phrases):
        if phrase in labels:
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

            obj = ET.SubElement(annotation, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = phrase

            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'

            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'

            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(x_min)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(y_min)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(x_max)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(y_max)

    xml_str = ET.tostring(annotation, encoding='utf-8')
    xml_pretty_str = minidom.parseString(xml_str).toprettyxml(indent='    ')

    # Obtener el siguiente ID para el archivo XML
    xml_id = get_next_id(output_dir, "annotated_image", ".xml")
    annotation_file_path = os.path.join(output_dir, f"annotated_image_{xml_id}.xml")

    # Guardar el archivo XML
    with open(annotation_file_path, 'w') as f:
        f.write(xml_pretty_str)
    print(f"Anotaciones Pascal VOC guardadas en: {annotation_file_path}")
    
def export_to_coco(image_path, boxes, phrases, output_dir, labels):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    annotation_dir = os.path.join(output_dir, "annotations")
    create_dir(annotation_dir)

    coco_output = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "COCO dataset generated",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [{
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": 1
        }],
        "annotations": [],
        "categories": [{"id": idx + 1, "name": label, "supercategory": "none"} for idx, label in enumerate(labels)]
    }

    annotation_id = len(os.listdir(annotation_dir)) + 1  # Obtener el ID siguiente
    annotation_file_name = f"annotations_coco_{annotation_id}.json"
    annotation_file_path = os.path.join(annotation_dir, annotation_file_name)

    for box, phrase in zip(boxes, phrases):
        class_id = labels.index(phrase) + 1 if phrase in labels else -1
        if class_id != -1:
            x_center, y_center, box_width, box_height = box
            x_min = int((x_center - box_width / 2.0) * width)
            y_min = int((y_center - box_height / 2.0) * height)
            box_width = int(box_width * width)
            box_height = int(box_height * height)

            bbox = [x_min, y_min, box_width, box_height]

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": 1,
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1

    with open(annotation_file_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"Anotaciones COCO guardadas en: {annotation_file_path}")

def export_to_coco_segmentation(image_path, labels, bboxes, masks, output_dir):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    coco_output = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "COCO dataset with segmentation generated",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [{
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "id": 1
        }],
        "annotations": [],
        "categories": [{"id": idx + 1, "name": label, "supercategory": "none"} for idx, label in enumerate(labels)]
    }

    for idx, (bbox, mask) in enumerate(zip(bboxes, masks)):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        # Convertir las coordenadas del bounding box a valores absolutos
        x1_abs, y1_abs, x2_abs, y2_abs = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

        # Recortar la máscara al bounding box
        mask_cropped = mask[:, y1_abs:y2_abs, x1_abs:x2_abs]

        # Obtener los píxeles de la máscara dentro del bounding box y formatear la lista
        mask_pixels = np.where(mask_cropped[0] > 0.5)
        segmentation = []
        for y, x in zip(mask_pixels[0], mask_pixels[1]):
            segmentation.extend([int(x + x1_abs), int(y + y1_abs)])

        annotation = {
            "id": idx + 1,
            "image_id": 1,
            "category_id": idx + 1,
            "bbox": [x1_abs, y1_abs, x2_abs - x1_abs, y2_abs - y1_abs],
            "area": (x2_abs - x1_abs) * (y2_abs - y1_abs),
            "segmentation": [segmentation],
            "iscrowd": 0
        }
        coco_output["annotations"].append(annotation)

    # Obtener el siguiente ID para el archivo JSON
    json_id = get_next_id(output_dir, "coco_segmentation", ".json")
    output_path = os.path.join(output_dir, f"coco_segmentation_{json_id}.json")

    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO segmentation annotation saved to {output_path}")

def export_to_yolo(image_path, boxes, phrases, output_dir, labels):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    image_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    create_dir(image_dir)
    create_dir(labels_dir)

    # Obtener el ID para los archivos de anotación
    txt_id = get_next_id(labels_dir, "annotated_image", ".txt")
    label_file_name = f"annotated_image_{txt_id}.txt"
    label_path = os.path.join(labels_dir, label_file_name)

    # Renombrar y copiar la imagen
    image_extension = os.path.splitext(image_path)[1]
    new_image_name = f"annotated_image_{txt_id}{image_extension}"
    new_image_path = os.path.join(image_dir, new_image_name)
    shutil.copy(image_path, new_image_path)
    print(f"Copied and renamed image to {new_image_path}")

    '''
    # Crear archivo classes.txt
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Saved classes to {classes_path}")
    '''
    # Guardar etiquetas en formato YOLO
    yolo_annotations = []

    for box, phrase in zip(boxes, phrases):
        class_id = labels.index(phrase) if phrase in labels else -1
        if class_id != -1:
            x_center, y_center, box_width, box_height = box
            x_center = max(0, min(x_center, 1))  # Asegurar que esté en el rango [0, 1]
            y_center = max(0, min(y_center, 1))
            box_width = max(0, min(box_width, 1))
            box_height = max(0, min(box_height, 1))
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_annotations))
    print(f"Saved YOLO annotations to {label_path}")

    # Crear archivo data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: ../train/images\n")
        f.write(f"val: ../valid/images\n")
        f.write(f"test: ../test/images\n\n")
        f.write(f"nc: {len(labels)}\n")
        f.write(f"names: {labels}\n")
    print(f"Saved data.yaml to {yaml_path}")


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def draw_bounding_boxes(image, boxes, phrases):
    for box, phrase in zip(boxes, phrases):
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, phrase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def annotate_with_masks(image, masks, bboxes, phrases, labels):
    annotated_image = image.copy()
    
    for mask, bbox, phrase in zip(masks, bboxes, phrases):
        color = (0, 255, 0)  # Verde para la máscara
        alpha = 0.5  # Transparencia
        
        # Crear una imagen en blanco del mismo tamaño que la original
        mask_color = np.zeros_like(image, dtype=np.uint8)
        
        # Aplicar color solo a las áreas de la máscara
        mask_color[mask == 1] = color  # Asegurarse de que la máscara sea booleana
        annotated_image = cv2.addWeighted(annotated_image, 1.0, mask_color, alpha, 0)
        
        # Dibujar la caja delimitadora
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Añadir la etiqueta
        label = f"{phrase}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_image

class ConfidenceThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar Umbral de Confianza")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)

        self.threshold = QLabel("35%")
        self.slider.valueChanged.connect(self.update_threshold)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Seleccione el umbral de confianza mínimo:"))
        layout.addWidget(self.slider)
        layout.addWidget(self.threshold)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)
        self.setLayout(layout)

    def update_threshold(self, value):
        self.threshold.setText(f"{value}%")

    def get_confidence_threshold(self):
        return self.slider.value() / 100.0

class ConfirmationDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirmar Segmentación")
        self.labels = []
        self.welcome_interface=WelcomeInterface()
        dialog_width = 800
        dialog_height = 600
        self.setFixedSize(dialog_width, dialog_height)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.image = image

        # Convertir la imagen a QImage y luego a QPixmap
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(dialog_width - 20, dialog_height - 80, Qt.KeepAspectRatio)

        self.image_label.setPixmap(scaled_pixmap)

        button_layout = QHBoxLayout()
        accept_button = QPushButton("Aceptar", self)
        reject_button = QPushButton("Rechazar", self)
        
        accept_button.clicked.connect(self.accept)
        reject_button.clicked.connect(self.reject)

        button_layout.addWidget(accept_button)
        button_layout.addWidget(reject_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

class CameraApp(QWidget):
    def __init__(self, capture_dir, labels):
        super().__init__()
        self.setWindowTitle("Aplicación de Cámara")
        self.resize(800, 600)
        
        self.capture_dir = capture_dir
        self.welcome_interface = WelcomeInterface()
        self.labels = labels

        # Barra de progreso
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        # Etiqueta de la cámara
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)

        # ComboBox para seleccionar la cámara
        self.camera_combo = QComboBox(self)
        self.populate_camera_list()
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        # Botones de control con colores específicos
        self.capture_button = self.create_button("Captura", "#3498db", "#2980b9", self.capture_and_label_image)
        self.add_label_button = self.create_button("Agregar Etiquetas", "#9b59b6", "#8e44ad", self.add_labels)
        self.upload_button = self.create_button("Cargar Imágenes", "#e67e22", "#d35400", self.upload_and_label_image)
        self.dataset_button = self.create_button("Generación de Dataset", "#f1c40f", "#f39c12", self.open_dataset_generation_interface)
        self.exit_button = self.create_button("Salir", "#e74c3c", "#c0392b", self.close_application)
        self.training_button = self.create_button("Retroceder a pestaña anterior", "#9C27B0", "#8E24AA", self.open_welcome_interface)

        # Grupo de botones de modelo
        self.model_group = QButtonGroup(self)
        self.dino_button = QRadioButton("GroundingDINO", self)
        self.sam_button = QRadioButton("GroundingDINO + SAM", self)
        self.dino_button.setChecked(True)
        self.model_group.addButton(self.dino_button)
        self.model_group.addButton(self.sam_button)

        # ComboBox para seleccionar el formato de exportación
        self.format_combo = QComboBox(self)
        self.update_format_combo()

        # Conectar señales de cambio de selección
        self.dino_button.toggled.connect(self.update_format_combo)
        self.sam_button.toggled.connect(self.update_format_combo)
        
        # Diseño de la interfaz
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.camera_combo, alignment=Qt.AlignCenter)

        control_layout = QGridLayout()
        control_layout.addWidget(self.capture_button, 0, 0)
        control_layout.addWidget(self.upload_button, 1, 1)
        control_layout.addWidget(self.dataset_button, 1, 0)
        control_layout.addWidget(self.add_label_button, 0, 1)
        control_layout.addWidget(self.training_button, 2, 0, 1, 2)
        control_layout.addWidget(self.exit_button, 3, 0, 1, 2)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Modelo a utilizar:", self))
        model_layout.addWidget(self.dino_button)
        model_layout.addWidget(self.sam_button)

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Formato de Exportación:", self))
        format_layout.addWidget(self.format_combo)

        main_layout.addLayout(model_layout)
        main_layout.addLayout(format_layout)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        # Configuración inicial
        self.capture_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Capturas")
        os.makedirs(self.capture_dir, exist_ok=True)
        self.camera_index = 0
        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_camera)
        self.labels = labels
        self.timer.start(30)

        self.load_models()
        self.change_camera(0)

    def create_button(self, text, color, hover_color, callback):
        button = QPushButton(text, self)
        button.setFont(QFont("Arial", 14))
        button.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; padding: 10px; }}"
                             f"QPushButton:hover {{ background-color: {hover_color}; }}")
        button.clicked.connect(callback)
        return button

    def open_dataset_generation_interface(self):
        self.dataset_generation_interface = DatasetGenerationInterface()
        self.dataset_generation_interface.show()
        self.close()

    def update_format_combo(self):
            self.format_combo.clear()
            if self.sam_button.isChecked():
                self.format_combo.addItem("COCO with Segmentation")
            else:
                self.format_combo.addItem("Pascal-VOC")
                self.format_combo.addItem("COCO")
                self.format_combo.addItem("YOLO")

    def open_welcome_interface(self):
        self.welcome_interface = WelcomeInterface()
        self.welcome_interface.show()
        self.close() 
 
    def load_models(self):
        # Cargar modelo GroundingDino
        self.groundingdino_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./checkpoints/groundingdino_swint_ogc.pth")

        # Cargar modelo SAM
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam_model)
            
    def populate_camera_list(self):
        index = 0
        while True:
            print(f"Trying to open camera with index {index}")
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                print(f"Camera {index} opened successfully")
                self.camera_combo.addItem(f"Cámara {index}")
                camera.release()
                index += 1
            else:
                print(f"Camera with index {index} could not be opened")
                break          
   
    def display_camera(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.display_image(frame)
            else:
                #print("Failed to read frame from camera!")
                self.camera_label.setText("Error: Failed to read frame")
        else:
            print("Camera not available!")
            self.camera_label.setText("Cámara no disponible")

    def upload_and_label_image(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Cargar Imágenes", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_paths:
            self.image_paths = file_paths
            self.current_image_index = 0

            if not self.labels:
                QMessageBox.critical(self, "Error", "No se han proporcionado etiquetas desde la interfaz de bienvenida. Por favor, añade etiquetas.")
                return

            mode_dialog = QMessageBox(self)
            mode_dialog.setWindowTitle("Modo de Etiquetado")
            mode_dialog.setText("¿Cómo quieres que se muestren las imágenes al ser etiquetadas?")
            one_by_one_button = mode_dialog.addButton("Una a Una", QMessageBox.AcceptRole)
            all_at_once_button = mode_dialog.addButton("Realizar todo el etiquetado sin mostrar previsualización", QMessageBox.RejectRole)
            mode_dialog.exec_()

            self.single_label_mode = (mode_dialog.clickedButton() == one_by_one_button)

            if not self.single_label_mode:
                threshold_dialog = ConfidenceThresholdDialog(self)
                if threshold_dialog.exec_() == QDialog.Accepted:
                    threshold = threshold_dialog.get_confidence_threshold()
                    if threshold is not None:
                        self.confidence_threshold = threshold
                    else:
                        QMessageBox.warning(self, "Advertencia", "Umbral inválido. Se usará el valor por defecto (0.5).")

            self.progress_bar.setMaximum(len(self.image_paths))
            self.progress_bar.setValue(0)
            self.process_next_image()
            
    def process_next_image(self):
        if self.current_image_index < len(self.image_paths):
            original_image_path = self.image_paths[self.current_image_index]
            image_id = get_next_id(self.capture_dir, "captura", ".jpg")

            new_image_path = os.path.join(self.capture_dir, f"captura_{image_id}.jpg")
            shutil.copy(original_image_path, new_image_path)

            self.capture_and_label_image_from_path(new_image_path, image_id)

        else:
            QMessageBox.information(self, "Completado", "Todas las imágenes han sido procesadas.")

    def change_camera(self, index):
        print("Selected camera index:", index)
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
        self.camera_index = index
        print("Opening camera with index:", self.camera_index)
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            print("Failed to open camera!")
            QMessageBox.warning(self, "Error", "¡No se pudo abrir la cámara!")
        else:
            self.timer.start(30)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

    def select_directory(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio", "", options=options)
        if dir_path:
            self.capture_dir = dir_path
            QMessageBox.information(self, "Directorio Seleccionado", f"Directorio seleccionado: {self.capture_dir}")

    def add_labels(self):
        input_text, ok_pressed = QInputDialog.getText(self, "Etiquetas", "Introduce etiquetas separadas por comas:")
        if ok_pressed:
            new_labels = [label.strip() for label in input_text.split(",")]
            new_labels = [label for label in new_labels if label]  # Filtrar etiquetas vacías
            if new_labels:
                # Combinar etiquetas nuevas con las existentes
                combined_labels = self.labels + new_labels
                self.labels = list(set(combined_labels))  # Eliminar duplicados
                QMessageBox.information(self, "Etiquetas", f"Etiquetas agregadas: {', '.join(self.labels)}")
            else:
                QMessageBox.critical(self, "Error", "No se han proporcionado etiquetas válidas. Por favor, introduce etiquetas separadas por comas.")
        else:
            QMessageBox.critical(self, "Error", "Se ha cancelado la entrada de etiquetas. Por favor, introduce etiquetas separadas por comas.")

    def close_application(self):
        self.camera.release()
        self.close()

    def get_color(self, idx):
        np.random.seed(idx)
        return tuple(np.random.randint(0, 255, 3).tolist())

    def capture_and_label_image(self):
        ret, frame = self.camera.read()
        if ret:
            # Obtener el siguiente ID para los archivos
            image_id = get_next_id(self.capture_dir, "captura", ".jpg")

            image_path = os.path.join(self.capture_dir, f"captura_{image_id}.jpg")
            cv2.imwrite(image_path, frame)
            
            annotated_frame = frame.copy()  # Crear una copia para la anotación

            # Combinar etiquetas locales y etiquetas de WelcomeInterface
            combined_labels = self.labels + self.welcome_interface.get_labels()
            combined_labels = list(set(combined_labels))  # Eliminar duplicados

            if self.dino_button.isChecked() or self.sam_button.isChecked():
                # Ejecutar GroundingDINO
                TEXT_PROMPT = ", ".join(combined_labels)
                BOX_THRESHOLD = 0.35
                TEXT_THRESHOLD = 0.25
                image_source, image = load_image(image_path)

                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device="cpu"
                )

                boxes = torch.tensor(boxes)  # Convertir boxes a tensor de PyTorch

                if self.sam_button.isChecked():
                    # Cargar la imagen desde la ruta
                    sam_image = cv2.imread(image_path)
                    self.sam_predictor.set_image(sam_image)

                    masks = []
                    bboxes = []
                    
                    H, W, _ = sam_image.shape
                    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])
                    
                    transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])
                    masks, _, _ = self.sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                    # Anotar la imagen con las máscaras y las cajas delimitadoras
                    for mask in masks:
                        annotated_frame = show_mask(mask[0], annotated_frame)

                    # Dibujar bounding boxes de GroundingDINO
                    annotated_frame = draw_bounding_boxes(annotated_frame, boxes_xyxy.int().numpy(), phrases)
                else:
                    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            
            # Guardar la imagen anotada
            annotated_image_path = os.path.join(self.capture_dir, f"annotated_image_{image_id}.jpg")
            cv2.imwrite(annotated_image_path, annotated_frame)

            # Mostrar la imagen anotada en el diálogo de confirmación
            confirmation_dialog = ConfirmationDialog(annotated_frame, self)
            
            if confirmation_dialog.exec_() == QDialog.Accepted:
                labels_path = os.path.join(self.capture_dir, f"labels_{image_id}.json")
                with open(labels_path, "w") as f:
                    json.dump(combined_labels, f)
                
                export_format = self.format_combo.currentText()
                
                if self.sam_button.isChecked():
                    # Crear la estructura de directorios para COCO with Segmentation
                    coco_segmentation_dir = os.path.join(self.capture_dir, "exportation_in_COCO_with_Segmentation")
                    annotations_dir = os.path.join(coco_segmentation_dir, "annotations")
                    images_dir = os.path.join(coco_segmentation_dir, "images")

                    create_dir(coco_segmentation_dir)
                    create_dir(annotations_dir)
                    create_dir(images_dir)

                    # Copiar la imagen original a la carpeta "images"
                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    print(f"Imagen copiada a: {images_dir}")

                    export_to_coco_segmentation(image_path, combined_labels, boxes, masks, annotations_dir)
                    QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato COCO with Segmentation.\nDirectorio: {self.capture_dir}")
                else:
                    if export_format == "Pascal-VOC":
                        # Crear la estructura de directorios para Pascal VOC
                        pascal_voc_dir = os.path.join(self.capture_dir, "exportation_in_pascalVOC")
                        annotations_dir = os.path.join(pascal_voc_dir, "Annotations")
                        images_dir = os.path.join(pascal_voc_dir, "images")

                        create_dir(pascal_voc_dir)
                        create_dir(annotations_dir)
                        create_dir(images_dir)

                        # Copiar la imagen original a la carpeta "images"
                        shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                        print(f"Imagen copiada a: {images_dir}")

                        export_to_pascal_voc_annotations(image_path, boxes, phrases, annotations_dir, combined_labels)
                        QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato Pascal-VOC.\nDirectorio: {self.capture_dir}")

                    elif export_format == "COCO":
                        # Crear la estructura de directorios para COCO
                        coco_dir = os.path.join(self.capture_dir, "exportation_in_COCO")
                        create_dir(coco_dir)
                        export_to_coco(image_path, boxes, phrases, coco_dir, combined_labels)
                        QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato COCO.\nDirectorio: {self.capture_dir}")

                    elif export_format == "YOLO":
                        # Crear la estructura de directorios para YOLO
                        yolo_dir = os.path.join(self.capture_dir, "exportation_in_YOLO")
                        create_dir(yolo_dir)
                        export_to_yolo(image_path, boxes, phrases, yolo_dir, combined_labels)

            else:
                print("Captura y etiquetado cancelados por el usuario.")
                os.remove(image_path)  # Eliminar la imagen original capturada
                os.remove(annotated_image_path)  # Eliminar la imagen anotada
                QMessageBox.information(self, "Cancelado", "Captura y etiquetado cancelados por el usuario.")

            print("BOXES: ", boxes)

        else:
            QMessageBox.critical(self, "Error", "No se pudo capturar la imagen.")

    def capture_and_label_image_from_path(self, image_path, image_id):
        combined_labels = self.labels + self.welcome_interface.get_labels()
        combined_labels = list(set(combined_labels))

        TEXT_PROMPT = ", ".join(combined_labels)
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device="cpu"
        )

        boxes = torch.tensor(boxes)

        if self.sam_button.isChecked():
            sam_image = cv2.imread(image_path)
            self.sam_predictor.set_image(sam_image)

            H, W, _ = sam_image.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            annotated_frame = cv2.imread(image_path)
            for mask in masks:
                annotated_frame = show_mask(mask[0], annotated_frame)

            annotated_frame = draw_bounding_boxes(annotated_frame, boxes_xyxy.int().numpy(), phrases)
        else:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        annotated_image_path = os.path.join(self.capture_dir, f"annotated_image_{image_id}.jpg")
        cv2.imwrite(annotated_image_path, annotated_frame)

        if not self.single_label_mode:
            confidence_threshold = self.confidence_threshold

            filtered_boxes = [box for box, logit in zip(boxes, logits) if logit >= confidence_threshold]
            filtered_phrases = [phrase for phrase, logit in zip(phrases, logits) if logit >= confidence_threshold]
            filtered_logits = [logit for logit in logits if logit >= confidence_threshold]
        else:
            filtered_boxes = boxes
            filtered_phrases = phrases
            filtered_logits = logits

        proceed_with_labeling = True
        if not self.single_label_mode:
            if any(logit < confidence_threshold for logit in logits):
                confirmation_dialog = ConfirmationDialog(annotated_frame, self)
                proceed_with_labeling = (confirmation_dialog.exec_() == QDialog.Accepted)
        else:
            confirmation_dialog = ConfirmationDialog(annotated_frame, self)
            proceed_with_labeling = (confirmation_dialog.exec_() == QDialog.Accepted)


        if proceed_with_labeling:
            labels_path = os.path.join(self.capture_dir, f"labels_{image_id}.json")
            with open(labels_path, "w") as f:
                json.dump(combined_labels, f)

            export_format = self.format_combo.currentText()

            if self.sam_button.isChecked():
                coco_segmentation_dir = os.path.join(self.capture_dir, "exportation_in_COCO_with_Segmentation")
                annotations_dir = os.path.join(coco_segmentation_dir, "annotations")
                images_dir = os.path.join(coco_segmentation_dir, "images")

                create_dir(coco_segmentation_dir)
                create_dir(annotations_dir)
                create_dir(images_dir)

                shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                export_to_coco_segmentation(image_path, combined_labels, boxes, masks, annotations_dir)
                QMessageBox.information(self, "Éxito", f"Imagen capturada y etiquetada con éxito en formato COCO with Segmentation.\nDirectorio: {self.capture_dir}")
            else:
                if export_format == "Pascal-VOC":
                    pascal_voc_dir = os.path.join(self.capture_dir, "exportation_in_pascalVOC")
                    annotations_dir = os.path.join(pascal_voc_dir, "Annotations")
                    images_dir = os.path.join(pascal_voc_dir, "images")

                    create_dir(pascal_voc_dir)
                    create_dir(annotations_dir)
                    create_dir(images_dir)

                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    export_to_pascal_voc_annotations(image_path, filtered_boxes, filtered_phrases, annotations_dir, combined_labels)

                elif export_format == "COCO":
                    coco_dir = os.path.join(self.capture_dir, "exportation_in_COCO")
                    images_dir = os.path.join(coco_dir, "images")
                    annotations_dir = os.path.join(coco_dir, "annotations")

                    create_dir(coco_dir)
                    create_dir(images_dir)
                    create_dir(annotations_dir)

                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    export_to_coco(image_path, filtered_boxes, filtered_phrases, coco_dir, combined_labels)

                elif export_format == "YOLO":
                    yolo_dir = os.path.join(self.capture_dir, "exportation_in_YOLO")
                    images_dir = os.path.join(yolo_dir, "images")
                    labels_dir = os.path.join(yolo_dir, "labels")

                    create_dir(yolo_dir)
                    create_dir(images_dir)
                    create_dir(labels_dir)

                    shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))
                    export_to_yolo(image_path, filtered_boxes, filtered_phrases, yolo_dir, combined_labels)

        self.current_image_index += 1
        self.progress_bar.setValue(self.current_image_index)
        if self.current_image_index < len(self.image_paths):
            self.process_next_image()
        else:
            QMessageBox.information(self, "Completado", "Todas las imágenes han sido procesadas.")

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WelcomeInterface()
    window.show()
    sys.exit(app.exec_())