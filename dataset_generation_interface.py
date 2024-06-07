from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QMessageBox, QDialog, QDialogButtonBox,
                            QInputDialog, QLineEdit, QGroupBox, QSlider, QScrollArea)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import os
import shutil
import random
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import torchvision
from ultralytics import YOLO
from PIL import Image
from torchvision import (transforms, models, datasets)
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection




class VOCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    xml_path = os.path.join(root, file.rsplit('.', 1)[0] + '.xml')
                    if os.path.exists(xml_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.parse_xml(xml_path))
                    else:
                        print(f"Warning: XML file not found for image {img_path}")

        if not self.image_paths:
            raise RuntimeError(f"No valid images found in directory {root_dir}.")

        self.class_names = list(set(self.labels))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # Convert labels to indices
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            label = root.find('object').find('name').text
            return label
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class DatasetGenerationInterface(QWidget):
    def __init__(self, parent=None, welcome_interface=None):
        super().__init__(parent)
        self.setWindowTitle("Generación de Dataset")
        self.resize(800, 600)

        self.dataset_dir = None
        self.welcome_interface = welcome_interface

        main_layout = QVBoxLayout()

        # Botón para seleccionar el dataset, ubicado en la parte superior
        self.select_dataset_button = self.create_button("Seleccionar Carpeta", "#000000", "#333333", "#333333", "#555555", self.select_dataset)
        main_layout.addWidget(self.select_dataset_button, alignment=Qt.AlignCenter)

        # Sección de Selección de Dataset
        dataset_group = QGroupBox("Seleccionar Porcentajes")
        dataset_layout = QVBoxLayout()

        # Sliders para seleccionar los porcentajes de división
        self.train_slider = self.create_slider()
        self.valid_slider = self.create_slider()
        self.test_slider = self.create_slider()

        # Etiquetas para mostrar los valores de los sliders
        self.train_label = QLabel("Train: 70%", self)
        self.valid_label = QLabel("Valid: 15%", self)
        self.test_label = QLabel("Test: 15%", self)

        # Añadir sliders y etiquetas al layout
        percentage_layout = QVBoxLayout()
        
        train_layout = QHBoxLayout()
        train_layout.addWidget(self.train_label)
        train_layout.addWidget(self.train_slider)
        
        valid_layout = QHBoxLayout()
        valid_layout.addWidget(self.valid_label)
        valid_layout.addWidget(self.valid_slider)
        
        test_layout = QHBoxLayout()
        test_layout.addWidget(self.test_label)
        test_layout.addWidget(self.test_slider)
        
        percentage_layout.addLayout(train_layout)
        percentage_layout.addLayout(valid_layout)
        percentage_layout.addLayout(test_layout)
        
        dataset_layout.addLayout(percentage_layout)

        self.split_dataset_button = self.create_button("Dividir Dataset", "#000000", "#333333", "#333333", "#555555", self.show_format_selection_popup)
        self.split_dataset_button.setEnabled(True)
        dataset_layout.addWidget(self.split_dataset_button, alignment=Qt.AlignCenter)

        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)

        # Sección de Selección de Modelo de Prueba
        model_group = QGroupBox("Seleccionar Modelo de Prueba")
        model_layout = QVBoxLayout()

        self.model_selection_combo = QComboBox(self)
        self.model_selection_combo.addItems(["YoloV8", "AlexNet", "Importa tu propio modelo"])  # Añade aquí los modelos que necesites
        self.model_selection_combo.currentIndexChanged.connect(self.on_model_selection)
        model_layout.addWidget(self.model_selection_combo, alignment=Qt.AlignCenter)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Botón para validar y entrenar
        self.validate_button = self.create_button("Realizar Entrenamiento", "#000000", "#333333", "#333333", "#555555", self.validate_and_train)
        main_layout.addWidget(self.validate_button, alignment=Qt.AlignCenter)

        # Botón para volver al menú principal
        self.back_button = self.create_button("Volver", "#e74c3c", "#c0392b", "#d35400", "#e74c3c", self.back_to_main_menu, small=True)
        main_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        # Etiqueta para mostrar el estado
        self.status_label = QLabel("", self)
        main_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        # Área para mostrar los resultados de la validación
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

        # Conectar sliders para actualizar los valores
        self.train_slider.valueChanged.connect(self.update_sliders)
        self.valid_slider.valueChanged.connect(self.update_sliders)
        self.test_slider.valueChanged.connect(self.update_sliders)

        # Inicializar los valores de los sliders
        self.train_slider.setValue(70)
        self.valid_slider.setValue(15)
        self.test_slider.setValue(15)
        self.update_sliders()

        


    def create_button(self, text, color_start, color_end, hover_color_start, hover_color_end, callback, small=False):
        button = QPushButton(text, self)
        font_size = 12 if small else 14
        button.setFont(QFont("Arial", font_size))
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color_start}, stop:1 {color_end});
                color: white;
                border: none;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {hover_color_start}, stop:1 {hover_color_end});
            }}
        """)
        button.clicked.connect(callback)
        return button


    def create_percentage_combo(self):
        combo = QComboBox(self)
        for i in range(101):
            combo.addItem(f"{i}%")
        return combo

    def split_dataset(self, func, popup):
        popup.accept()
        func()

    def create_slider(self):
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, 100)
        slider.setSingleStep(1)
        slider.setTickInterval(10)
        slider.setTickPosition(QSlider.TicksBelow)
        return slider

    def update_sliders(self):
        total = self.train_slider.value() + self.valid_slider.value() + self.test_slider.value()

        # Adjust the sliders to make sure they sum up to 100%
        if total != 100:
            if self.sender() == self.train_slider:
                difference = total - 100
                self.adjust_slider(self.valid_slider, self.test_slider, difference)
            elif self.sender() == self.valid_slider:
                difference = total - 100
                self.adjust_slider(self.train_slider, self.test_slider, difference)
            elif self.sender() == self.test_slider:
                difference = total - 100
                self.adjust_slider(self.train_slider, self.valid_slider, difference)

        self.train_label.setText(f"Train: {self.train_slider.value()}%")
        self.valid_label.setText(f"Valid: {self.valid_slider.value()}%")
        self.test_label.setText(f"Test: {self.test_slider.value()}%")
        
        
    def adjust_slider(self, slider1, slider2, difference):
        if slider1.value() - difference >= 0:
            slider1.setValue(slider1.value() - difference)
        elif slider2.value() - difference >= 0:
            slider2.setValue(slider2.value() - difference)
        else:
            remainder = difference - slider1.value()
            slider1.setValue(0)
            slider2.setValue(slider2.value() - remainder)
            
    def select_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_() == QFileDialog.Accepted:
            self.dataset_dir = file_dialog.selectedFiles()[0]
            self.status_label.setText(f"Dataset seleccionado: {self.dataset_dir}")
            self.split_dataset_button.setEnabled(True)           

    def show_format_selection_popup(self):
        popup = QDialog(self)
        popup.setWindowTitle("Seleccionar Formato")
        layout = QVBoxLayout()

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(popup.accept)
        button_box.rejected.connect(popup.reject)

        layout.addWidget(QLabel("Elige uno de estos 4 formatos:"))
        formats = [("Yolo", self.split_dataset_yolo), 
                ("Coco", self.split_dataset_coco), 
                ("Pascal-Voc", self.split_dataset_pascal_voc), 
                ("Segmentación", self.split_dataset_segmentation)]

        for format_name, format_func in formats:
            button = QPushButton(format_name)
            button.clicked.connect(lambda _, func=format_func: self.split_dataset(func, popup))
            layout.addWidget(button)

        layout.addWidget(button_box)
        popup.setLayout(layout)
        popup.exec_()
  
    def split_dataset_pascal_voc(self):
        if self.dataset_dir:
            # Obtener los porcentajes seleccionados
            train_percentage = int(self.train_slider.value()) / 100
            valid_percentage = int(self.valid_slider.value()) / 100
            test_percentage = int(self.test_slider.value()) / 100

            if train_percentage + valid_percentage + test_percentage != 1.0:
                self.status_label.setText("Los porcentajes deben sumar 100%.")
                return

            # Directorios para guardar las divisiones del dataset
            split_dir = os.path.join(self.dataset_dir, 'split_dataset')
            train_dir = os.path.join(split_dir, 'train')
            valid_dir = os.path.join(split_dir, 'valid')
            test_dir = os.path.join(split_dir, 'test')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Directorios de imágenes y anotaciones
            images_dir = os.path.join(self.dataset_dir, 'images')
            annotations_dir = os.path.join(self.dataset_dir, 'annotations')

            images = []
            annotations = {}

            # Recopilar archivos de imágenes
            for root, _, filenames in os.walk(images_dir):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        base_name = os.path.splitext(file)[0]
                        images.append(os.path.join(root, file))
                        annotations[base_name] = []

            # Recopilar archivos de anotaciones
            for root, _, filenames in os.walk(annotations_dir):
                for file in filenames:
                    if file.lower().endswith(('.txt', '.xml', '.json')):
                        base_name = os.path.splitext(file)[0]
                        if base_name in annotations:
                            annotations[base_name].append(os.path.join(root, file))
                        else:
                            annotations[base_name] = [os.path.join(root, file)]

            # Emparejar las imágenes con sus anotaciones correspondientes
            paired_files = [(img, annotations.get(os.path.splitext(os.path.basename(img))[0], [])) for img in images]

            # Filtrar solo los pares completos (con al menos una anotación)
            paired_files = [pair for pair in paired_files if pair[1]]

            print(f"Total pairs found: {len(paired_files)}")  # Mensaje de depuración

            random.shuffle(paired_files)
            train_pairs = paired_files[:int(train_percentage * len(paired_files))]
            valid_pairs = paired_files[int(train_percentage * len(paired_files)):int((train_percentage + valid_percentage) * len(paired_files))]
            test_pairs = paired_files[int((train_percentage + valid_percentage) * len(paired_files)):]

            print(f"Train pairs: {len(train_pairs)}")  # Mensaje de depuración
            print(f"Valid pairs: {len(valid_pairs)}")  # Mensaje de depuración
            print(f"Test pairs: {len(test_pairs)}")  # Mensaje de depuración

            def copy_files(pairs, dest_dir):
                for img, ann_files in pairs:
                    shutil.copy(img, os.path.join(dest_dir, os.path.basename(img)))
                    print(f"Copied {img} to {dest_dir}")  # Mensaje de depuración
                    for ann in ann_files:
                        shutil.copy(ann, os.path.join(dest_dir, os.path.basename(ann)))
                        print(f"Copied {ann} to {dest_dir}")  # Mensaje de depuración

            copy_files(train_pairs, train_dir)
            copy_files(valid_pairs, valid_dir)
            copy_files(test_pairs, test_dir)

            self.show_popup("Dataset dividido en carpetas 'train', 'valid' y 'test'.")
            self.status_label.setText("Dataset dividido en carpetas 'train', 'valid' y 'test'.")

    def split_dataset_yolo(self):
        if self.dataset_dir:
            # Mostrar el diálogo para introducir las etiquetas
            labels_text, ok = QInputDialog.getText(self, "Introduce las etiquetas", "Etiquetas (separadas por comas):", QLineEdit.Normal)
            
            if not ok or not labels_text:
                self.status_label.setText("Operación cancelada o no se introdujeron etiquetas.")
                return
            
            # Procesar las etiquetas introducidas
            labels = [label.strip() for label in labels_text.split(',')]
            
            if not labels:
                self.status_label.setText("No se introdujeron etiquetas válidas.")
                return

            # Obtener los porcentajes seleccionados
            train_percentage = int(self.train_percentage_combo.currentText().strip('%')) / 100
            valid_percentage = int(self.valid_percentage_combo.currentText().strip('%')) / 100
            test_percentage = int(self.test_percentage_combo.currentText().strip('%')) / 100

            if train_percentage + valid_percentage + test_percentage != 1.0:
                self.status_label.setText("Los porcentajes deben sumar 100%.")
                return

            # Directorios para guardar las divisiones del dataset
            split_dir = os.path.join(self.dataset_dir, 'split_dataset')
            train_dir = os.path.join(split_dir, 'train')
            valid_dir = os.path.join(split_dir, 'valid')
            test_dir = os.path.join(split_dir, 'test')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Subcarpetas para imágenes y anotaciones
            train_images_dir = os.path.join(train_dir, 'images')
            train_annotations_dir = os.path.join(train_dir, 'annotations')
            valid_images_dir = os.path.join(valid_dir, 'images')
            valid_annotations_dir = os.path.join(valid_dir, 'annotations')
            test_images_dir = os.path.join(test_dir, 'images')
            test_annotations_dir = os.path.join(test_dir, 'annotations')

            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(train_annotations_dir, exist_ok=True)
            os.makedirs(valid_images_dir, exist_ok=True)
            os.makedirs(valid_annotations_dir, exist_ok=True)
            os.makedirs(test_images_dir, exist_ok=True)
            os.makedirs(test_annotations_dir, exist_ok=True)

            # Directorios de imágenes y anotaciones
            images_dir = os.path.join(self.dataset_dir, 'images')
            annotations_dir = os.path.join(self.dataset_dir, 'annotations')

            images = []
            annotations = {}

            # Recopilar archivos de imágenes
            for root, _, filenames in os.walk(images_dir):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        base_name = os.path.splitext(file)[0]
                        images.append(os.path.join(root, file))
                        annotations[base_name] = []

            # Recopilar archivos de anotaciones
            for root, _, filenames in os.walk(annotations_dir):
                for file in filenames:
                    if file.lower().endswith(('.txt', '.xml', '.json')):
                        base_name = os.path.splitext(file)[0]
                        if base_name in annotations:
                            annotations[base_name].append(os.path.join(root, file))
                        else:
                            annotations[base_name] = [os.path.join(root, file)]

            # Emparejar las imágenes con sus anotaciones correspondientes
            paired_files = [(img, annotations.get(os.path.splitext(os.path.basename(img))[0], [])) for img in images]

            # Filtrar solo los pares completos (con al menos una anotación)
            paired_files = [pair for pair in paired_files if pair[1]]

            print(f"Total pairs found: {len(paired_files)}")  # Mensaje de depuración

            random.shuffle(paired_files)
            train_pairs = paired_files[:int(train_percentage * len(paired_files))]
            valid_pairs = paired_files[int(train_percentage * len(paired_files)):int((train_percentage + valid_percentage) * len(paired_files))]
            test_pairs = paired_files[int((train_percentage + valid_percentage) * len(paired_files)):]

            print(f"Train pairs: {len(train_pairs)}")  # Mensaje de depuración
            print(f"Valid pairs: {len(valid_pairs)}")  # Mensaje de depuración
            print(f"Test pairs: {len(test_pairs)}")  # Mensaje de depuración

            def copy_files(pairs, img_dest_dir, ann_dest_dir):
                for img, ann_files in pairs:
                    shutil.copy(img, os.path.join(img_dest_dir, os.path.basename(img)))
                    print(f"Copied {img} to {img_dest_dir}")  # Mensaje de depuración
                    for ann in ann_files:
                        shutil.copy(ann, os.path.join(ann_dest_dir, os.path.basename(ann)))
                        print(f"Copied {ann} to {ann_dest_dir}")  # Mensaje de depuración

            copy_files(train_pairs, train_images_dir, train_annotations_dir)
            copy_files(valid_pairs, valid_images_dir, valid_annotations_dir)
            copy_files(test_pairs, test_images_dir, test_annotations_dir)

            # Crear el archivo data.yaml
            data_yaml = {
                'train': os.path.relpath(train_images_dir, split_dir),
                'val': os.path.relpath(valid_images_dir, split_dir),
                'test': os.path.relpath(test_images_dir, split_dir),
                'nc': len(labels),
                'names': labels
            }

            with open(os.path.join(split_dir, 'data.yaml'), 'w') as yaml_file:
                yaml.dump(data_yaml, yaml_file, default_flow_style=True)

            self.show_popup("Dataset dividido en carpetas 'train', 'valid' y 'test' con subcarpetas 'images' y 'annotations'.")
            self.status_label.setText("Dataset dividido en carpetas 'train', 'valid' y 'test' con subcarpetas 'images' y 'annotations'.")
             
    def split_dataset_coco(self):
         if self.dataset_dir:
            # Obtener los porcentajes seleccionados
            train_percentage = int(self.train_percentage_combo.currentText().strip('%')) / 100
            valid_percentage = int(self.valid_percentage_combo.currentText().strip('%')) / 100
            test_percentage = int(self.test_percentage_combo.currentText().strip('%')) / 100

            if train_percentage + valid_percentage + test_percentage != 1.0:
                self.status_label.setText("Los porcentajes deben sumar 100%.")
                return

            # Directorios para guardar las divisiones del dataset
            split_dir = os.path.join(self.dataset_dir, 'split_dataset')
            train_dir = os.path.join(split_dir, 'train')
            valid_dir = os.path.join(split_dir, 'valid')
            test_dir = os.path.join(split_dir, 'test')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Directorios de imágenes y anotaciones
            images_dir = os.path.join(self.dataset_dir, 'images')
            annotations_dir = os.path.join(self.dataset_dir, 'annotations')

            images = []
            annotations = {}

            # Recopilar archivos de imágenes
            for root, _, filenames in os.walk(images_dir):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        base_name = os.path.splitext(file)[0]
                        images.append(os.path.join(root, file))
                        annotations[base_name] = []

            # Recopilar archivos de anotaciones
            for root, _, filenames in os.walk(annotations_dir):
                for file in filenames:
                    if file.lower().endswith(('.txt', '.xml', '.json')):
                        base_name = os.path.splitext(file)[0]
                        if base_name in annotations:
                            annotations[base_name].append(os.path.join(root, file))
                        else:
                            annotations[base_name] = [os.path.join(root, file)]

            # Emparejar las imágenes con sus anotaciones correspondientes
            paired_files = [(img, annotations.get(os.path.splitext(os.path.basename(img))[0], [])) for img in images]

            # Filtrar solo los pares completos (con al menos una anotación)
            paired_files = [pair for pair in paired_files if pair[1]]

            print(f"Total pairs found: {len(paired_files)}")  # Mensaje de depuración

            random.shuffle(paired_files)
            train_pairs = paired_files[:int(train_percentage * len(paired_files))]
            valid_pairs = paired_files[int(train_percentage * len(paired_files)):int((train_percentage + valid_percentage) * len(paired_files))]
            test_pairs = paired_files[int((train_percentage + valid_percentage) * len(paired_files)):]

            print(f"Train pairs: {len(train_pairs)}")  # Mensaje de depuración
            print(f"Valid pairs: {len(valid_pairs)}")  # Mensaje de depuración
            print(f"Test pairs: {len(test_pairs)}")  # Mensaje de depuración

            def copy_files(pairs, dest_dir):
                for img, ann_files in pairs:
                    shutil.copy(img, os.path.join(dest_dir, os.path.basename(img)))
                    print(f"Copied {img} to {dest_dir}")  # Mensaje de depuración
                    for ann in ann_files:
                        shutil.copy(ann, os.path.join(dest_dir, os.path.basename(ann)))
                        print(f"Copied {ann} to {dest_dir}")  # Mensaje de depuración

            copy_files(train_pairs, train_dir)
            copy_files(valid_pairs, valid_dir)
            copy_files(test_pairs, test_dir)

            self.show_popup("Dataset dividido en carpetas 'train', 'valid' y 'test'.")
            self.status_label.setText("Dataset dividido en carpetas 'train', 'valid' y 'test'.")   

    def split_dataset_segmentation(self):
         if self.dataset_dir:
            # Obtener los porcentajes seleccionados
            train_percentage = int(self.train_percentage_combo.currentText().strip('%')) / 100
            valid_percentage = int(self.valid_percentage_combo.currentText().strip('%')) / 100
            test_percentage = int(self.test_percentage_combo.currentText().strip('%')) / 100

            if train_percentage + valid_percentage + test_percentage != 1.0:
                self.status_label.setText("Los porcentajes deben sumar 100%.")
                return

            # Directorios para guardar las divisiones del dataset
            split_dir = os.path.join(self.dataset_dir, 'split_dataset')
            train_dir = os.path.join(split_dir, 'train')
            valid_dir = os.path.join(split_dir, 'valid')
            test_dir = os.path.join(split_dir, 'test')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Directorios de imágenes y anotaciones
            images_dir = os.path.join(self.dataset_dir, 'images')
            annotations_dir = os.path.join(self.dataset_dir, 'annotations')

            images = []
            annotations = {}

            # Recopilar archivos de imágenes
            for root, _, filenames in os.walk(images_dir):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        base_name = os.path.splitext(file)[0]
                        images.append(os.path.join(root, file))
                        annotations[base_name] = []

            # Recopilar archivos de anotaciones
            for root, _, filenames in os.walk(annotations_dir):
                for file in filenames:
                    if file.lower().endswith(('.txt', '.xml', '.json')):
                        base_name = os.path.splitext(file)[0]
                        if base_name in annotations:
                            annotations[base_name].append(os.path.join(root, file))
                        else:
                            annotations[base_name] = [os.path.join(root, file)]

            # Emparejar las imágenes con sus anotaciones correspondientes
            paired_files = [(img, annotations.get(os.path.splitext(os.path.basename(img))[0], [])) for img in images]

            # Filtrar solo los pares completos (con al menos una anotación)
            paired_files = [pair for pair in paired_files if pair[1]]

            print(f"Total pairs found: {len(paired_files)}")  # Mensaje de depuración

            random.shuffle(paired_files)
            train_pairs = paired_files[:int(train_percentage * len(paired_files))]
            valid_pairs = paired_files[int(train_percentage * len(paired_files)):int((train_percentage + valid_percentage) * len(paired_files))]
            test_pairs = paired_files[int((train_percentage + valid_percentage) * len(paired_files)):]

            print(f"Train pairs: {len(train_pairs)}")  # Mensaje de depuración
            print(f"Valid pairs: {len(valid_pairs)}")  # Mensaje de depuración
            print(f"Test pairs: {len(test_pairs)}")  # Mensaje de depuración

            def copy_files(pairs, dest_dir):
                for img, ann_files in pairs:
                    shutil.copy(img, os.path.join(dest_dir, os.path.basename(img)))
                    print(f"Copied {img} to {dest_dir}")  # Mensaje de depuración
                    for ann in ann_files:
                        shutil.copy(ann, os.path.join(dest_dir, os.path.basename(ann)))
                        print(f"Copied {ann} to {dest_dir}")  # Mensaje de depuración

            copy_files(train_pairs, train_dir)
            copy_files(valid_pairs, valid_dir)
            copy_files(test_pairs, test_dir)

            self.show_popup("Dataset dividido en carpetas 'train', 'valid' y 'test'.")
            self.status_label.setText("Dataset dividido en carpetas 'train', 'valid' y 'test'.")   

    def show_popup(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Información")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def back_to_main_menu(self):
        from welcome_interface import WelcomeInterface
        self.welcome_interface = WelcomeInterface()
        self.welcome_interface.show()
        self.close()

    def on_model_selection(self):
        selected_model = self.model_selection_combo.currentText()
        if selected_model == "YoloV8":
            self.select_dataset_for_yolov8()
    
    def select_dataset_for_yolov8(self):
        dataset_dir = QFileDialog.getExistingDirectory(self, "Seleccionar directorio del dataset")

        if not dataset_dir:
            QMessageBox.warning(self, "Advertencia", "No se seleccionó ningún directorio.")
            return

        self.dataset_dir = dataset_dir
        self.validate_and_train()

    def train(self):
            # Cuadro de diálogo para seleccionar dispositivo
        items = ("CPU", "GPU")
        item, ok = QInputDialog.getItem(self, "Seleccionar Dispositivo", "Elija el dispositivo para entrenamiento:", items, 0, False)
        if ok and item:
            device = 'cpu' if item == "CPU" else 'cuda'

            if not self.dataset_dir:
                QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ningún dataset.")
                return

            data_yaml_path = os.path.join(self.dataset_dir, 'data.yaml')
            if not os.path.exists(data_yaml_path):
                QMessageBox.warning(self, "Advertencia", "No se encontró el archivo data.yaml en el directorio del dataset.")
                return

            try:
                # Entrenar el modelo con el dataset personalizado
                model = YOLO('yolov8n.pt').to(device)  # Cargar el modelo preentrenado
                model.train(data=data_yaml_path, epochs=10, lr0=0.01)  # Ajusta los parámetros según sea necesario
            except Exception as e:
                QMessageBox.warning(self, "Advertencia", f"Error durante el entrenamiento: {str(e)}")
            else:
                QMessageBox.information(self, "Información", "Entrenamiento completado.")
   
    def train_alexnet(self):
        # Preguntar al usuario si desea usar CPU o GPU
        devices = ["CPU", "GPU"]
        device_choice, ok = QInputDialog.getItem(self, "Seleccione el dispositivo", "Seleccione el dispositivo para entrenar:", devices, 0, False)
        if not ok:
            return

        device = torch.device("cuda:0" if device_choice == "GPU" and torch.cuda.is_available() else "cpu")

        # Configuración de las transformaciones para los datos de entrenamiento y validación
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Cargar los datos en formato Pascal VOC
        data_dir = self.dataset_dir
        try:
            image_datasets = {
                'train': VOCDataset(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
                'valid': VOCDataset(os.path.join(data_dir, 'valid'), transform=data_transforms['valid']),
                'test': VOCDataset(os.path.join(data_dir, 'test'), transform=data_transforms['test'])
            }
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error al cargar el dataset: {e}")
            return

        dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                    for x in ['train', 'valid', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        
        class_names = image_datasets['train'].class_names

        # Inicializar el modelo AlexNet preentrenado
        model_ft = models.alexnet(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Usar un optimizador SGD
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Planificar una tasa de aprendizaje que decrezca con el tiempo
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # Listas para almacenar pérdidas y precisiones
        train_losses, valid_losses, test_losses = [], [], []
        train_accuracies, valid_accuracies, test_accuracies = [], [], []

        # Entrenar el modelo
        num_epochs = 15
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Cada época tiene una fase de entrenamiento, validación y prueba
            for phase in ['train', 'valid', 'test']:
                if phase == 'train':
                    model_ft.train()  # Configurar el modelo en modo entrenamiento
                else:
                    model_ft.eval()   # Configurar el modelo en modo evaluación

                running_loss = 0.0
                running_corrects = 0

                # Iterar sobre los datos
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Limpiar los gradientes de las variables optimizadas
                    optimizer_ft.zero_grad()

                    # Hacer la propagación hacia adelante
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Hacer la propagación hacia atrás y optimizar solo en la fase de entrenamiento
                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()

                    # Estadísticas
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Guardar las pérdidas y precisiones
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)
                elif phase == 'valid':
                    valid_losses.append(epoch_loss)
                    valid_accuracies.append(epoch_acc)
                else:
                    test_losses.append(epoch_loss)
                    test_accuracies.append(epoch_acc)

            print()

        # Guardar el modelo
        torch.save(model_ft.state_dict(), 'model_final.pth')

        # Guardar el historial de entrenamiento
        with open('training_history.txt', 'w') as f:
            f.write(f'Train Losses: {train_losses}\n')
            f.write(f'Train Accuracies: {train_accuracies}\n')
            f.write(f'Validation Losses: {valid_losses}\n')
            f.write(f'Validation Accuracies: {valid_accuracies}\n')
            f.write(f'Test Losses: {test_losses}\n')
            f.write(f'Test Accuracies: {test_accuracies}\n')

        # Visualizar las gráficas
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10,5))
        plt.title("Loss Over Epochs")
        plt.plot(train_losses, label="Train Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,5))
        plt.title("Accuracy Over Epochs")
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(valid_accuracies, label="Validation Accuracy")
        plt.plot(test_accuracies, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        QMessageBox.information(self, "Entrenamiento de AlexNet", "El entrenamiento de AlexNet se ha completado.")


    
    def train_model(self):
            # Cuadro de diálogo para seleccionar el archivo del modelo
            model_file, _ = QFileDialog.getOpenFileName(self, "Importa tu propio modelo", "", "Model Files (*.pt *.pth)")
            if not model_file:
                QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ningún modelo.")
                return None

            # Cuadro de diálogo para seleccionar dispositivo
            items = ("CPU", "GPU")
            item, ok = QInputDialog.getItem(self, "Seleccionar Dispositivo", "Elija el dispositivo para entrenamiento:", items, 0, False)
            if not ok or not item:
                return None

            device = 'cpu' if item == "CPU" else 'cuda'

            if not self.dataset_dir:
                QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ningún dataset.")
                return None

            try:
                # Cargar el state_dict del modelo especificado por el usuario
                state_dict = torch.load(model_file, map_location=device)

                # Inicializar una instancia del modelo y cargar el state_dict
                # Detectar automáticamente el tipo de modelo
                if "fasterrcnn" in model_file.lower():
                    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
                else:
                    model = torchvision.models.resnet50(weights=None)  # Cambia esto según sea necesario

                model.load_state_dict(state_dict)
                model.to(device)

                # Configurar el optimizador y la función de pérdida
                if isinstance(model, torchvision.models.detection.FasterRCNN):
                    params = [p for p in model.parameters() if p.requires_grad]
                    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
                    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
                    criterion = None  # No se necesita para Faster R-CNN
                else:
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                # Transformaciones y DataLoader
                transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

                if isinstance(model, torchvision.models.detection.FasterRCNN):
                    # Cargar datos del conjunto de entrenamiento
                    dataset = []
                    for filename in os.listdir(os.path.join(self.dataset_dir, 'train')):
                        if filename.endswith('.jpg'):
                            image_path = os.path.join(self.dataset_dir, 'train', filename)
                            xml_path = image_path.replace('.jpg', '.xml')
                            dataset.append((image_path, xml_path))
                            
                    train_dataset = VOCDataset(dataset, transform=transform)
                    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
                else:
                    train_dataset = datasets.ImageFolder(self.dataset_dir, transform=transform)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

                # Función de entrenamiento
                def train_model_generic(model, device, train_loader, optimizer, criterion, scheduler, epochs=10):
                    model.train()
                    train_losses = []
                    train_accuracies = []

                    for epoch in range(epochs):
                        epoch_loss = 0
                        correct = 0
                        total = 0

                        for batch_idx, (inputs, labels) in enumerate(train_loader):
                            optimizer.zero_grad()
                            if isinstance(model, torchvision.models.detection.FasterRCNN):
                                inputs = list(image.to(device) for image in inputs)
                                targets = [{k: v.to(device) for k, v in t.items()} for t in labels]
                                loss_dict = model(inputs, targets)
                                losses = sum(loss for loss in loss_dict.values())
                                losses.backward()
                                optimizer.step()
                                epoch_loss += losses.item()
                            else:
                                inputs, labels = inputs.to(device), labels.to(device)
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()

                                epoch_loss += loss.item()
                                _, predicted = outputs.max(1)
                                total += labels.size(0)
                                correct += predicted.eq(labels).sum().item()

                            if scheduler:
                                scheduler.step()

                        epoch_loss /= len(train_loader)
                        accuracy = 100. * correct / total if not isinstance(model, torchvision.models.detection.FasterRCNN) else None

                        train_losses.append(epoch_loss)
                        if accuracy is not None:
                            train_accuracies.append(accuracy)

                        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}', end='')
                        if accuracy is not None:
                            print(f', Accuracy: {accuracy:.2f}%')

                    return train_losses, train_accuracies

                # Entrenar el modelo y obtener métricas
                train_losses, train_accuracies = train_model_generic(model, device, train_loader, optimizer, criterion, lr_scheduler if isinstance(model, torchvision.models.detection.FasterRCNN) else None, epochs=10)

                # Guardar el modelo entrenado
                model_save_path, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo Entrenado", "", "Model Files (*.pt *.pth)")
                if model_save_path:
                    torch.save(model.state_dict(), model_save_path)

                # Generar y mostrar las gráficas
                epochs = range(1, 11)

                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.plot(epochs, train_losses, 'b', label='Train Loss')
                plt.title('Loss over Epochs')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()

                if train_accuracies:
                    plt.subplot(1, 2, 2)
                    plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
                    plt.title('Accuracy over Epochs')
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy')
                    plt.legend()

                plt.tight_layout()
                plt.show()

            except Exception as e:
                QMessageBox.warning(self, "Advertencia", f"Error durante el entrenamiento: {str(e)}")
                return None
            else:
                QMessageBox.information(self, "Información", "Entrenamiento completado.")
                return model  # Devuelve el modelo entrenado

    
    
    
    def validate_and_train(self):
        selected_model = self.model_selection_combo.currentText()
        if selected_model == "YoloV8":
            self.train()
        elif selected_model == "AlexNet":
            self.train_alexnet()
        elif selected_model == "Importa tu propio modelo":
            self.train_model()
        # Añadir lógica para otros modelos si es necesario
