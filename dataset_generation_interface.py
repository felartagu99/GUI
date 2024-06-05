from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QMessageBox, QDialog, QDialogButtonBox,
                            QInputDialog, QLineEdit, QGroupBox, QSlider, QScrollArea)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import os
import shutil
import random
import yaml
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms





class DatasetGenerationInterface(QWidget):
    def __init__(self, parent=None, welcome_interface=None):
        super().__init__(parent)
        self.setWindowTitle("Generación de Dataset")
        self.resize(800, 600)

        self.dataset_dir = None
        self.welcome_interface = welcome_interface

        main_layout = QVBoxLayout()

        # Botón para seleccionar el dataset, ubicado en la parte superior
        self.select_dataset_button = self.create_button("Seleccionar Carpeta", "#3498db", "#2980b9", self.select_dataset)
        main_layout.addWidget(self.select_dataset_button, alignment=Qt.AlignCenter)

        # Sección de Selección de Dataset
        dataset_group = QGroupBox("Seleccionar Dataset")
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

        self.split_dataset_button = self.create_button("Dividir Dataset", "#3498db", "#2980b9", self.show_format_selection_popup)
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
        self.validate_button = self.create_button("Realizar Inferencia", "#27ae60", "#2ecc71", self.validate_and_train)
        main_layout.addWidget(self.validate_button, alignment=Qt.AlignCenter)

        # Botón para volver al menú principal
        self.back_button = self.create_button("Volver", "#e74c3c", "#c0392b", self.back_to_main_menu, small=True)
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
        
    def create_button(self, text, color, hover_color, callback, small=False):
        button = QPushButton(text, self)
        font_size = 10 if small else 14
        button.setFont(QFont("Arial", font_size))
        button.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; padding: 5px; }}"
                             f"QPushButton:hover {{ background-color: {hover_color}; }}")
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
   

    def validate_and_train(self):
        selected_model = self.model_selection_combo.currentText()
        if selected_model == "YoloV8":
            self.train()
        # Añadir lógica para otros modelos si es necesario
