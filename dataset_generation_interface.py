from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import os
import shutil
import random

class DatasetGenerationInterface(QWidget):
    def __init__(self, parent=None, welcome_interface_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Generación de Dataset")
        self.resize(400, 400)

        self.dataset_dir = None
        self.welcome_interface_callback = welcome_interface_callback

        layout = QVBoxLayout()

        # Botón para seleccionar dataset
        self.select_dataset_button = self.create_button("Seleccionar Dataset", "#3498db", "#2980b9", self.select_dataset)
        layout.addWidget(self.select_dataset_button, alignment=Qt.AlignCenter)

        # Desplegables para seleccionar los porcentajes de división
        self.train_percentage_combo = self.create_percentage_combo()
        self.valid_percentage_combo = self.create_percentage_combo()
        self.test_percentage_combo = self.create_percentage_combo()

        # Establecer valores iniciales
        self.train_percentage_combo.setCurrentIndex(70)
        self.valid_percentage_combo.setCurrentIndex(15)
        self.test_percentage_combo.setCurrentIndex(15)

        # Añadir desplegables al layout
        percentage_layout = QHBoxLayout()
        percentage_layout.addWidget(QLabel("Train:"))
        percentage_layout.addWidget(self.train_percentage_combo)
        percentage_layout.addWidget(QLabel("Valid:"))
        percentage_layout.addWidget(self.valid_percentage_combo)
        percentage_layout.addWidget(QLabel("Test:"))
        percentage_layout.addWidget(self.test_percentage_combo)
        layout.addLayout(percentage_layout)

        # Botón para dividir dataset
        self.split_dataset_button = self.create_button("Dividir Dataset", "#3498db", "#2980b9", self.split_dataset)
        self.split_dataset_button.setEnabled(False)
        layout.addWidget(self.split_dataset_button, alignment=Qt.AlignCenter)

        # Botón para volver al menú principal
        self.back_button = self.create_button("Volver", "#e74c3c", "#c0392b", self.back_to_main_menu, small=True)
        layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        # Etiqueta para mostrar el estado
        self.status_label = QLabel("", self)
        layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

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


    def split_dataset(self):
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

