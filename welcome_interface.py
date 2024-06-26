import os
import sys
import webbrowser
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSlider, QMessageBox,
                             QInputDialog, QHBoxLayout)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from dataset_generation_interface import DatasetGenerationInterface

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
class WelcomeInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bienvenido a la Aplicación de Etiquetado")
        self.setFixedSize(500, 400)
        
        # Layout principal
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Título
        self.title_label = QLabel("Generar nuevo Dataset", self)
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Espacio entre título y botones
        self.layout.addStretch(1)

        # Botón para seleccionar directorio
        self.dir_button = QPushButton("Seleccionar Directorio", self)
        self.dir_button.setFont(QFont("Arial", 14))
        self.dir_button.clicked.connect(self.select_directory)
        self.layout.addWidget(self.dir_button)

        # Botón para seleccionar etiquetas
        self.labels_button = QPushButton("Seleccionar Etiquetas", self)
        self.labels_button.setFont(QFont("Arial", 14))
        self.labels_button.clicked.connect(self.add_labels)
        self.layout.addWidget(self.labels_button)

        # Inicializar etiquetas
        self.labels = []

        # Espacio entre slider y botón de comenzar
        self.layout.addStretch(1)

        self.start_button = QPushButton("Comenzar con el etiquetado", self)
        self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.start_button.clicked.connect(self.start_main_window)
        self.layout.addWidget(self.start_button)

        self.dataset_generation_interface_button = QPushButton("Ir a Generar el Dataset", self)
        self.dataset_generation_interface_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.dataset_generation_interface_button.setStyleSheet("background-color: #43b6d5; color: white; padding: 10px; border-radius: 5px;")
        self.dataset_generation_interface_button.clicked.connect(self.start_dataset_generation_interface)
        self.layout.addWidget(self.dataset_generation_interface_button)

        # Botón para abrir el manual de uso
        self.manual_button = QPushButton("Manual de Uso", self)
        self.manual_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.manual_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px; border-radius: 5px;")
        self.manual_button.clicked.connect(self.open_manual)
        self.layout.addWidget(self.manual_button)

        self.close_button = QPushButton("Salir", self)
        self.close_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.close_button.setStyleSheet("background-color: #c0392b; color: white; padding: 10px; border-radius: 5px;")
        self.close_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.close_button)

        self.layout.addStretch(1)

        self.capture_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Capturas")
        self.labels = [] 

    def select_directory(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio", "", options=options)
        if dir_path:
            self.capture_dir = dir_path
            QMessageBox.information(self, "Directorio Seleccionado", f"Directorio seleccionado: {self.capture_dir}")

    def get_labels(self):
        return self.labels
    
    def add_labels(self):
        input_text, ok_pressed = QInputDialog.getText(self, "Etiquetas", "Introduce etiquetas separadas por comas:")
        if ok_pressed:
            labels = [label.strip() for label in input_text.split(",")]
            labels = [label for label in labels if label] 
            if labels:
                self.labels = labels
                QMessageBox.information(self, "Etiquetas", f"Etiquetas agregadas: {', '.join(self.labels)}")
            else:
                QMessageBox.critical(self, "Error", "No se han proporcionado etiquetas válidas. Por favor, introduce etiquetas separadas por comas.")
        else:
            QMessageBox.critical(self, "Error", "Se ha cancelado la entrada de etiquetas. Por favor, introduce etiquetas separadas por comas.")

    def close_application(self):
        self.close()
    
    def start_dataset_generation_interface(self):
        self.dataset_generation_interface = DatasetGenerationInterface()
        self.dataset_generation_interface.show()
        
    def start_main_window(self):
        from main import CameraApp  # Importamos aquí para evitar la importación circular
        if not self.capture_dir:
            QMessageBox.critical(self, "Error", "Por favor, selecciona un directorio antes de continuar.")
            return
        if not self.labels:
            QMessageBox.critical(self, "Error", "Por favor, añade etiquetas antes de continuar.")
            return
        self.hide()
        self.main_window = CameraApp(self.capture_dir, self.labels)
        self.main_window.show()

    def open_manual(self):
        url = "https://github.com/felartagu99/GUI/blob/main/README.md"
        webbrowser.open(url)