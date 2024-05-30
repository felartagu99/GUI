import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QApplication)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt



class TrainingInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interfaz de Entrenamiento")
        self.resize(400, 300)

        layout = QVBoxLayout()

        # Botón para cargar dataset
        self.load_dataset_button = self.create_button("Cargar Dataset", "#3498db", "#2980b9", self.load_dataset)
        layout.addWidget(self.load_dataset_button, alignment=Qt.AlignCenter)

        # Botón para introducir modelo
        self.load_model_button = self.create_button("Introducir Modelo", "#e67e22", "#d35400", self.load_model)
        layout.addWidget(self.load_model_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def create_button(self, text, color, hover_color, callback):
        button = QPushButton(text, self)
        button.setFont(QFont("Arial", 14))
        button.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; padding: 10px; }}"
                             f"QPushButton:hover {{ background-color: {hover_color}; }}")
        button.clicked.connect(callback)
        return button

    def load_dataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_() == QFileDialog.Accepted:
            dataset_dir = file_dialog.selectedFiles()[0]
            print(f"Dataset cargado: {dataset_dir}")
            # Aquí puedes agregar más lógica para manejar el dataset cargado

    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        model_file, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo", "", "Model Files (*.h5 *.hdf5 *.pt *.pth);;All Files (*)", options=options)
        if model_file:
            print(f"Modelo cargado: {model_file}")
            # Aquí puedes agregar más lógica para manejar el modelo cargado


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingInterface()
    window.show()
    sys.exit(app.exec_())

