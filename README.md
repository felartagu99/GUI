# GUI

# Aplicación de Cámara con Etiquetado de Imágenes

## Descripción
Esta aplicación de cámara permite capturar imágenes desde una cámara conectada y etiquetarlas utilizando modelos de visión por computadora como GroundingDINO y SAM. Las imágenes etiquetadas pueden exportarse en varios formatos, incluidos Pascal-VOC, COCO y YOLO.

## Requisitos
- Python 3.7 o superior
- Librerías de Python:
  - os
  - shutil
  - sys
  - PIL (Pillow)
  - cv2 (OpenCV)
  - numpy
  - json
  - supervision
  - torch
  - PyQt5
  - segment_anything
  - xml
- Modelos preentrenados:
  - GroundingDINO
  - SAM

## Instalación
1. Clonar el repositorio:
    ```sh
    git clone https://github.com/felartagu99/GUI.git
    ```
2. Navegar al directorio del proyecto:
    ```sh
    cd GUI
    ```
3. Crear un entorno virtual e instalar los requisitos:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # En Windows usar `venv\\Scripts\\activate`
    pip install -r requirements.txt
    ```
4. Descargar los modelos preentrenados y colocarlos en la carpeta `checkpoints`.

## Uso
1. Ejecutar la aplicación:
    ```sh
    python main.py
    ```
2. Seleccionar la cámara desde el combo box.
3. Seleccionar el modelo de etiquetado (GroundingDINO o GroundingDINO + SAM).
4. Determinar los objetos que vamos a capturar mediante el boton "Agregar Etiquetas".
4. Capturar imágenes y etiquetarlas:
    - Presionar el botón "Captura" para tomar una foto y etiquetarla.
    - Presionar el botón "Cargar Imágenes" para cargar imágenes desde el sistema de archivos y etiquetarlas.
5. Seleccionar el formato de exportación (Pascal-VOC, COCO, YOLO, COCO with Segmentation) y exportar las imágenes etiquetadas.

## Funcionalidades
- **Captura de imágenes:** Permite capturar imágenes desde una cámara conectada.
- **Cargar imágenes:** Permite cargar imágenes desde el sistema de archivos.
- **Etiquetado de imágenes:** Utiliza modelos de visión por computadora para etiquetar objetos en las imágenes.
- **Exportación de anotaciones:** Exporta las anotaciones en formatos Pascal-VOC, COCO, YOLO y COCO with Segmentation.

## Estructura del Código
- **main.py:** Archivo principal que contiene la lógica de la aplicación y la interfaz gráfica.
- **funciones auxiliares:** Funciones para la creación de directorios, generación de nombres de archivos, exportación de anotaciones y visualización de resultados.

## Créditos
Desarrollado por [Manuel Zamora Pérez y Felipe Artengo Aguado]

"""