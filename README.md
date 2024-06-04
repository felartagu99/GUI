# GUI

# Aplicación de Cámara con Etiquetado de Imágenes

## Descripción
Esta aplicación de cámara permite capturar imágenes desde una cámara conectada o importarlas desde un directorio y etiquetarlas utilizando modelos de visión por computadora como GroundingDINO y SAM. Las imágenes etiquetadas pueden exportarse en varios formatos, incluidos Pascal-VOC, COCO y YOLO. Por otro lado, permite hacer fine-tunning de diferentes modelos con los datasets generados por la aplicación.

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

## Manuel de uso
1. Ejecutar la aplicación:
    ```sh
    python main.py
    ```
2. En la pantalla de bienvenda:
     * Seleccionar Directorio: En este botón se establece el directorio donde se guardan los etiquetados.
     * Añadir etiquetas: Aquí se añaden los labels de los objetos que se van a capturar.
     * Generación Dataset: En caso de disponer de un conjuunto de imágenes ya etiquedas puedes dirigirte directamente a la interfaz de generación y pruebas.
     * Comenzar: Botón que nos lleva a la interfaz de etiquetado.
     * Salir.

3. En la interfaz de captura:      
     * Seleccionar la cámara desde el combo box.
     * Seleccionar el modelo de etiquetado (GroundingDINO o GroundingDINO + SAM).
     * Determinar los objetos que vamos a capturar mediante el boton "Agregar Etiquetas", esto solo en caso de necesitar más etiquetas aparte de las ya añadidas en la welcome interface.
     * Presionar el botón "Captura" para tomar una foto y etiquetarla.
     * Presionar el botón "Cargar Imágenes" para cargar imágenes desde el sistema de archivos y etiquetarlas.
     * Seleccionar el formato de exportación (Pascal-VOC, COCO, YOLO, COCO with Segmentation) y exportar las imágenes etiquetadas.

4. En la interfaz de generación de dataset y prueba:
     * Seleccionar la carpeta.
     * Aplicar los porcentajes de división para train, test, valid.
     * Elegir el formato.

## Funcionalidades
- **Captura de imágenes:** Permite capturar imágenes desde una cámara conectada.
- **Cargar imágenes:** Permite cargar imágenes desde el sistema de archivos.
- **Etiquetado de imágenes:** Utiliza modelos de visión por computadora para etiquetar objetos en las imágenes.
- **Exportación de anotaciones:** Exporta las anotaciones en formatos Pascal-VOC, COCO, YOLO y COCO with Segmentation.

## Estructura del Código
- **welcome_interface.py:** Pantalla de bienvenida.
- **main.py:** Archivo principal que contiene la lógica de la aplicación y la interfaz gráfica.
- **generation_interface.py:** Interfaz de generación de dataset y fine-tunning de modelos.
- **funciones auxiliares:** Funciones para la creación de directorios, generación de nombres de archivos, exportación de anotaciones y visualización de resultados.

## Créditos
Desarrollado por [Manuel Zamora Pérez y Felipe Artengo Aguado]

"""
