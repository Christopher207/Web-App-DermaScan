import cv2
import numpy as np

def process_image(input_path, output_path):
    # Cargar la imagen
    img = cv2.imread(input_path)
    if img is None:
        print("Error: No se pudo cargar la imagen")
        return
    
    # 1. Aclarar la imagen (aumentar brillo)
    brightness_value = 50  # Valor para aclarar la imagen (ajustar según necesidad)
    brightened = cv2.convertScaleAbs(img, alpha=1, beta=brightness_value)
    
    # 2. Dividir en canales BGR
    b, g, r = cv2.split(brightened)
    
    # 3. Aplicar threshold a cada canal para eliminar ruido
    # Valores de threshold (ajustar según necesidad)
    thresh_value = 120
    max_value = 255
    
    _, b_thresh = cv2.threshold(b, thresh_value, max_value, cv2.THRESH_BINARY)
    _, g_thresh = cv2.threshold(g, thresh_value, max_value, cv2.THRESH_BINARY)
    _, r_thresh = cv2.threshold(r, thresh_value, max_value, cv2.THRESH_BINARY)
    
    # Combinar los canales procesados
    merged = cv2.merge((b_thresh, g_thresh, r_thresh))
    
    # 4. Guardar la imagen resultante
    cv2.imwrite(output_path, merged)
    print(f"Imagen procesada guardada como: {output_path}")

# Ejemplo de uso
input_image = "ESAN University.jpg"  # Reemplaza con tu imagen de entrada
output_image = "output_processed.jpg"  # Nombre para la imagen de salida

process_image(input_image, output_image)