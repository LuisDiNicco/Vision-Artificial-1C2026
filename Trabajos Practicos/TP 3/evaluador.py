"""
Script para evaluar y usar el modelo entrenado en imágenes de prueba.

Permite cargar modelos entrenados y realizar predicciones sobre imágenes
del conjunto de prueba o imágenes nuevas.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración
IMG_SIZE = 150
NUM_CLASSES = 6

# Mapeo de clases
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_NAMES_ES = ['edificios', 'bosques', 'glaciares', 'montañas', 'mar', 'calles']

DATASET_PATH = "Imaganes de Paisajes"
TEST_PATH = os.path.join(DATASET_PATH, "seg_test")


def cargar_modelo(ruta_modelo):
    """
    Carga un modelo entrenado desde archivo.
    
    Args:
        ruta_modelo: ruta al archivo del modelo (.h5)
        
    Returns:
        modelo: modelo de Keras cargado
    """
    print(f"\n📂 Cargando modelo desde '{ruta_modelo}'...")
    
    if not os.path.exists(ruta_modelo):
        print(f"❌ Error: archivo '{ruta_modelo}' no encontrado")
        return None
    
    modelo = keras.models.load_model(ruta_modelo)
    print(f"✓ Modelo cargado exitosamente")
    print(f"  Parámetros totales: {modelo.count_params():,}")
    
    return modelo


def cargar_imagen(ruta_imagen):
    """
    Carga y prepara una imagen para predicción.
    
    Pasos:
    1. Cargar imagen
    2. Redimensionar a 150x150
    3. Normalizar píxeles (dividir por 255)
    4. Agregar dimensión de batch
    
    Args:
        ruta_imagen: ruta a la imagen
        
    Returns:
        imagen_procesada: numpy array listo para predicción
        imagen_original: PIL Image para mostrar
    """
    
    # Cargar imagen
    img = Image.open(ruta_imagen).convert('RGB')
    imagen_original = img.copy()
    
    # Redimensionar
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convertir a array y normalizar
    img_array = np.array(img) / 255.0
    
    # Agregar dimensión de batch: (150, 150, 3) → (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, imagen_original


def predecir_imagen(modelo, ruta_imagen):
    """
    Realiza una predicción sobre una imagen individual.
    
    Args:
        modelo: modelo de Keras
        ruta_imagen: ruta a la imagen
        
    Returns:
        prediccion: probabilidades para cada clase
        clase_predicha: índice de la clase con mayor probabilidad
    """
    
    img_array, img_original = cargar_imagen(ruta_imagen)
    
    # Realizar predicción
    # La salida es un array de probabilidades para cada clase
    prediccion = modelo.predict(img_array, verbose=0)
    clase_predicha = np.argmax(prediccion[0])
    confianza = prediccion[0][clase_predicha]
    
    return prediccion[0], clase_predicha, confianza, img_original


def mostrar_prediccion(ruta_imagen, prediccion, clase_predicha, confianza):
    """
    Muestra una imagen y su predicción.
    
    Args:
        ruta_imagen: ruta de la imagen
        prediccion: probabilidades para cada clase
        clase_predicha: índice de la clase predicha
        confianza: probabilidad de la clase predicha
    """
    
    img_array, img_original = cargar_imagen(ruta_imagen)
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mostrar imagen
    axes[0].imshow(img_original)
    axes[0].set_title(f"Imagen: {os.path.basename(ruta_imagen)}")
    axes[0].axis('off')
    
    # Gráfica de probabilidades
    colores = ['#2ecc71' if i == clase_predicha else '#95a5a6' for i in range(NUM_CLASSES)]
    axes[1].barh(CLASS_NAMES_ES, prediccion, color=colores)
    axes[1].set_xlabel('Probabilidad')
    axes[1].set_title(f"Predicción: {CLASS_NAMES_ES[clase_predicha]}\n(Confianza: {confianza:.2%})")
    axes[1].set_xlim([0, 1])
    
    # Mostrar valores en barras
    for i, v in enumerate(prediccion):
        axes[1].text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()


def evaluar_en_conjunto_prueba(modelo):
    """
    Evalúa el modelo en todo el conjunto de prueba.
    
    Calcula:
    - Exactitud general
    - Reporte de clasificación (precision, recall, F1)
    - Matriz de confusión
    
    Args:
        modelo: modelo entrenado
    """
    
    print(f"\n📊 Evaluando en conjunto de prueba...")
    
    # Generador para cargar datos de prueba
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_generator = ImageDataGenerator(rescale=1./255)
    
    test_data = test_generator.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Predicciones en todo el conjunto
    print("Realizando predicciones (esto puede tardar)...")
    predicciones = modelo.predict(test_data, verbose=1)
    
    # Clases predichas y reales
    clases_predichas = np.argmax(predicciones, axis=1)
    clases_reales = test_data.classes
    
    # Calcular métricas
    exactitud = np.mean(clases_predichas == clases_reales)
    
    print(f"\n✓ Evaluación completada")
    print(f"\n📈 Exactitud general: {exactitud:.4f} ({exactitud*100:.2f}%)")
    
    # Reporte detallado
    print(f"\n📋 Reporte de clasificación:\n")
    print(classification_report(clases_reales, clases_predichas, target_names=CLASS_NAMES_ES))
    
    # Matriz de confusión
    cm = confusion_matrix(clases_reales, clases_predichas)
    
    # Graficar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES_ES, yticklabels=CLASS_NAMES_ES,
                cbar_kws={'label': 'Cantidad'})
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=150, bbox_inches='tight')
    print("\n✓ Matriz de confusión guardada como 'matriz_confusion.png'")
    plt.show()


def demostrar_predicciones_aleatorias(modelo, cantidad=6):
    """
    Realiza predicciones sobre imágenes aleatorias del conjunto de prueba.
    
    Args:
        modelo: modelo entrenado
        cantidad: cantidad de imágenes a mostrar
    """
    
    print(f"\n🎲 Mostrando {cantidad} predicciones aleatorias...")
    
    # Obtener lista de todas las imágenes
    imagenes = []
    for clase_idx, clase_nombre in enumerate(CLASS_NAMES):
        clase_path = os.path.join(TEST_PATH, clase_nombre)
        if os.path.exists(clase_path):
            archivos = os.listdir(clase_path)
            for archivo in archivos:
                ruta_completa = os.path.join(clase_path, archivo)
                imagenes.append((ruta_completa, clase_nombre))
    
    # Seleccionar aleatoriamente
    indices_aleatorios = np.random.choice(len(imagenes), cantidad, replace=False)
    
    # Crear grid de figuras
    filas = (cantidad + 2) // 3
    fig, axes = plt.subplots(filas, 3, figsize=(15, 5*filas))
    axes = axes.flatten() if cantidad > 1 else [axes]
    
    for idx_plot, idx_imagen in enumerate(indices_aleatorios):
        ruta_imagen, clase_real = imagenes[idx_imagen]
        
        # Predicción
        prediccion, clase_predicha, confianza, img_original = predecir_imagen(modelo, ruta_imagen)
        
        # Mostrar en grid
        ax = axes[idx_plot]
        ax.imshow(img_original)
        
        # Color: verde si correcto, rojo si incorrecto
        clase_correcta = (CLASS_NAMES[clase_predicha] == clase_real)
        color = 'green' if clase_correcta else 'red'
        
        titulo = f"Real: {clase_real}\n"
        titulo += f"Pred: {CLASS_NAMES[clase_predicha]}\n"
        titulo += f"Conf: {confianza:.2%}"
        
        ax.set_title(titulo, color=color, fontweight='bold')
        ax.axis('off')
    
    # Ocultar axes sobrantes
    for idx in range(idx_plot + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('predicciones_aleatorias.png', dpi=150, bbox_inches='tight')
    print("✓ Predicciones guardadas como 'predicciones_aleatorias.png'")
    plt.show()


def main():
    """
    Función principal: menú para evaluar modelos.
    """
    
    print("\n" + "="*70)
    print("EVALUADOR DE MODELOS - Clasificación de Paisajes")
    print("="*70)
    
    # Seleccionar modelo
    print("\n¿Cuál modelo deseas evaluar?")
    print("1. modelo_base.h5 (sin optimizaciones)")
    print("2. modelo_augmentation.h5 (con data augmentation)")
    print("3. modelo_optimizado.h5 (completamente optimizado)")
    print("4. Otro modelo (especificar ruta)")
    
    opcion = input("\nOpción (1-4): ").strip()
    
    modelos_disponibles = {
        '1': 'modelo_base.h5',
        '2': 'modelo_augmentation.h5',
        '3': 'modelo_optimizado.h5'
    }
    
    if opcion in modelos_disponibles:
        ruta_modelo = modelos_disponibles[opcion]
    elif opcion == '4':
        ruta_modelo = input("Ingresa la ruta del modelo: ").strip()
    else:
        print("❌ Opción inválida")
        return
    
    # Cargar modelo
    modelo = cargar_modelo(ruta_modelo)
    if modelo is None:
        return
    
    # Menú de opciones
    while True:
        print("\n" + "-"*70)
        print("¿Qué deseas hacer?")
        print("1. Evaluar en conjunto de prueba (con métricas)")
        print("2. Ver predicciones aleatorias")
        print("3. Predecir imagen específica")
        print("4. Salir")
        
        opcion = input("\nOpción (1-4): ").strip()
        
        if opcion == '1':
            evaluar_en_conjunto_prueba(modelo)
        
        elif opcion == '2':
            cantidad = input("¿Cuántas imágenes? (default: 6): ").strip()
            cantidad = int(cantidad) if cantidad.isdigit() else 6
            demostrar_predicciones_aleatorias(modelo, cantidad)
        
        elif opcion == '3':
            ruta_imagen = input("Ingresa la ruta de la imagen: ").strip()
            if os.path.exists(ruta_imagen):
                prediccion, clase_predicha, confianza, img_original = predecir_imagen(modelo, ruta_imagen)
                mostrar_prediccion(ruta_imagen, prediccion, clase_predicha, confianza)
            else:
                print(f"❌ Archivo no encontrado: {ruta_imagen}")
        
        elif opcion == '4':
            print("👋 Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")


if __name__ == "__main__":
    main()
