"""
TP 3 - Clasificación de paisajes naturales con CNN
Grupo 4 - Vision Artificial (1C 2026)

Script para entrenar un modelo de red neuronal convolucional que clasifique
imágenes de paisajes en 6 categorías: buildings, forest, glacier, mountain, sea, street.

Se entrena el modelo con diferentes niveles de optimización para comparar resultados.

Arquitectura: Red neuronal convolucional (CNN) con PyTorch
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Importar PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Configuración
IMG_SIZE = 150  # Tamaño de las imágenes (150x150)
BATCH_SIZE = 32  # Cantidad de imágenes por lote de entrenamiento
EPOCHS = 30  # Máximo de épocas (puede parar antes con early stopping)
NUM_CLASSES = 6  # Cantidad de categorías
LEARNING_RATE = 0.001  # Tasa de aprendizaje inicial

# Verificar si GPU está disponible
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n💻 Dispositivo: {DEVICE}")

# Mapeo de clases
CLASS_NAMES = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

# Ruta del dataset
# Nota: Las carpetas tienen estructura anidada (seg_train/seg_train/, etc)
DATASET_PATH = "Imaganes de Paisajes"
TRAIN_PATH = os.path.join(DATASET_PATH, "seg_train", "seg_train")  # Carpeta anidada
TEST_PATH = os.path.join(DATASET_PATH, "seg_test", "seg_test")    # Carpeta anidada


class ImagenetDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes de paisajes.
    
    Hereda de torch.utils.data.Dataset y proporciona:
    - Carga automática de imágenes desde carpetas
    - Transformaciones (redimensionamiento, normalización, augmentation)
    - Mapeo automático de carpetas a etiquetas numéricas
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: directorio raíz con subdirectorios para cada clase
            transform: transformaciones a aplicar a cada imagen
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Mapear clases a índices
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                
                # Cargar rutas de todas las imágenes en esta clase
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(idx)
    
    def __len__(self):
        """Retorna cantidad total de imágenes"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retorna una imagen y su etiqueta.
        
        Args:
            idx: índice de la imagen
            
        Returns:
            imagen: tensor de imagen (3, 150, 150)
            etiqueta: índice de clase (0-5)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Cargar imagen
        imagen = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            imagen = self.transform(imagen)
        
        return imagen, label


class ModeloBase(nn.Module):
    """
    Modelo CNN BASE sin optimizaciones.
    
    Arquitectura simple:
    - 3 bloques convolucionales (32, 64, 128 filtros)
    - Max pooling después de cada bloque
    - 2 capas densas
    
    Sin Batch Normalization, Dropout ni regularización.
    """
    
    def __init__(self, num_classes=6):
        super(ModeloBase, self).__init__()
        
        # Bloque 1: Conv(32) → MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2: Conv(64) → MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque 3: Conv(128) → MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calcular tamaño de entrada para capa densa
        # Después de 3 poolings: 150 → 75 → 37 → 18
        # Con padding: 150 → 75 → 37 → 18
        self.fc_input_size = 128 * 18 * 18
        
        # Capas densas
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass del modelo.
        
        Args:
            x: tensor de entrada (batch_size, 3, 150, 150)
            
        Returns:
            tensor de logits (batch_size, 6)
        """
        # Bloque 1
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Bloque 2
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Bloque 3
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Aplanar
        x = x.view(-1, self.fc_input_size)
        
        # Capas densas
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Sin activación (se aplica en loss)
        
        return x


class ModeloOptimizado(nn.Module):
    """
    Modelo CNN OPTIMIZADO con técnicas avanzadas.
    
    Mejoras aplicadas:
    - Batch Normalization: normaliza activaciones entre capas
    - Dropout: apaga aleatoriamente neuronas (regularización)
    - Más capas convolucionales para mayor capacidad
    - Inicialización de pesos mejorada
    
    Optimizaciones (explicadas en comentarios):
    1. Batch Normalization: evita problemas de gradientes
    2. Dropout: previene overfitting
    3. Arquitectura más profunda: mayor capacidad expresiva
    4. Inicialización de kaiming: mejora convergencia
    """
    
    def __init__(self, num_classes=6):
        super(ModeloOptimizado, self).__init__()
        
        # Bloque 1: Conv → BatchNorm → ReLU → Dropout → MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normaliza activaciones
        self.drop1 = nn.Dropout(0.3)   # Apaga 30% de neuronas
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2: Conv → BatchNorm → ReLU → Dropout → MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque 3: Conv → BatchNorm → ReLU → Dropout → MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bloque 4: Conv adicional para mayor capacidad
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(0.4)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Tamaño después de 4 poolings: 150 → 75 → 37 → 18 → 9
        self.fc_input_size = 256 * 9 * 9
        
        # Capas densas con normalización
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.5)  # Mayor dropout aquí
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
        # Inicialización de Kaiming para mejor convergencia
        self._init_weights()
    
    def _init_weights(self):
        """
        Inicializa los pesos de las capas usando Kaiming initialization.
        
        Esto mejora la convergencia del modelo durante el entrenamiento.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass del modelo optimizado"""
        # Bloque 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool1(x)
        
        # Bloque 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool2(x)
        
        # Bloque 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool3(x)
        
        # Bloque 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool4(x)
        
        # Aplanar
        x = x.view(-1, self.fc_input_size)
        
        # Capas densas
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
        
        x = self.fc3(x)
        
        return x


def cargar_datos(usar_augmentation=False):
    """
    Carga las imágenes del dataset desde el disco.
    
    Transformaciones aplicadas:
    - Redimensionamiento a 150x150
    - Conversión a tensor
    - Normalización (dividir por 255)
    - Opcionalmente: Data Augmentation
    
    Args:
        usar_augmentation: si aplicar data augmentation a entrenamiento
        
    Returns:
        train_loader: DataLoader para datos de entrenamiento
        test_loader: DataLoader para datos de prueba
    """
    
    print("\n📁 Cargando datos...")
    
    # Transformaciones para PRUEBA (solo normalización)
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para ENTRENAMIENTO
    if usar_augmentation:
        # CON Data Augmentation (Optimización 1)
        transform_train = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(20),           # Rotación ±20°
            transforms.RandomAffine(0, translate=(0.2, 0.2)),  # Desplazamiento
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.RandomHorizontalFlip(p=0.5), # Flip horizontal
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        print("  ✓ Data Augmentation activado")
    else:
        # SIN Data Augmentation
        transform_train = transform_test
    
    # Cargar datasets
    train_dataset = ImagenetDataset(TRAIN_PATH, transform=transform_train)
    test_dataset = ImagenetDataset(TEST_PATH, transform=transform_test)
    
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    print(f"✓ Datos cargados correctamente")
    print(f"  - Imágenes de entrenamiento: {len(train_dataset)}")
    print(f"  - Imágenes de prueba: {len(test_dataset)}")
    
    return train_loader, test_loader


def entrenar_epoca(modelo, train_loader, criterio, optimizador, device):
    """
    Entrena el modelo por una época.
    
    Args:
        modelo: modelo a entrenar
        train_loader: DataLoader con datos de entrenamiento
        criterio: función de pérdida (CrossEntropyLoss)
        optimizador: optimizador (Adam)
        device: dispositivo (CPU o GPU)
        
    Returns:
        pérdida promedio de la época
        precisión promedio de la época
    """
    
    modelo.train()
    perdida_total = 0
    correctos = 0
    total = 0
    
    for batch_idx, (imagenes, etiquetas) in enumerate(train_loader):
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)
        
        # Forward pass
        salida = modelo(imagenes)
        perdida = criterio(salida, etiquetas)
        
        # Backward pass
        optimizador.zero_grad()  # Limpiar gradientes anteriores
        perdida.backward()       # Calcular gradientes
        optimizador.step()       # Actualizar pesos
        
        # Estadísticas
        perdida_total += perdida.item()
        _, predicciones = salida.max(1)
        correctos += predicciones.eq(etiquetas).sum().item()
        total += etiquetas.size(0)
        
        # Mostrar progreso cada 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)}", end='\r')
    
    perdida_promedio = perdida_total / len(train_loader)
    precisión = correctos / total
    
    return perdida_promedio, precisión


def evaluar_modelo(modelo, test_loader, criterio, device):
    """
    Evalúa el modelo en el conjunto de validación/prueba.
    
    Args:
        modelo: modelo a evaluar
        test_loader: DataLoader con datos de prueba
        criterio: función de pérdida
        device: dispositivo (CPU o GPU)
        
    Returns:
        pérdida promedio en validación
        precisión promedio en validación
    """
    
    modelo.eval()
    perdida_total = 0
    correctos = 0
    total = 0
    
    with torch.no_grad():  # No calcular gradientes para validación
        for imagenes, etiquetas in test_loader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)
            
            salida = modelo(imagenes)
            perdida = criterio(salida, etiquetas)
            
            perdida_total += perdida.item()
            _, predicciones = salida.max(1)
            correctos += predicciones.eq(etiquetas).sum().item()
            total += etiquetas.size(0)
    
    perdida_promedio = perdida_total / len(test_loader)
    precisión = correctos / total
    
    return perdida_promedio, precisión


def entrenar_modelo(modelo, train_loader, test_loader, nombre_modelo="modelo",
                   usar_lr_scheduling=False):
    """
    Entrena un modelo completo durante múltiples épocas.
    
    Args:
        modelo: modelo a entrenar
        train_loader: DataLoader de entrenamiento
        test_loader: DataLoader de validación
        nombre_modelo: nombre para guardar el modelo
        usar_lr_scheduling: si reducir learning rate dinámicamente
        
    Returns:
        history: diccionario con histórico de entrenamiento
    """
    
    # Configuración del entrenamiento
    criterio = nn.CrossEntropyLoss()  # Función de pérdida para clasificación
    optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)
    
    # Scheduler para reducir learning rate (Optimización 3)
    if usar_lr_scheduling:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizador,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    # Variables para early stopping
    mejor_val_perdida = float('inf')
    epochs_sin_mejora = 0
    patience_early_stop = 5
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\n🚀 Entrenando modelo '{nombre_modelo}'...")
    print(f"   LR Scheduling: {'✓' if usar_lr_scheduling else '✗'}")
    
    inicio = time.time()
    
    for epoca in range(EPOCHS):
        # Entrenar
        train_loss, train_acc = entrenar_epoca(modelo, train_loader, criterio,
                                              optimizador, DEVICE)
        
        # Validar
        val_loss, val_acc = evaluar_modelo(modelo, test_loader, criterio, DEVICE)
        
        # Guardar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Mostrar progreso
        print(f"Época {epoca+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling (Optimización 3)
        if usar_lr_scheduling:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < mejor_val_perdida:
            mejor_val_perdida = val_loss
            epochs_sin_mejora = 0
            # Guardar mejor modelo
            torch.save(modelo.state_dict(), f"{nombre_modelo}_mejor.pt")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= patience_early_stop:
                print(f"\n⚠️  Early stopping en época {epoca+1}")
                # Cargar mejor modelo
                modelo.load_state_dict(torch.load(f"{nombre_modelo}_mejor.pt"))
                break
    
    tiempo_entrenamiento = time.time() - inicio
    print(f"✓ Entrenamiento completado en {tiempo_entrenamiento/60:.2f} minutos")
    
    # Guardar modelo final
    torch.save(modelo.state_dict(), f"{nombre_modelo}.pt")
    print(f"✓ Modelo guardado como '{nombre_modelo}.pt'")
    
    return history


def mostrar_metricas(history, test_loader, modelo, nombre_experimento, device):
    """
    Muestra y calcula métricas finales del modelo.
    
    Args:
        history: histórico del entrenamiento
        test_loader: datos de prueba
        modelo: modelo entrenado
        nombre_experimento: nombre del experimento
        device: dispositivo
    """
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS - {nombre_experimento}")
    print(f"{'='*70}")
    
    # Métricas finales
    train_acc = history['train_acc'][-1]
    val_acc = history['val_acc'][-1]
    train_loss = history['train_loss'][-1]
    val_loss = history['val_loss'][-1]
    
    print(f"\n📈 Métricas finales (época {len(history['train_loss'])}):")
    print(f"   Precisión entrenamiento: {train_acc:.4f}")
    print(f"   Precisión validación:    {val_acc:.4f}")
    print(f"   Pérdida entrenamiento:   {train_loss:.4f}")
    print(f"   Pérdida validación:      {val_loss:.4f}")
    print(f"   Diferencia (overfitting):{(train_acc - val_acc):.4f}")
    
    # Evaluar en conjunto de prueba
    test_loss, test_acc = evaluar_modelo(modelo, test_loader, 
                                        nn.CrossEntropyLoss(), device)
    
    print(f"\n🎯 Evaluación en conjunto de prueba:")
    print(f"   Precisión: {test_acc:.4f}")
    print(f"   Pérdida:   {test_loss:.4f}")
    
    return {
        'nombre': nombre_experimento,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'epochs': len(history['train_loss'])
    }


def graficar_comparacion(resultados):
    """Crea gráficas comparativas de resultados"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nombres = [r['nombre'] for r in resultados]
    test_accs = [r['test_acc'] for r in resultados]
    test_losses = [r['test_loss'] for r in resultados]
    
    # Gráfica de precisión
    axes[0].bar(nombres, test_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_ylabel('Precisión en Test')
    axes[0].set_title('Comparación de Precisión')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(test_accs):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Gráfica de pérdida
    axes[1].bar(nombres, test_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_ylabel('Pérdida en Test')
    axes[1].set_title('Comparación de Pérdida')
    for i, v in enumerate(test_losses):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('comparacion_resultados.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráficas guardadas como 'comparacion_resultados.png'")
    plt.show()


def main():
    """
    Función principal: ejecuta los entrenamientos progresivos.
    
    Entrena 3 modelos:
    1. Modelo base SIN optimizaciones
    2. Modelo con data augmentation
    3. Modelo optimizado (augmentation + BatchNorm + Dropout + LR scheduling)
    """
    
    print("\n" + "="*70)
    print("TP 3 - CLASIFICACIÓN DE PAISAJES NATURALES")
    print("Entrenamientos progresivos con optimizaciones")
    print("="*70)
    
    resultados = []
    
    # ===== EXPERIMENTO 1: Modelo BASE =====
    print("\n\n" + "#"*70)
    print("# EXPERIMENTO 1: Modelo BASE (SIN optimizaciones)")
    print("#"*70)
    
    train_loader, test_loader = cargar_datos(usar_augmentation=False)
    modelo_base = ModeloBase(num_classes=NUM_CLASSES).to(DEVICE)
    
    print(f"\n📊 Parámetros del modelo: {sum(p.numel() for p in modelo_base.parameters()):,}")
    
    history_base = entrenar_modelo(modelo_base, train_loader, test_loader,
                                   nombre_modelo="modelo_base",
                                   usar_lr_scheduling=False)
    
    resultado_base = mostrar_metricas(history_base, test_loader, modelo_base,
                                      "Modelo BASE (sin optimizaciones)", DEVICE)
    resultados.append(resultado_base)
    
    # ===== EXPERIMENTO 2: Con Data Augmentation =====
    print("\n\n" + "#"*70)
    print("# EXPERIMENTO 2: Con Data Augmentation")
    print("#"*70)
    print("\n💡 OPTIMIZACIÓN 1: Data Augmentation")
    print("   Aplica transformaciones aleatorias (rotación, zoom, desplazamiento)")
    print("   para aumentar artificialmente el tamaño del dataset.")
    
    train_loader_aug, test_loader_aug = cargar_datos(usar_augmentation=True)
    modelo_aug = ModeloBase(num_classes=NUM_CLASSES).to(DEVICE)
    
    print(f"\n📊 Parámetros del modelo: {sum(p.numel() for p in modelo_aug.parameters()):,}")
    
    history_aug = entrenar_modelo(modelo_aug, train_loader_aug, test_loader_aug,
                                  nombre_modelo="modelo_augmentation",
                                  usar_lr_scheduling=False)
    
    resultado_aug = mostrar_metricas(history_aug, test_loader_aug, modelo_aug,
                                     "Modelo + Data Augmentation", DEVICE)
    resultados.append(resultado_aug)
    
    # ===== EXPERIMENTO 3: Modelo OPTIMIZADO =====
    print("\n\n" + "#"*70)
    print("# EXPERIMENTO 3: Modelo OPTIMIZADO")
    print("# (Augmentation + BatchNorm + Dropout + LR Scheduling)")
    print("#"*70)
    print("\n💡 OPTIMIZACIÓN 2: Batch Normalization")
    print("   Normaliza las activaciones entre capas para mejor estabilidad.")
    print("\n💡 OPTIMIZACIÓN 3: Dropout")
    print("   Apaga aleatoriamente neuronas para evitar overfitting.")
    print("\n💡 OPTIMIZACIÓN 4: Learning Rate Scheduling")
    print("   Reduce la tasa de aprendizaje si la validación no mejora.")
    print("\n💡 OPTIMIZACIÓN 5: Kaiming Initialization")
    print("   Inicialización de pesos mejorada para convergencia más rápida.")
    
    train_loader_opt, test_loader_opt = cargar_datos(usar_augmentation=True)
    modelo_opt = ModeloOptimizado(num_classes=NUM_CLASSES).to(DEVICE)
    
    print(f"\n📊 Parámetros del modelo: {sum(p.numel() for p in modelo_opt.parameters()):,}")
    
    history_opt = entrenar_modelo(modelo_opt, train_loader_opt, test_loader_opt,
                                  nombre_modelo="modelo_optimizado",
                                  usar_lr_scheduling=True)
    
    resultado_opt = mostrar_metricas(history_opt, test_loader_opt, modelo_opt,
                                     "Modelo OPTIMIZADO", DEVICE)
    resultados.append(resultado_opt)
    
    # ===== RESUMEN FINAL =====
    print("\n\n" + "="*70)
    print("📊 RESUMEN COMPARATIVO FINAL")
    print("="*70)
    
    print(f"\n{'Experimento':<40} {'Test Acc':<12} {'Test Loss':<12}")
    print("-"*70)
    
    for r in resultados:
        print(f"{r['nombre']:<40} {r['test_acc']:<12.4f} {r['test_loss']:<12.4f}")
    
    # Calcular mejoras
    mejora_acc = resultados[-1]['test_acc'] - resultados[0]['test_acc']
    mejora_loss = resultados[0]['test_loss'] - resultados[-1]['test_loss']
    
    print("\n" + "-"*70)
    print(f"✨ Mejora en precisión (BASE → OPTIMIZADO): +{mejora_acc:.4f} ({mejora_acc/resultados[0]['test_acc']*100:.2f}%)")
    print(f"✨ Mejora en pérdida (BASE → OPTIMIZADO):  -{mejora_loss:.4f} ({mejora_loss/resultados[0]['test_loss']*100:.2f}%)")
    
    # Guardar resultados en JSON
    with open('resultados_entrenamiento.json', 'w') as f:
        json.dump([{k: float(v) if isinstance(v, (float, np.floating)) else v 
                   for k, v in r.items()} for r in resultados], f, indent=2)
    print(f"\n✓ Resultados guardados en 'resultados_entrenamiento.json'")
    
    # Graficar
    graficar_comparacion(resultados)


if __name__ == "__main__":
    main()
