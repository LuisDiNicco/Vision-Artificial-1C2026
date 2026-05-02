# TP 2 - Vision Artificial (1C 2026)

## Datos del trabajo practico
- Materia: Vision Artificial
- Institucion: UNLaM
- Cuatrimestre: 1C 2026

## Integrantes (Grupo 4)

| DNI | Apellido, Nombre |
|---:|---|
| 43.630.151 | Antonioli, Iván Oscar |
| 43.664.669 | Di Nicco, Luis Demetrio |
| 41.069.597 | Rojas, Tomas Ian |

## Consigna (resumen)
La consigna completa está en el archivo de la carpeta (`Consigna TP 2 Clasificación con machine learning`).

Resumen: construir un sistema que clasifique formas geométricas (por ejemplo: cuadrado, triángulo, estrella) usando descriptores invariantes (momentos de Hu) y un clasificador de Machine Learning entrenado con muestras capturadas por webcam.

## Implementacion actual (Grupo 4)
La implementación incluida en este directorio permite:

- Generar descriptores invariantes (momentos de Hu) desde la webcam y guardarlos en `dataset.csv`.
- Entrenar un clasificador (árbol de decisión) con `entrenador.py` para producir `modelo.joblib`.
- Clasificar en tiempo real usando `clasificador.py`, que detecta el contorno más grande en la imagen, calcula los momentos de Hu y predice la etiqueta.

## Estructura de archivos

- `generadorDescriptores.py`: interfaz para capturar muestras desde la webcam y guardar Hu moments en `dataset.csv`.
- `dataset.csv`: CSV con columnas `hu1..hu7, etiqueta` (muestras ya generadas).
- `entrenador.py`: carga `dataset.csv`, entrena un `DecisionTreeClassifier` y guarda `modelo.joblib`.
- `clasificador.py`: carga `modelo.joblib` y clasifica formas en tiempo real desde la webcam.
- `modelo.joblib`: modelo entrenado (si existe).

## Como funciona (resumen)
1. Ejecutar `generadorDescriptores.py` y presentar la forma frente a la cámara. Seleccionar la etiqueta (1,2,3...) y presionar `ESPACIO` para guardar la muestra.
2. Una vez recopiladas suficientes muestras, ejecutar `entrenador.py` para generar `modelo.joblib`.
3. Ejecutar `clasificador.py` para clasificar en tiempo real sobre la webcam.

Detalles técnicos:
- Se aplica una binarización y filtrado por área para extraer el contorno principal.
- Se calculan los 7 momentos de Hu como descriptores invariantes.
- El clasificador por defecto es un árbol de decisión entrenado con `scikit-learn`.

## Controles
- En `generadorDescriptores.py`:
	- `1`, `2`, `3` → cambiar etiqueta activa
	- `ESPACIO` → guardar muestra
	- `Q` → salir
- En `clasificador.py`:
	- `Q` → salir

## Instalacion y ejecucion
1. Instalar dependencias:

```bash
pip install opencv-python numpy scikit-learn pandas joblib
```

2. Para generar muestras:

```bash
python "generadorDescriptores.py"
```

3. Para entrenar el modelo (genera `modelo.joblib`):

```bash
python "entrenador.py"
```

4. Para ejecutar el clasificador en tiempo real:

```bash
python "clasificador.py"
```

## Notas
- Ajustar `UMBRAL_BINARIO` en los scripts si la iluminación del ambiente no es adecuada.
- `AREA_MIN` evita que ruido o pequeñas manchas se consideren como objeto.
- El enfoque con momentos de Hu funciona bien para formas simétricas; para problemas más complejos conviene usar descriptores más robustos o modelos CNN.