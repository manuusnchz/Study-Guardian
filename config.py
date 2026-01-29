# config.py

# --- RUTAS DE MODELOS ---
URL_FACE = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
URL_HAND = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
URL_OBJ = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

FILE_FACE = 'face_landmarker.task'
FILE_HAND = 'hand_landmarker.task'
FILE_OBJ = 'efficientdet.tflite'

# --- PARÁMETROS DE SUEÑO ---
# AJUSTA AQUÍ LA SENSIBILIDAD DE LOS OJOS
# Bajar si detecta sueño mirando al libro (ej: 0.012)
# Subir si no detecta sueño (ej: 0.025)
UMBRAL_OJOS_CERRADOS = 0.005 
TIEMPO_PARA_DORMIRSE = 2.0 
TIEMPO_BUFFER_OJOS = 3.0   # Tiempo de seguridad antes de juzgar

# --- PARÁMETROS DE AGUA ---
UMBRAL_BEBER = 250         # Radio de detección (px)
DURACION_MINIMA_TRAGO = 2.0

# --- PARÁMETROS DE DISTRACCIÓN ---
UMBRAL_IGNORAR_MOVIL = 60  # Píxeles cerca de la boca para ignorar (anti-bocata)
CONF_OBJETOS = 0.4         # Confianza mínima de la IA (0.1 a 1.0)

# --- PARÁMETROS DE ILUMINACIÓN ---
# Rango de 0 (oscuridad total) a 255 (luz pura)
UMBRAL_OSCURIDAD = 100

# --- PARÁMETROS DE AUSENCIA ---
TIEMPO_PARA_PAUSA = 6.0  # Segundos sin verte para pausar el estudio