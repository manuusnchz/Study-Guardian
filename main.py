import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import math

# --- 1. DESCARGA DE MODELOS (Cara, Manos y Objetos) ---
def descargar_modelo(url, filename):
    if not os.path.exists(filename):
        print(f"⏳ Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("✅ Descargado.")
        except Exception as e:
            print(f"❌ Error: {e}")

# Modelos necesarios
descargar_modelo("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 'face_landmarker.task')
descargar_modelo("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 'hand_landmarker.task')
descargar_modelo("https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite", 'efficientdet.tflite')

# --- 2. CONFIGURACIÓN ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Detector de Cara (Sueño)
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, num_faces=1)

# Detector de Manos (Agua)
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, num_hands=2)

# Detector de Objetos (Móvil)
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
obj_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='efficientdet.tflite'),
    running_mode=VisionRunningMode.VIDEO,
    score_threshold=0.5, # Confianza mínima del 50%
    category_allowlist=['cell phone']) # Solo nos interesa el móvil

# --- PARÁMETROS DE COMPORTAMIENTO ---
# Sueño
UMBRAL_OJOS_CERRADOS = 0.02 # Distancia vertical párpados (Ajustar si es necesario)
TIEMPO_PARA_DORMIRSE = 2.0  # Segundos con ojos cerrados para pitar

# Agua
UMBRAL_BEBER = 250
DURACION_MINIMA_TRAGO = 1.5

# Variables de Estado
contador_agua = 0
estado_bebiendo = False
inicio_gesto_agua = None

inicio_ojos_cerrados = None
alerta_sueño = False

inicio_distraccion = None
tiempo_distraccion_total = 0.0

cap = cv2.VideoCapture(0) # 0 o 1 según tu cámara

print(">>> STUDY GUARDIAN ACTIVADO 📚")
print(">>> Q: Salir | R: Resetear contadores")

# Iniciamos los 3 detectores
with FaceLandmarker.create_from_options(face_options) as face_detector, \
     HandLandmarker.create_from_options(hand_options) as hand_detector, \
     ObjectDetector.create_from_options(obj_options) as obj_detector:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        # Espejo y Color
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = int(time.time() * 1000)
        h, w, _ = image.shape

        # --- EJECUTAR IAs ---
        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        obj_result = obj_detector.detect_for_video(mp_image, timestamp_ms)

        # Variables visuales
        color_estado = (0, 255, 0) # Verde (Bien)
        mensaje_central = "ESTUDIANDO..."
        
        # ---------------------------------------------------------
        # 1. DETECTOR DE MÓVIL (DISTRACCIÓN)
        # ---------------------------------------------------------
        hay_movil = False
        if obj_result.detections:
            for detection in obj_result.detections:
                # Obtenemos la caja del objeto
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                
                # Dibujar recuadro rojo alrededor del móvil
                cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)
                cv2.putText(image, "MOVIL DETECTADO", (start_point[0], start_point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                hay_movil = True

        if hay_movil:
            if inicio_distraccion is None: inicio_distraccion = time.time()
            mensaje_central = "¡¡ SUELTA EL MÓVIL !!"
            color_estado = (0, 0, 255)
        else:
            if inicio_distraccion is not None:
                # Sumamos el tiempo que has estado distraído
                tiempo_distraccion_total += (time.time() - inicio_distraccion)
                inicio_distraccion = None

        # ---------------------------------------------------------
        # 2. DETECTOR DE SUEÑO (OJOS CERRADOS)
        # ---------------------------------------------------------
        boca_x, boca_y = 0, 0 # Para el agua luego
        ojos_cerrados = False
        
        if face_result.face_landmarks:
            face = face_result.face_landmarks[0]
            
            # Párpado superior (159) e inferior (145) del ojo izquierdo
            ojo_izq_arriba = face[159]
            ojo_izq_abajo = face[145]
            # Ojo derecho (386, 374)
            ojo_der_arriba = face[386]
            ojo_der_abajo = face[374]

            # Calcular distancia vertical
            dist_izq = abs(ojo_izq_arriba.y - ojo_izq_abajo.y)
            dist_der = abs(ojo_der_arriba.y - ojo_der_abajo.y)
            promedio_apertura = (dist_izq + dist_der) / 2

            # Guardar boca para el agua
            boca = face[13]
            boca_x, boca_y = int(boca.x * w), int(boca.y * h)

            # LÓGICA DE SUEÑO
            if promedio_apertura < UMBRAL_OJOS_CERRADOS:
                ojos_cerrados = True
                if inicio_ojos_cerrados is None:
                    inicio_ojos_cerrados = time.time()
                
                tiempo_cerrado = time.time() - inicio_ojos_cerrados
                
                # Barra de progreso de sueño
                pct = min(int((tiempo_cerrado/TIEMPO_PARA_DORMIRSE)*100), 100)
                mensaje_central = f"DURMIENDO... {pct}%"
                
                if tiempo_cerrado > TIEMPO_PARA_DORMIRSE:
                    alerta_sueño = True
                    mensaje_central = "¡¡ DESPIERTA !!"
                    color_estado = (0, 0, 255)
                    # SONIDO EN UBUNTU (Beep del sistema)
                    print("\a") 
                    # Opcional: Descomenta esto si tienes 'spd-say' instalado
                    # os.system('spd-say "Despierta" &') 
            else:
                inicio_ojos_cerrados = None
                alerta_sueño = False
            
            # Dibujar ojos
            color_ojos = (0, 0, 255) if ojos_cerrados else (0, 255, 0)
            puntos_ojo = [159, 145, 386, 374]
            for p in puntos_ojo:
                px, py = int(face[p].x * w), int(face[p].y * h)
                cv2.circle(image, (px, py), 2, color_ojos, -1)

        # ---------------------------------------------------------
        # 3. DETECTOR DE AGUA
        # ---------------------------------------------------------
        mano_x, mano_y = 0, 0
        hay_mano = False
        mensaje_agua = ""
        
        if hand_result.hand_landmarks:
            hay_mano = True
            muneca = hand_result.hand_landmarks[0][0]
            mano_x, mano_y = int(muneca.x * w), int(muneca.y * h)
            cv2.circle(image, (mano_x, mano_y), 6, (255, 100, 0), -1)

        if boca_x > 0 and hay_mano:
            dist = math.sqrt((boca_x - mano_x)**2 + (boca_y - mano_y)**2)
            if dist < UMBRAL_BEBER:
                if inicio_gesto_agua is None: inicio_gesto_agua = time.time()
                t_trans = time.time() - inicio_gesto_agua
                if t_trans >= DURACION_MINIMA_TRAGO:
                    mensaje_agua = "+1 TRAGO"
                    if not estado_bebiendo:
                        contador_agua += 1
                        estado_bebiendo = True
                else:
                    mensaje_agua = "Validando..."
            else:
                inicio_gesto_agua = None
                estado_bebiendo = False
            
            # Línea visual
            if dist < UMBRAL_BEBER:
                cv2.line(image, (boca_x, boca_y), (mano_x, mano_y), (0, 255, 255), 2)

        # ---------------------------------------------------------
        # HUD (PANEL DE CONTROL)
        # ---------------------------------------------------------
        # Fondo semitransparente arriba
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        # Estado Principal (Grande)
        cv2.putText(image, mensaje_central, (w//2 - 150, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_estado, 3)

        # Stats (Abajo a la izquierda)
        # Fondo negro pequeño
        cv2.rectangle(image, (10, h-90), (250, h-10), (0,0,0), -1)
        
        # Agua
        cv2.putText(image, f"Agua: {contador_agua} {mensaje_agua}", (20, h-60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Tiempo Distraído
        m = int(tiempo_distraccion_total // 60)
        s = int(tiempo_distraccion_total % 60)
        cv2.putText(image, f"Distraccion: {m:02d}:{s:02d}", (20, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        cv2.imshow('STUDY GUARDIAN v1', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): 
            contador_agua = 0
            tiempo_distraccion_total = 0

cap.release()
cv2.destroyAllWindows()