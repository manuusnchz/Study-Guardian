import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
import urllib.request
import math


def formatear_tiempo(segundos):
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"

# --- 1. DESCARGA AUTOMÁTICA DE MODELOS ---
def descargar_modelo(url, filename):
    if not os.path.exists(filename):
        print(f"⏳ Descargando {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("✅ Descargado.")

descargar_modelo("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 'face_landmarker.task')
descargar_modelo("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 'hand_landmarker.task')

# --- 2. CONFIGURACIÓN DE LA IA ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# --- CONFIGURACIÓN DE ENTRENAMIENTO ---
UMBRAL_GIRO = 0.04
UMBRAL_DE_PIE = 0.08
UMBRAL_BEBER = 70
DURACION_MINIMA_TRAGO = 2.0  # Segundos que debes mantener la mano para que cuente
TECLAS_ACTIVAS = False 

calibrated_y = None
contador_agua = 0
estado_bebiendo = False 
tiempo_sentado = 0.0
tiempo_de_pie = 0.0
ultimo_tiempo = time.time()


# Variables para el cronómetro del agua
inicio_gesto_agua = None 

cap = cv2.VideoCapture(1) 

print(">>> SISTEMA 'CYCLING COACH v3' (Anti-Falsos Positivos) 🚴💧")
print(">>> Pulsa 'C' para CALIBRAR | 'R' para RESETEAR AGUA | 'Q' para SALIR")

with FaceLandmarker.create_from_options(face_options) as face_detector, \
     HandLandmarker.create_from_options(hand_options) as hand_detector:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = int(time.time() * 1000)

        tiempo_actual = time.time()
        delta_tiempo = tiempo_actual - ultimo_tiempo
        ultimo_tiempo = tiempo_actual

        # Sumar tiempo según la postura detectada en el frame ANTERIOR
        # (o la que se detecte abajo, pero actualizar aquí asegura que siempre cuenta)
        if "DE PIE" in status_posture:
            tiempo_de_pie += delta_tiempo
        elif "SENTADO" in status_posture:
            tiempo_sentado += delta_tiempo

        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        
        h, w, _ = image.shape
        status_turn = "CENTRO"
        status_posture = "SENTADO"
        color_posture = (0, 255, 0)
        color_turn = (0, 255, 0)
        
        boca_x, boca_y = 0, 0
        mano_x, mano_y = 0, 0
        hay_cara = False
        hay_mano = False

        # --- CARA ---
        if face_result.face_landmarks:
            hay_cara = True
            face = face_result.face_landmarks[0]
            nose = face[1]
            nose_x, nose_y = nose.x * w, nose.y * h
            left_ear = face[234]
            right_ear = face[454]
            boca = face[13]
            boca_x, boca_y = boca.x * w, boca.y * h

            dist_nose_left = abs(nose.x - left_ear.x)
            dist_nose_right = abs(nose.x - right_ear.x)

            if nose.x < left_ear.x or dist_nose_left < UMBRAL_GIRO:
                status_turn = "IZQUIERDA <--"
                color_turn = (0, 255, 255)
                if TECLAS_ACTIVAS: pyautogui.press('left')
            elif nose.x > right_ear.x or dist_nose_right < UMBRAL_GIRO:
                status_turn = "DERECHA -->"
                color_turn = (0, 255, 255)
                if TECLAS_ACTIVAS: pyautogui.press('right')

            if calibrated_y is not None:
                if nose.y < (calibrated_y - UMBRAL_DE_PIE):
                    status_posture = "¡¡ DE PIE !! ⚡"
                    color_posture = (0, 0, 255)
                else:
                    status_posture = "SENTADO"
            else:
                status_posture = "NO CALIBRADO"
                color_posture = (200, 200, 200)

            cv2.circle(image, (int(nose_x), int(nose_y)), 5, color_posture, -1)
            cv2.circle(image, (int(boca_x), int(boca_y)), 3, (255, 0, 255), -1)

        # --- MANOS ---
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            hay_mano = True
            muneca = hand[0]
            mano_x, mano_y = muneca.x * w, muneca.y * h
            cv2.circle(image, (int(mano_x), int(mano_y)), 8, (255, 100, 0), -1)

        # --- CÁLCULO INTELIGENTE DE AGUA (CON TEMPORIZADOR) ---
        mensaje_agua = ""
        color_linea = (200, 200, 200)
        
        if hay_cara and hay_mano:
            distancia = math.sqrt((boca_x - mano_x)**2 + (boca_y - mano_y)**2)
            
            if distancia < UMBRAL_BEBER:
                # 1. Si es el primer frame que detecta la mano cerca, iniciamos el cronómetro
                if inicio_gesto_agua is None:
                    inicio_gesto_agua = time.time()
                
                # 2. Calcular cuánto tiempo lleva la mano ahí
                tiempo_transcurrido = time.time() - inicio_gesto_agua
                
                if tiempo_transcurrido >= DURACION_MINIMA_TRAGO:
                    # ¡YA HA PASADO EL SEGUNDO! ES UN TRAGO REAL
                    color_linea = (0, 255, 0) # Verde
                    mensaje_agua = "¡TRAGO REGISTRADO! +1"
                    
                    if not estado_bebiendo:
                        contador_agua += 1
                        estado_bebiendo = True # Bloqueamos para no sumar más en este gesto
                else:
                    # AÚN ESTÁ VALIDANDO (Anti-Rascarse)
                    color_linea = (0, 255, 255) # Amarillo
                    porcentaje = int((tiempo_transcurrido / DURACION_MINIMA_TRAGO) * 100)
                    mensaje_agua = f"Validando... {porcentaje}%"
            
            else:
                # Si aleja la mano, reseteamos todo
                inicio_gesto_agua = None
                estado_bebiendo = False
            
            # Dibujar línea
            cv2.line(image, (int(boca_x), int(boca_y)), (int(mano_x), int(mano_y)), color_linea, 2)

        else:
            # Si pierde de vista la mano o la cara, reseteamos
            inicio_gesto_agua = None

        # --- HUD ---
        cv2.rectangle(image, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.putText(image, f"Giro: {status_turn}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_turn, 2)
        cv2.putText(image, f"Postura: {status_posture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_posture, 2)

        # --- NUEVO: MOSTRAR CRONÓMETROS ---
        texto_tiempos = f"Pie: {formatear_tiempo(tiempo_de_pie)} | Sentado: {formatear_tiempo(tiempo_sentado)}"
        cv2.putText(image, texto_tiempos, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        color_agua = (255, 255, 255)
        if estado_bebiendo: color_agua = (0, 255, 0) # Verde al confirmar
        elif inicio_gesto_agua is not None: color_agua = (0, 255, 255) # Amarillo validando
        
        cv2.putText(image, f"Agua: {contador_agua} | {mensaje_agua}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_agua, 2)
        
        if not calibrated_y:
            cv2.putText(image, "[Pulsa C para Calibrar]", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow('AI Cycling Coach - Final', image)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): contador_agua = 0
        elif key == ord('c'):
            if hay_cara:
                calibrated_y = face_result.face_landmarks[0][1].y
                print("--- CALIBRADO ---")

cap.release()
cv2.destroyAllWindows()