# main.py
import cv2
import time
import math
import mediapipe as mp
import numpy as np

# Importamos nuestros módulos
import config
from servicios import GestorIA
import visuales

def main():
    # 1. INICIALIZACIÓN
    gestor_ia = GestorIA() # Carga los modelos automáticamente
    cap = cv2.VideoCapture(0)
    
    # Variables de estado
    contador_agua = 0
    estado_bebiendo = False
    inicio_gesto_agua = None
    
    inicio_ojos_cerrados = None
    inicio_distraccion = None
    tiempo_distraccion_total = 0.0

    print(">>> STUDY GUARDIAN MODULAR ACTIVADO 📚")

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        # Preparar imagen
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = int(time.time() * 1000)

        # 2. DETECCIÓN (Usando nuestro módulo servicios)
        face_result, hand_result, obj_result = gestor_ia.detectar(mp_image, timestamp_ms)

        # Variables para este frame
        mensaje_central = "ESTUDIANDO..."
        color_estado = (0, 255, 0)
        boca_x, boca_y = 0, 0

        # --- LÓGICA 1: CARA Y SUEÑO ---
        if face_result.face_landmarks:
            face = face_result.face_landmarks[0]
            boca = face[13]
            boca_x, boca_y = int(boca.x * w), int(boca.y * h)

            # Cálculo Ojos
            dist_izq = abs(face[159].y - face[145].y)
            dist_der = abs(face[386].y - face[374].y)
            apertura = (dist_izq + dist_der) / 2
            
            # Debug visual (para calibrar)
            visuales.mostrar_debug_ojos(image, apertura, w)

            if apertura < config.UMBRAL_OJOS_CERRADOS:
                if inicio_ojos_cerrados is None: inicio_ojos_cerrados = time.time()
                t_cerrado = time.time() - inicio_ojos_cerrados
                
                pct = min(int((t_cerrado/config.TIEMPO_PARA_DORMIRSE)*100), 100)
                mensaje_central = f"DURMIENDO... {pct}%"
                
                if t_cerrado > config.TIEMPO_PARA_DORMIRSE:
                    mensaje_central = "¡¡ DESPIERTA !!"
                    color_estado = (0, 0, 255)
                    print("\a")
                
                visuales.dibujar_ojos(image, face, w, h, True)
            else:
                inicio_ojos_cerrados = None
                visuales.dibujar_ojos(image, face, w, h, False)

        # --- LÓGICA 2: OBJETOS (MÓVIL) ---
        hay_movil = False
        if obj_result.detections:
            for det in obj_result.detections:
                cat = det.categories[0].category_name
                bbox = det.bounding_box
                
                # Centro objeto
                cx = bbox.origin_x + bbox.width // 2
                cy = bbox.origin_y + bbox.height // 2
                dist_boca = 999
                if boca_x > 0:
                    dist_boca = math.sqrt((boca_x - cx)**2 + (boca_y - cy)**2)

                if cat == 'cell phone':
                    if dist_boca < config.UMBRAL_IGNORAR_MOVIL:
                        visuales.dibujar_objeto(image, bbox, "IGNORADO", (200,200,200))
                    else:
                        hay_movil = True
                        visuales.dibujar_objeto(image, bbox, "MOVIL", (0,0,255))
                elif cat == 'bottle':
                    visuales.dibujar_objeto(image, bbox, "AGUA", (255,255,0))

        if hay_movil:
            if inicio_distraccion is None: inicio_distraccion = time.time()
            mensaje_central = "¡¡ SUELTA EL MÓVIL !!"
            color_estado = (0, 0, 255)
        else:
            if inicio_distraccion is not None:
                tiempo_distraccion_total += (time.time() - inicio_distraccion)
                inicio_distraccion = None

        # --- LÓGICA 3: AGUA ---
        hay_mano = False
        mano_x, mano_y = 0, 0
        if hand_result.hand_landmarks:
            hay_mano = True
            mn = hand_result.hand_landmarks[0][0]
            mano_x, mano_y = int(mn.x * w), int(mn.y * h)
            cv2.circle(image, (mano_x, mano_y), 5, (255,100,0), -1)

        msg_agua = ""
        if boca_x > 0 and hay_mano:
            dist = math.sqrt((boca_x - mano_x)**2 + (boca_y - mano_y)**2)
            
            # Dibujo de radio de detección (Debug)
            if dist < config.UMBRAL_BEBER:
                cv2.circle(image, (boca_x, boca_y), config.UMBRAL_BEBER, (0, 255, 255), 2)
                
                if inicio_gesto_agua is None: inicio_gesto_agua = time.time()
                if (time.time() - inicio_gesto_agua) > config.DURACION_MINIMA_TRAGO:
                    msg_agua = "+1"
                    if not estado_bebiendo:
                        contador_agua += 1
                        estado_bebiendo = True
                else:
                    msg_agua = "..."
            else:
                inicio_gesto_agua = None
                estado_bebiendo = False

        # 3. DIBUJAR HUD FINAL
        datos_hud = {
            'agua': contador_agua,
            'msg_agua': msg_agua,
            'tiempo_distraccion': tiempo_distraccion_total
        }
        image = visuales.dibujar_hud(image, mensaje_central, color_estado, datos_hud)

        cv2.imshow('Study Guardian Modular', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()