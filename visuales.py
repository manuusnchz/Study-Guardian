# visuales.py
import cv2
import config

def dibujar_hud(image, mensaje_central, color_estado, contadores):
    h, w, _ = image.shape
    
    # Capa semitransparente superior
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # Mensaje Central
    cv2.putText(image, mensaje_central, (w//2 - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_estado, 3)

    # Panel inferior stats
    cv2.rectangle(image, (10, h-90), (300, h-10), (0,0,0), -1)
    
    # Agua
    cv2.putText(image, f"Agua: {contadores['agua']} {contadores['msg_agua']}", (20, h-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Distracción
    m = int(contadores['tiempo_distraccion'] // 60)
    s = int(contadores['tiempo_distraccion'] % 60)
    cv2.putText(image, f"Distraccion: {m:02d}:{s:02d}", (20, h-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    
    return image

def dibujar_ojos(image, face, w, h, cerrados):
    # Puntos: 159/145 (Izq), 386/374 (Der)
    col = (0, 0, 255) if cerrados else (0, 255, 0)
    for p in [159, 145, 386, 374]:
        px, py = int(face[p].x * w), int(face[p].y * h)
        cv2.circle(image, (px, py), 2, col, -1)

def dibujar_objeto(image, bbox, texto, color):
    cv2.rectangle(image, (bbox.origin_x, bbox.origin_y), 
                 (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), color, 2)
    cv2.putText(image, texto, (bbox.origin_x, bbox.origin_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def mostrar_debug_ojos(image, apertura, w):
    # Muestra el valor numérico para calibrar
    cv2.putText(image, f"Ojos: {apertura:.4f}", (w - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)