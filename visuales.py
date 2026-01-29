# visuales.py
import cv2
import config

def dibujar_hud(image, mensaje_central, color_estado, contadores):
    h, w, _ = image.shape
    
    # 1. GESTIÓN DE PAUSA (Oscurecer pantalla)
    overlay = image.copy()
    
    # Si estamos en pausa, oscurecemos MÁS y cambiamos el mensaje
    if contadores.get('en_pausa', False):
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1) # Pantalla completa negra
        alpha = 0.7
        mensaje_central = "AUSENTE - PAUSADO"
        color_estado = (150, 150, 150) # Gris
    else:
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1) # Solo barra superior
        alpha = 0.6

    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 2. TEXTOS
    # Mensaje Central
    cv2.putText(image, mensaje_central, (w//2 - 200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_estado, 3)

    # Panel inferior stats
    cv2.rectangle(image, (10, h-90), (350, h-10), (0,0,0), -1)
    
    # Tiempo de Estudio (NUEVO)
    t_estudio = contadores.get('tiempo_estudio', 0)
    m_est = int(t_estudio // 60)
    s_est = int(t_estudio % 60)
    cv2.putText(image, f"Tiempo Estudio: {m_est:02d}:{s_est:02d}", (20, h-65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Agua
    cv2.putText(image, f"Agua: {contadores['agua']} {contadores['msg_agua']}", (20, h-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Distracción
    m_dis = int(contadores['tiempo_distraccion'] // 60)
    s_dis = int(contadores['tiempo_distraccion'] % 60)
    cv2.putText(image, f"Distraccion: {m_dis:02d}:{s_dis:02d}", (20, h-15), 
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


def dibujar_aviso_luz(image, brillo):
    # Solo dibujamos si la luz es insuficiente
    if brillo < config.UMBRAL_OSCURIDAD:
        h, w, _ = image.shape
        
        # Texto de aviso abajo a la derecha
        texto = f"LUZ BAJA ({int(brillo)})"
        cv2.putText(image, texto, (w - 280, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Icono de "bombilla apagada" (círculo gris)
        cv2.circle(image, (w - 40, h - 40), 10, (100, 100, 100), -1)