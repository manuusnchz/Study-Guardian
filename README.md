<div align="center">

  # 🛡️ STUDY GUARDIAN AI
  ### Asistente de Productividad y Ergonomía basado en Visión por Computador

  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
    <img src="https://img.shields.io/badge/MediaPipe-Pose_&_Face-00BACC?style=for-the-badge&logo=google&logoColor=white" />
    <img src="https://img.shields.io/badge/NumPy-Math-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  </p>

  

  <p>
    <b>Study Guardian</b> es un sistema inteligente que monitoriza en tiempo real la salud y el foco del estudiante. Utiliza múltiples modelos de IA para detectar fatiga, malas posturas y distracciones, generando un informe detallado al finalizar la sesión.
  </p>

</div>

---

## 🚀 Características Principales

Este proyecto integra **3 modelos de IA concurrentes** para analizar el comportamiento del usuario sin hardware adicional, solo una webcam.

| Funcionalidad | Descripción Técnica |
| :--- | :--- |
| **📵 Detector de Distracciones** | Detecta el uso de teléfonos móviles mediante **Object Detection**. Incluye lógica de filtrado espacial para evitar falsos positivos al comer. |
| **📏 Corrector Postural** | Calibración en tiempo real. Calcula la distancia euclidiana de puntos clave faciales (nariz) respecto a un eje calibrado. |
| **😴 Alerta de Fatiga** | Monitoriza el **EAR (Eye Aspect Ratio)** para detectar somnolencia y micro-sueños. Incluye buffers de tiempo para evitar falsas alarmas por parpadeo. |
| **💧 Tracking de Hidratación** | Detecta el gesto de beber agua combinando detección de manos, objetos (botellas) y proximidad a la boca. |
| **💡 Sensor de Iluminación** | Analiza el histograma de la imagen para alertar sobre condiciones de luz dañinas para la vista. |
| **📊 Smart Reporting** | Generación automática de informes `.txt` con estadísticas de sesión (tiempo de foco, pausas, salud). |

---

## 🛠️ Arquitectura del Proyecto

El código ha sido refactorizado desde un script monolítico a una **arquitectura modular** para asegurar escalabilidad y mantenimiento.

```text
📁 STUDY-GUARDIAN/
│
├── 📄 main.py         # Orquestador principal (Bucle de eventos y Lógica de Negocio)
├── 📄 servicios.py    # Capa de Infraestructura (Gestión de Modelos MediaPipe y Descargas)
├── 📄 visuales.py     # Capa de Presentación (Renderizado de HUD, Alertas y Gráficos)
├── 📄 config.py       # Configuración centralizada (Umbrales, Rutas y Constantes)
└── 📄 requirements.txt
