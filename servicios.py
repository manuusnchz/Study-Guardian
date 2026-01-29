# servicios.py
import os
import urllib.request
import mediapipe as mp
import config  # Importamos tu configuración

class GestorIA:
    def __init__(self):
        self._descargar_modelos()
        self._iniciar_detectores()

    def _descargar_modelos(self):
        detectores = [
            (config.URL_FACE, config.FILE_FACE),
            (config.URL_HAND, config.FILE_HAND),
            (config.URL_OBJ, config.FILE_OBJ)
        ]
        for url, filename in detectores:
            if not os.path.exists(filename):
                print(f"⏳ Descargando {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print("✅ Descargado.")
                except Exception as e:
                    print(f"❌ Error descargando {filename}: {e}")

    def _iniciar_detectores(self):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Detector de Cara
        options_face = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.FILE_FACE),
            running_mode=VisionRunningMode.VIDEO, num_faces=1)
        self.face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(options_face)

        # Detector de Manos
        options_hand = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.FILE_HAND),
            running_mode=VisionRunningMode.VIDEO, num_hands=2)
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(options_hand)

        # Detector de Objetos
        options_obj = mp.tasks.vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=config.FILE_OBJ),
            running_mode=VisionRunningMode.VIDEO,
            score_threshold=config.CONF_OBJETOS,
            category_allowlist=['cell phone', 'bottle'])
        self.obj_detector = mp.tasks.vision.ObjectDetector.create_from_options(options_obj)

    def detectar(self, mp_image, timestamp):
        # Ejecuta las 3 IAs y devuelve los resultados
        face = self.face_detector.detect_for_video(mp_image, timestamp)
        hand = self.hand_detector.detect_for_video(mp_image, timestamp)
        obj = self.obj_detector.detect_for_video(mp_image, timestamp)
        return face, hand, obj