import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# config detecção / model_selection=0 para rostos pertos, =1 para distancias maiores
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Erro ao Acessar a Câmera")
    exit()

while True:
    retorno, frame = cam.read()
    
    if not retorno:
        print("Erro ao Capturar Vídeo")
        break

    # converte o vídeo para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detecção
    resultado = face_detection.process(rgb_frame)

    contador = 0

    if resultado.detections:
        for detection in resultado.detections:
            contador += 1
            
            # desenha as detecçções no vídeo
            mp_drawing.draw_detection(frame, detection)

    cv2.putText(frame, f"Integrantes: {contador}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sistema de contagem", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
face_detection.close()