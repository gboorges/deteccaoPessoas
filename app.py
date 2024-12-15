import cv2
import mediapipe as mp

# Config para a detecção dos rostos e das mãos
mp_face_detection = mp.solutions.face_detection
mp_hand_detection = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Config detecção rostos / model_selection=0 para rostos pertos, =1 para distâncias maiores
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Config detecção mãos / model_selection=0 para rostos pertos, =1 para distâncias maiores
hands = mp_hand_detection.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Erro ao Acessar a Câmera")
    exit()

while True:
    # Lê o frame da câmera
    retorno, frame = cam.read()
    
    if not retorno:
        print("Erro ao Capturar Vídeo")
        break

    # Converte o vídeo para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecção dos rostos
    resultado_faces = face_detection.process(rgb_frame)

    # Detecção das mãos
    resultado_maos = hands.process(rgb_frame)

    contador_faces = 0
    contador_maos = 0

    # Contabilização dos rostos detectados
    if resultado_faces.detections:
        for detection in resultado_faces.detections:
            contador_faces += 1
            
            # Desenha as detecções no vídeo
            mp_drawing.draw_detection(frame, detection)
    
    # Contabilização das mãos detectadas
    if resultado_maos.multi_hand_landmarks:
        for landmarks in resultado_maos.multi_hand_landmarks:
            contador_maos += 1

            # Desenha as detecções no vídeo
            mp_drawing.draw_landmarks(frame, landmarks, mp_hand_detection.HAND_CONNECTIONS)

    # Exibi o texto de contagem de rostos e mãos
    cv2.putText(frame, f"Rostos detectados: {contador_faces}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Maos detectadas: {contador_maos}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Abre uma janela com a exibição da imagem com as detecções
    cv2.imshow("Sistema de Contagem de Rostos e Mãos", frame)

    # Comando para encerrar a exibição da imagem
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cam.release()
cv2.destroyAllWindows()
face_detection.close()
hands.close()