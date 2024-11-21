import cv2 as cv

# Detector de faces
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Objeto de Vídeo
video = cv.VideoCapture(0)

# Looping para ler os frames do vídeo
while True:

    # Leitura do primeiro frame
    ret, frame = video.read()
    if not ret:
        break

    # Converter o frame para escala cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar o rosto no frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(300, 300))
    
    # Desenhar um retângulo ao redor do rosto
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    
    # Apresentar o primeiro frame
    cv.imshow("Frame", frame)
    if cv.waitKey(33) == ord("q"):
        break

video.release()
cv.destroyAllWindows()