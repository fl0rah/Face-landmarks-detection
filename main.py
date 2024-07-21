import cv2

lbf_model = 'lbfmodel.yaml'
landmark = cv2.face.createFacemarkLBF()
landmark.loadModel(lbf_model)
vid = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    success, landmarks = landmark.fit(gray, faces)
    
    # keteri nkarum
    
    for lnd in landmarks:
        for x, y in lnd[0]:
            cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), 1)
    
    cv2.imshow('Camera_of_Emotions', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('o'):
        cv2.imwrite('emotion_cam.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

vid.release()
cv2.destroyAllWindows()




