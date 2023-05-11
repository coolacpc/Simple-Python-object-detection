import cv2

model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(model_path)

cap = cv2.VideoCapture(0)
image_counter = 0
capture_mode = False

def capture_image():
    global image_counter
    global capture_mode

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if capture_mode and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            image_path = f"face_{image_counter}.jpg"
            cv2.imwrite(image_path, face_img)
            image_counter += 1
            print(f"Captured image: {image_path}")

        capture_mode = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

cv2.namedWindow('Object Detection')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        capture_mode = True
        print("Capture mode activated")

    if key == ord('s'):
        capture_image()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
