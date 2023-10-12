import cv2

model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(model_path)

# Increase the display size
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

image_counter = 0
capture_mode = False
recording_mode = False
out = None  # VideoWriter object

# Indicator text settings
font = cv2.FONT_HERSHEY_SIMPLEX
indicator_position = (10, 30)
font_scale = 1
font_color = (0, 0, 255)
font_thickness = 2

# Menu settings
menu_font_scale = 0.8
menu_font_color = (255, 255, 255)
menu_font_thickness = 1
menu_position = (10, 650)  # Adjust the position to the bottom
menu_spacing = 30  # Reduce the spacing
menu_items = ["Capture (c)", "Stop Capture (s)",
              "Start Recording (r)", "Stop Recording (e)", "Quit (q)"]


def draw_menu(frame):
    for i, item in enumerate(menu_items):
        y = menu_position[1] + i * menu_spacing
        cv2.putText(frame, item, (menu_position[0], y), font,
                    menu_font_scale, menu_font_color, menu_font_thickness)


def capture_image():
    global image_counter
    global capture_mode

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

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
        cv2.putText(frame, 'Face Detected', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if recording_mode:
        out.write(frame)  # Write the frame to the video
        # Add a recording indicator
        cv2.putText(frame, 'Recording...', indicator_position,
                    font, font_scale, font_color, font_thickness)

    draw_menu(frame)  # Draw the menu on the frame

    cv2.imshow('Object Detection', frame)


cv2.namedWindow('Object Detection')

# Flag to track whether the window should be closed
window_closed = False

while not window_closed:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if recording_mode:
        out.write(frame)  # Write the frame to the video
        # Add a recording indicator
        cv2.putText(frame, 'Recording...', indicator_position,
                    font, font_scale, font_color, font_thickness)

    draw_menu(frame)  # Draw the menu on the frame

    cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        capture_mode = True
        print("Capture mode activated")

    if key == ord('s'):
        capture_image()

    if key == ord('r'):
        recording_mode = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving the video
        out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                              (1280, 720))  # Increased resolution

    if key == ord('e'):
        recording_mode = False
        if out:
            out.release()
            print("Recording stopped and saved as 'output.avi'")

    if key == ord('q'):
        break

    # Check if the window close button has been clicked
    if cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
        window_closed = True

cap.release()
cv2.destroyAllWindows()
