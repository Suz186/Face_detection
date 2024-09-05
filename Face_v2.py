import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Setup Mediapipe Face Detection and Face Mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)

video_capture = cv2.VideoCapture(0)
face_counter = 0
output_folder = 'captured_faces'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_detection = face_detection.process(rgb_frame)

    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

            # Process face mesh
            results_mesh = face_mesh.process(rgb_frame)
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    # Save the captured face image
                    face_filename = os.path.join(output_folder, f"face_{face_counter}.jpg")
                    cv2.imwrite(face_filename, frame)
                    face_counter += 1
                    print(f"Captured and saved face as {face_filename}")

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
