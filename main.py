import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model_final_image_2.keras")  # Replace with your model path

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels (update according to your model)
emotion_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']



# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale (for better detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y + h, x:x + w]

            # Preprocess face for the model
            face = cv2.resize(face, (224, 224))  # Resize to match model input
            face = image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = tf.keras.applications.efficientnet_v2.preprocess_input(face)  # Normalize

            # Predict emotion
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]  # Get the highest probability emotion

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Display prediction
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
