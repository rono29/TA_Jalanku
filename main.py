import cv2
import time
import imutils
import numpy as np
import tensorflow as tf
from keras.models import load_model
import geocoder_clear as geocoder
import os
import mysql.connector
import base64
import requests
import tflite_runtime.interpreter as tflite

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# akses database
config = mysql.connector.connect(
    host="localhost",
    user="rono",
    password="password",
    database="lat_long3",
)
cursor = config.cursor()

size = 224

# Define function for prediction using TensorFlow Lite
def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (224, 224))
    currentFrame = currentFrame.reshape(1, 224, 224, 3).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], currentFrame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    max_prob_index = np.argmax(output_data)
    max_prob = np.max(output_data)
    labels = ["crack", "plain", "potholes"]
    if max_prob > 0.90:
        return labels[max_prob_index], max_prob
    return "none", 0

# Main function
if __name__ == '__main__':
    camera = cv2.VideoCapture(-1)
    _, reference_frame = camera.read()
    avg_frame = np.float32(reference_frame)
    ref_frame = None
    frame_count = 0

    i = 0
    g = geocoder.ip('192.168.100.152')
    result_path = "storage_img_coord"
    starting_time = time.time()
    start_point = g.latlng  # Start point coordinates
    distance_threshold = 10  # Threshold distance in meters
    prev_point = 0

    # Define API key and endpoint
    API_KEY = '67d62c77f74f4115befc4bc7106cfc6f'
    API_ENDPOINT = 'https://api.opencagedata.com/geocode/v1/json'
    show_pred = True

    # Loop until interrupted
    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        pothole_type, prob = predict_pothole(frame)
        keypress_toshow = cv2.waitKey(1)

        
        
        if (show_pred):
            if pothole_type in ["crack", "potholes"] and prob >= 0.95:
                cv2.putText(clone, f'{pothole_type.capitalize()} Detected: Probability {prob*100:.2f}%', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
                # Calculate distance from previous point
                current_point = g.latlng
                distance = geocoder.distance(prev_point, current_point)
                if distance >= distance_threshold:
                    # Save image and location data
                    image_filename = f'{pothole_type}{i}.jpg'
                    cv2.imwrite(os.path.join(result_path, image_filename), frame)
                    with open(os.path.join(result_path, f'{pothole_type}{i}.txt'), 'w') as f:
                        f.write(str(g.latlng))
                    params = {'q': f'{g.lat},{g.lng}', 'key': API_KEY}
                    response = requests.get(API_ENDPOINT, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if data and "results" in data and len(data["results"]) > 0:
                            result = data["results"][0]
                            if "components" in result and isinstance(result["components"], dict):
                                city = result["components"].get("city")
                                subdistrict = city
                        kelurahan = subdistrict
                        pothole = f'{pothole_type}{i}.txt'
                        status = 'belum di perbaiki'
                        with open(os.path.join(result_path, image_filename), 'rb') as image_files:
                            encoded_string = base64.b64encode(image_files.read()).decode('utf-8')
                        f1 = [pothole, g.lat, g.lng, encoded_string, kelurahan, status]
                        cursor.execute("INSERT INTO coordinates (pothole,latitude,longitude,image,kelurahan,status) VALUES (%s,%s,%s,%s,%s,%s)", f1)
                        config.commit()
                        prev_point = current_point
                        i += 1

        cv2.imshow("Video original", clone)
        frame_count += 1

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
