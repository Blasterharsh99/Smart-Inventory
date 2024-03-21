import os.path
import requests
import cv2
import numpy as np
import imutils
from ultralytics import YOLO

model = YOLO("Models/SKU110K_5epochs.pt")
loreal_model = YOLO('Models/Loreal_50epochs.pt')
def predict_inventory(frame):
    prediction = model.predict(frame, project='Temp', name='Photos', show=True)
    return prediction

def count(results):
    for result in results:
        dabba = result.boxes
    return len(dabba)

url = "http://192.168.137.128:8080/shot.jpg"
i=1

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)

    result = predict_inventory(img)

    output = count(result)
    output_folder = f'Temp/Photos{i}/'
    predicted_image_path = os.path.join(output_folder,"image0.jpg")
    if os.path.exists(predicted_image_path):
        output_image = cv2.imread(predicted_image_path)
        cv2.putText(output_image, f"Prediction:{output}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Android_cam", output_image)
        i+=1
    else:
        cv2.putText(img, f"Prediction:{output}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Android_cam", img)
        i+=1

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()