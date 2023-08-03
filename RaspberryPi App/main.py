import cv2
import os
import numpy as np
from cv2 import dnn_superres
import tensorflow as tf
import time

class RulerNumberData:
    def __init__(self, image, x_min, y_min, x_max, y_max, confidence):
        self.image = image
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.confidence = confidence
def process_up_scale_image(image):
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "./EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 3)
    result = sr.upsample(image)
    # Save the image
    cv2.imwrite(".upscaled.png", result)
    return result
def convert_to_binary_otsu(image):
    # Xác định ngưỡng tự động bằng phương pháp Otsu
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image
def validateResult(all_numbers_detected):
    numbers_check = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    for number_check in numbers_check:
        if number_check not in all_numbers_detected:
            if len(all_numbers_detected) == 0:
                return True
            print("Fail validate!")
            return False
        all_numbers_detected.remove(number_check)
    return True
def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)
def detectRuler(image, conf):
    """
    Phát hiện đối tượng thước và trả về hình ảnh của đối tượng thước đó (Đã crop khỏi ảnh ban đầu)\n
    Args:
        image (Opencv) : Hình ảnh đầu vào
        conf : Ngưỡng
    Returns
        Trả về hình ảnh thước đã crop (Opencv)
    """
    # Đường dẫn tới model TensorFlow Lite (.tflite)
    model_path = 'ModelDetectRuler-fp16.tflite'

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_origin = image
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)  # Mở rộng kích thước ảnh để tương thích với input shape của model
    image = image.astype(np.float32) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # get tensor  x(1, 25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = np.argmax(output_data[0][:, 5:], axis=1)
    # Áp dụng non-maximum suppression
    selected_indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold = 0.8, nms_threshold = 0.9)
    output_data = np.column_stack((boxes[selected_indices], scores[selected_indices], classes[selected_indices]))

    ruler = []
    mirror = []
    for item in output_data:
        x, y, w, h = item[:4]
        x1, y1, x2, y2 = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]) * 640
        score = item[4:5]
        label = item[5:]
        if score > conf:
            if label == 0:
                ruler.append([int(x1), int(y1), int(x2), int(y2), score, label])
            elif label == 1:
                mirror.append([int(x1), int(y1), int(x2), int(y2), score, label])
    scale_x = image_origin.shape[1] / 640
    scale_y = image_origin.shape[0] / 640
    if len(mirror) > 0:
        box_origin = [
                int(mirror[0][0] * scale_x),
                int(mirror[0][1] * scale_y),
                int(mirror[0][2] * scale_x),
                int(mirror[0][3] * scale_y)
            ]
        cv2.imwrite("./CacheDetect/mirrorCropped.png", image_origin[box_origin[1]:box_origin[3], box_origin[0]:box_origin[2]])
        # cv2.rectangle(image_origin, (box_origin[0], box_origin[1]), (box_origin[2], box_origin[3]), (0, 255, 0), 4)
        image_origin_rec = image_origin
        cv2.rectangle(image_origin_rec, (box_origin[0], box_origin[1]), (box_origin[2], box_origin[3]), (255, 0, 0), 4)
        cv2.putText(image_origin_rec, "Mirror: " + str(mirror[0][4]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("./CacheDetect/mirrorDetected.png", image_origin_rec)
    if len(ruler) > 0:
        box_origin = [
                int(ruler[0][0] * scale_x),
                int(ruler[0][1] * scale_y),
                int(ruler[0][2] * scale_x),
                int(ruler[0][3] * scale_y)
            ]
        cv2.imwrite("./CacheDetect/rulerCropped.png", image_origin[box_origin[1]:box_origin[3], box_origin[0]:box_origin[2]])
        # cv2.rectangle(image_origin, (box_origin[0], box_origin[1]), (box_origin[2], box_origin[3]), (0, 255, 0), 4)
        image_origin_rec = image_origin
        cv2.rectangle(image_origin_rec, (box_origin[0], box_origin[1]), (box_origin[2], box_origin[3]), (0, 255, 0), 4)
        cv2.putText(image_origin_rec, "Ruler: " + str(ruler[0][4]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("./CacheDetect/rulerDetected.png", image_origin_rec)
        return image_origin[box_origin[1]:box_origin[3], box_origin[0]:box_origin[2]], image_origin_rec
    
    return None, image
    
def detectNumberInRuler(image, conf):
    """
    Phát hiện đối tượng chữ số trên thước và trả về các đối tượng chữ số đã crop\n
    Args:
        rulerImage (Opencv) : Hình ảnh thước đầu vào
        conf : Ngưỡng
    Returns
        Trả về một mảng chứa các hình ảnh của chữ số đã crop
    """
    # Load YOLOv5 model
    model_path = 'ModelDetectNumberInRuler_best-fp16.tflite'

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # Lấy thông tin input và output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Đọc ảnh và tiền xử lý
    image_origin = image
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)  # Mở rộng kích thước ảnh để tương thích với input shape của model
    image = image.astype(np.float32) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # get tensor  x(1, 25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = np.squeeze(output_data[..., 5:]) # get classes
    selected_indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold = 0.8, nms_threshold = 0.9)
    data = []
    for index in selected_indices:
        data.append([np.hstack((boxes[index], scores[index], classes[index]))])
    data_handle = []
    for item in data:
        x = item[0][0]
        y = item[0][1]
        w = item[0][2]
        h = item[0][3]
        x1, y1, x2, y2 = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]) * 640
        score = item[0][4]
        classNumber = item[0][5]
        data_handle.append([int(x1), int(y1), int(x2), int(y2), score, classNumber])
    scale_x = image_origin.shape[1] / 640
    scale_y = image_origin.shape[0] / 640
    data_return = []
    y = 100
    for index, item in enumerate(data_handle):
        y += 50
        box_origin = [
            int(item[0] * scale_x),
            int(item[1] * scale_y),
            int(item[2] * scale_x),
            int(item[3] * scale_y)
        ]
        numberImageCropped = image_origin[box_origin[1]:box_origin[3], box_origin[0]:box_origin[2]]
        cv2.imwrite("test2.png", numberImageCropped)
        data_return.append(RulerNumberData(numberImageCropped, box_origin[0], box_origin[1], box_origin[2], box_origin[3], item[4]))
        cv2.imwrite("./CacheDetect/numberDetectedCropped_{}.png".format(index), numberImageCropped)
        cv2.rectangle(image_origin, (box_origin[0], box_origin[1]), (box_origin[2], box_origin[3]), (0, 255, 0), 1)
        cv2.putText(image_origin, str(item[4]), (box_origin[2], box_origin[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite("./CacheDetect/numberDetected.png", image_origin)
    return data_return, image_origin
def rulerNumberClassification(image, conf):
    """
    Phân loại chữ số của một ảnh\n
    Args:
        image (Opencv) : Hình ảnh đầu vào
        conf : Ngưỡng
    Returns
        Một chữ số đại diện cho chữ số dự đoán từ ảnh đầu vào
    """
    # Đường dẫn tới model TensorFlow Lite (.tflite)
    model_path = 'best_model_me.tflite'

    # Load model TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Lấy thông tin input và output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # image = process_up_scale_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = convert_to_binary_otsu(image)
    binary_image = cv2.bitwise_not(binary_image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.imwrite("TestBlue.png", image)

    # Đảm bảo ảnh có giá trị từ 0 đến 255

    image = cv2.resize(image, (32, 32))  # Resize ảnh về kích thước 32x32
    image = np.expand_dims(image, axis=0)  # Mở rộng kích thước ảnh để tương thích với input shape của model
    image = image.astype(np.float32) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

    # Đặt dữ liệu vào tensor input
    interpreter.set_tensor(input_details[0]['index'], image)

    # Chạy mô hình
    interpreter.invoke()

    # Lấy kết quả từ tensor output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)
    predicted_class = np.argmax(output_data)
    #print(predicted_class)
    cv2.putText(image, str(output_data[0][predicted_class]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print("P : " + str(output_data[0][predicted_class]))
    if 0 <= predicted_class <= 8:
        if output_data[0][predicted_class] >= conf:
            return predicted_class + 1
    elif predicted_class == 9:
        if output_data[0][predicted_class] >= conf:
            return 0
def measurementWaterLevel(image):
    rulerImageOrigin = image
    all_numbers_detected = []
    minNumberDetect = 9
    maxNumberDetect = 0
    minNumberDetectYMax = 0
    rulerDetected, image_origin_rec =  detectRuler(image, 0.6)
    if rulerDetected is not None:
        numbers_detected, rulerImageOrigin = detectNumberInRuler(rulerDetected, 0.6)
        for number_detected in numbers_detected:
            cv2.imwrite("test3.png", number_detected.image)
            number_predict = rulerNumberClassification(number_detected.image, 0.25) # Phân loại chữ số trên thước
            print("Dự đoán {}, xác suất {}".format(number_predict, number_detected.confidence))
            if number_predict is not None:
                all_numbers_detected.append(number_predict)
                if len(all_numbers_detected) == 0:
                    minNumberDetectYMax = number_detected.y_max
                if minNumberDetect > number_predict: # Kiểm tra chữ số nhỏ nhất được phát hiện
                    minNumberDetect = number_predict
                    minNumberDetectYMax = number_detected.y_max
                if maxNumberDetect < number_predict:
                    maxNumberDetect = number_predict
    end_time = time.time()
    if len(all_numbers_detected) > 0:
        # Validate
        validate_result =  validateResult(all_numbers_detected)
        if not validate_result:
            return None, image_origin_rec, rulerImageOrigin

        #print("Thời gian thực thi : {}".format(end_time - start_time))
        #print(f"Chiều cao pixel của thước : {rulerDetected.shape[0]}, vị trí thấp nhất của chữ số : {minNumberDetectYMax}")
        #print("Chữ số nhỏ nhất : {}", minNumberDetect)
        #print("Chữ số lớn nhất : {}", maxNumberDetect)
        # print("Độ cao mực nước : {}", (maxNumberDetect - minNumberDetect + 1 ) * 2 + (rulerDetected.shape[0] - minNumberDetectYMax) / minNumberDetectYMax * (maxNumberDetect - minNumberDetect + 1) * 2)
        result = (maxNumberDetect - minNumberDetect + 1 ) * 2 + (rulerDetected.shape[0] - minNumberDetectYMax) / minNumberDetectYMax * (maxNumberDetect - minNumberDetect + 1) * 2
        print(rulerDetected.shape[0])
        print(result)
        return result, image_origin_rec, rulerImageOrigin
    else:
        return None, image_origin_rec, rulerImageOrigin

import cv2
from cv2 import VideoCapture
import threading
import paho.mqtt.client as mqtt
import time
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

broker_address = "broker.emqx.io"
broker_port = 1883
topic = "traistorm"

def on_connect(client, userdata, flags, rc):
    print("Kết nối thành công!")

def on_publish(client, userdata, mid):
    print("Đã gửi thông điệp thành công")
def on_disconnect(client, userdata, rc):
    global isConnect
    if rc != 0:
        print("Unexpected MQTT disconnection. Will auto-reconnect")
        
client = mqtt.Client()
client.on_connect = on_connect
client.on_publish = on_publish
client.on_disconnect = on_disconnect

try:
    client.connect(broker_address, broker_port, 60)
except:
    print("An exception occurred")
prev_time = 0
start_time_ms = 0
res = None
ret = None
frame = None
waterLevel = 0

i = 0
index = 0
client.loop_start()
cam = VideoCapture(0)
while True:
    try:
        print("Res: {}".format(res))
        ret, frame = cam.read()
        if ret:
            print("Take sceenshoot!")
            cv2.imwrite("NewData6/" + str(i) + ".png", frame)
        i += 1
        # cv2.imshow("window", frame)

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        start_time_measurement = time.time()
        res, image_origin_rec, rulerImageOrigin = measurementWaterLevel(frame)
        # Xử lý khung ảnh tại đây (ví dụ: lưu ảnh, hiển thị, xử lý...)
        end_time_measurement = time.time()
        time_measurement = end_time_measurement - start_time_measurement
        print("Time measurement : {}".format(time_measurement))
        start_time_measurement = time.time()
        cv2.imwrite("./Cache6/" + str(i) + "_" + str(res) + "_" + str(time_measurement) + ".png", image_origin_rec)

        # Gửi thông tin về mực nước
        try:
            if res is None:
                client.publish(topic, str(-1) + "_" + str(time_measurement))
            else:
                client.publish(topic, str(res) + "_" + str(time_measurement))
        except:
            print("Error publish data")
        
        
    except:
        print("Error when measurement")
        time.sleep(2)
# Giải phóng video và đóng cửa sổ hiển thị
cv2.destroyAllWindows()
