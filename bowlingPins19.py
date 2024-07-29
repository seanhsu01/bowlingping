from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import sys
import tensorflow as tf
import keras
import cv2
from keras.datasets import mnist
import PIL.Image as Image
import matplotlib.pylab as plt
import sys

IMAGE_SHAPE = (224, 224)

# 載入訓練好的模型
model = tf.keras.models.load_model('C:/Users/User/Desktop/Bowling/bowling_num_1107')

# 設定資料夾路徑
folder_path = 'C:\\Users\\User\\OneDrive\\桌面\\BowlingPinsDetector\\saved_images'

# 定義數字類別標籤
class_labels = ["N1", "N10", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]
#儲存辨識結果
result_labele = []
# 遍歷資料夾中的每張圖片
def bowling():
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        # 預處理待辨識的圖片
        sample_img_1 = Image.open(img_path).resize(IMAGE_SHAPE)
        # Assuming sample_img_1 is a grayscale image
        sample_img_1 = np.expand_dims(sample_img_1, axis=-1)  # Add color channel

        # Resize the image to match the model's input shape
        sample_img_1 = tf.image.resize(sample_img_1, (224, 224))

        # Convert grayscale to RGB
        sample_img_1 = tf.image.grayscale_to_rgb(sample_img_1)

        # Add batch dimension and convert to float
        sample_img_1 = tf.expand_dims(tf.cast(sample_img_1, tf.float32), axis=0)

        # Now you can use the resized and reshaped image for prediction
        result = model.predict(sample_img_1)

        sample_img_1 = np.array(sample_img_1) / 255.0

        result = model.predict(sample_img_1)

        predicted_class = tf.math.argmax(result[0], axis=-1)

        # 將批次維度擠壓掉，只保留圖片的形狀
        # 壓縮大小為1的維度
        sample_img_1 = np.squeeze(sample_img_1, axis=0)

        # 顯示預測結果
        print(f"File: {filename}, Predicted Class: {class_labels}, Probability: {result[0][predicted_class]}")

        #顯示圖片
        plt.imshow(sample_img_1)
        plt.axis('off')
        predicted_class_name = class_labels[predicted_class]
        result_labele.append(predicted_class_name.title())
        print(set(result_labele))
        _ = plt.title("Prediction: " + predicted_class_name.title())
        plt.show()


# Readme 2023/8/28 How to run this file
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_30.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_33.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_34.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_36.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_37.jpg --min_edge_threshold 300 --max_edge_threshold 500