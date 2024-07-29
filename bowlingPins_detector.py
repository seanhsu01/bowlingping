import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#from preprocessor import makesquare,resize_pixel
#from sklearn.preprocessing import makesquare,resize_pixel

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#classifier = load_model('digitsCNN.h5')
classifier = load_model('mnist-model.h5')

def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed

    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((height - width)/2)
            #print("Padding 1= ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = int((width - height)/2)
            #print("Padding 2= ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square


def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions

    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg




def main():

    parser = argparse.ArgumentParser(description='Find Hough circles from the image.')
    parser.add_argument('image_path', type=str, help='Full path of the input image.')
    parser.add_argument('--min_edge_threshold', type=int, help='Minimum threshold value for edge detection. Default 100.')
    parser.add_argument('--max_edge_threshold', type=int, help='Maximum threshold value for edge detection. Default 200.')

    args = parser.parse_args()

    img_path = args.image_path
    min_edge_threshold = 100
    max_edge_threshold = 200

    if args.min_edge_threshold:
        min_edge_threshold = args.min_edge_threshold

    if args.max_edge_threshold:
        max_edge_threshold = args.max_edge_threshold

    input_img = cv2.imread(img_path)

    #Edge detection on the input image
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    #ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

    saved_edge_image_path = "edge_{}".format(img_path)
    saved_edge_image = cv2.imwrite(saved_edge_image_path, edge_image)

    # Save the saved_edge_image
    if (saved_edge_image):
        print("Edge Image is saved successfully")
    else:
        print("Edge Image is not saved")

    # Displaying both image
    #cv2.imshow('Edge Image', edge_image)
    #cv2.waitKey(0) # press any key to exit

    #on real image
    print("saved_edge_image_path = {}".format(saved_edge_image_path))
    image = cv2.imread(saved_edge_image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('image',gray)
    #cv2.waitKey(0)


    blur = cv2.GaussianBlur(gray,(5,5),50)
    #cv2.imshow('blurred',blur)
    #cv2.waitKey(0)


    #https://www.wongwonggoods.com/all-posts/python/python_opencv/opencv-threshold/
    ret, thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv2.imshow('thres',thresh)
    #cv2.waitKey(0)


    kernel = np.ones((1,1),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)
    bg = cv2.dilate(closing,kernel,iterations=1)
    #cv2.imshow('bg',bg)
    #cv2.waitKey(0)

    canny = cv2.Canny(bg,20,150)
    #cv2.imshow('canny',canny)
    #cv2.waitKey(0)

    contours, hierarchy=cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours number:"+str(len(contours)))

    #cv2.destroyAllWindows()


    full_n = []

    image_count = 0
    for cnt in contours:

        #cv2.drawContours(image, contours, -1, (255,0,0), 1, lineType=cv2.LINE_AA)
        (x,y,w,h) = cv2.boundingRect(cnt)
        print("{}, {}, {}, {}".format(x, y, w, h))

        if  (w >= 45 and w <= 52 and h>= 30 and h <= 35) or (w >= 58 and w <= 62 and h>= 40 and h <= 45) :

            print("-----> {}, {}, {}, {}".format(x, y, w, h))

            roi = gray[y:y+h,x:x+w]
            output_folder = "saved_images"
            saved_image_path = os.path.join(output_folder, f"object_{image_count}.png")  # 使用os.path.join组合文件夹路径和文件名

            # 保存截取的图像
            cv2.imwrite(saved_image_path, roi)
            image_count += 1

            cv2.drawContours(image, [cnt], 0, (0,255,0), 3)


            ret, roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
            #print(roi)
            roi = makeSquare(roi)
            #print(roi.shape)
            roi = resize_to_pixel(28,roi)

            #cv2.imshow('roi',roi)
            #cv2.waitKey(0)

            roi = roi/255
            roi = roi.reshape(1,28,28,1)

            predict = classifier.predict(roi)[0]
            res = np.argmax(predict, axis=0)
            #print("predict = {}".format(predict))


            #res = str(classifier.predict_classes(roi)[0])
            #print("The number is: {}".format(res))
            #full_n.append(res)

            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.putText(image,res,(x,y+50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            print(img_path)



    # Save the saved_detected_image
    saved_detected_image_path = "detected_{}".format(img_path)

    saved_detected_image = cv2.imwrite(saved_detected_image_path, image)

    if (saved_detected_image):
        print("detected Pins Image is saved successfully")
    else:
        print("detected Pins Image is not saved")

if __name__ == "__main__":
    main()
    
    
    
#Readme 2023/8/28 How to run this file
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_30.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_33.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_34.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py LINE_ALBUM_bowlingPins_230827_36.jpg --min_edge_threshold 300 --max_edge_threshold 500
# python bowlingPins_detector.py detected_with_ids_LINE_ALBUM_bowlingPins_230827_39.jpg --min_edge_threshold 300 --max_edge_threshold 500