






# better face detection algorithm

# example of face detection with mtcnn
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
 
# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array






# poor face detection algorithm

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import glob



i = 0
crop_imgs = []
no_face = []
filenames = glob.glob('/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba/*.jpg')
filenames.sort()
for filename in filenames: 
    img = cv2.imread(filename)
    # im=Image.open(filename)
    # im = im.convert('L')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    try:
        (x,y,w,h) = faces[-1]
        crop_img = img[y:y+h, x:x+w]
        dim = (64, 64)
        resize_img = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        crop_imgs.append(resize_img)
        cv2.imwrite("/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba_resized/" + filename[-10:-4] + ".jpg", resize_img)
        i+= 1
    except:
        # no_face.append(img)
        # crop_imgs.append(np.ones((100,100,3)))
        # cv2.imwrite("/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba_resized/" + filename[-10:-4] + ".jpg", np.ones((100,100,3)))
        pixels = extract_face("/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba/000036.jpg")
        dim = (64, 64)
        resize_img = cv2.resize(pixels, dim, interpolation = cv2.INTER_AREA)
        crop_imgs.append(resize_img)
        im_rgb = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba_resized/" + filename[-10:-4] + ".jpg", im_rgb)

        print(i)
        i+= 1
        
    # if i ==500:
    #     break



 
# load the photo and extract the face
pixels = extract_face("/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba/000036.jpg")
# plot the extracted face
pyplot.imshow(pixels)
# show the plot
pyplot.show()





i = 0
crop_imgs = []
no_face = []
filenames = glob.glob('/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba/*.jpg')
filenames.sort()
for filename in filenames: 

    pixels = extract_face(filename)
    dim = (64, 64)
    resize_img = cv2.resize(pixels, dim, interpolation = cv2.INTER_AREA)
    crop_imgs.append(resize_img)
    im_rgb = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/home/abdollahpour/python/Datasets/CelebA/img_align_celeba/img_align_celeba_resized/" + filename[-10:-4] + ".jpg", im_rgb)
    i+= 1
    print(i)


        
    # if i ==500:
    #     break



