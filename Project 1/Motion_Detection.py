#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Function to read the input Images
def read_input_images():
    path = "C:/Users/Shiva Kumar Dande/Downloads/RedChair/RedChair/"
    input_images = []
    #Reading the images from the given path
    #and storing them in a list input_images[]
    for image_Name in os.listdir(path):
        img = cv.imread(os.path.join(path, image_Name))
        if img is not None:
            input_images.append(img)
    
    #No of images in the list
    n = len(input_images)
    return input_images, n

# Method to convert the images to gray scale
def covert_to_grayScale(images):
    gray_img = []
    for img in images:
        gray_img.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return gray_img

# Function to apply the operator(1D Differential operator or 1D Gaussian operator) 
# on the given images(Gray scaled images or Smoothened Gray scaled images)
def generate_masks(images, operator):
    output_images = [[], [], []]
    mask = []
    #applying differential operator to the gray scaled images
    #output is saved in output_images
    #output_images[0] contains images multiplied with differential_operator[0] = -1/2
    #output_images[1] contains images multiplied with differential_operator[1] = 0
    #output_images[2] contains images multiplied with differential_operator[2] = 1/2
    for i in range(len(operator)):
        for j in range(len(images)):
            output_images[i].append(operator[i] * images[j])
    
    #mask[0] = output_images[0][0] + output_images[1][1]  output_images[1][2]
#     num_ops = len(output_images)
#     for j in range(len(output_images[0])-(num_ops-1)):
#         mask.append(output_images[0][j] + output_images[1][j+1] + output_images[2][j+2])

    num_ops = len(output_images)

    for j in range(len(output_images[0])-(num_ops-1)):
        sm = 0
        for i in range(num_ops):
            sm += output_images[i][j+i]
        mask.append(sm)
    mask = np.array(mask)
    
    return mask

# Applying the threshold on the given masks
# mask values would be 1 if abs(mask) > threshold else 0
def generate_thresholded_masks(mask, threshold):
    mask = 1*(abs(mask) > threshold)
    return mask

# Applying the thresholded mask on the given 
# Images (Gray scaled images or Smoothened Gray scaled images)  
def apply_thresholded_mask(images, mask, operator):
    #applying the mask to the images
    #mask[0] applied to images[0], images[1], images[2]
    #mask[1] applied to images[1], images[2], images[3]
    result = []
    for i in range(len(mask)):
        for j in range(i, i + len(operator)):
            result.append(mask[i] * images[j])
    
    result = np.array(result)
#     result_1 = []
#     for i in range(0, len(result)-2, 3):
#         result_1.append(result[0+i] + result[1+i] + result[2+i])
    
#     result_1 = np.array(result_1)
#     print('size of result is : ')
#     print(np.shape(result))
#     print(np.shape(result_1))
    return result

# Function to generate 1D derivative of Gaussian with given sigma
def gaussian_derivative(x, sigma):
    A = 1 / (sigma * np.sqrt(2 * np.pi))
    return -A * x * np.exp(-(x**2) / (2 * sigma**2))

def box_filter(images, size):
    images = np.array(images)
    blur_images = cv.blur(images, (size, size))
    return blur_images

def generate_2D_gaussian_filter(ksize, sigma):
    # Create a 2D Gaussian filter
    x, y = np.meshgrid(np.linspace(-1, 1, ksize), np.linspace(-1, 1, ksize))
    d = np.sqrt(x*x + y*y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    # Normalize the filter
    g = g / g.sum()
    return g

def apply_gaussian_2D_filter(images, gaussian_filter):
    images = np.array(images)
    smoothened_images = []
    # Apply the Gaussian filter to each image
    for image in images:
        smoothened_image = cv.filter2D(image, -1, gaussian_filter)
        smoothened_images.append(smoothened_image)
    smoothened_images = np.array(smoothened_images)
    return smoothened_images

def display(image1, image2, waitTime, name):
    display_images = np.concatenate((image1, image2), axis = 2)
#     cv.imshow(name, display_images[140])
#     cv.waitKey(0)
#     cv.destroyAllWindows()
    for i in range(len(display_images)):
        if i % 10 == 0:
            cv.imshow(name, display_images[i])
            cv.waitKey(waitTime)
    cv.destroyAllWindows()

# Function to apply 1D differential Operator on the given 
# Images (Gray scaled images or Smoothened Gray scaled images)
def apply_1D_differential_operator(images, differential_operator, threshold):
    # Applying 1D differential operator on the given images
    differential_operator_mask = generate_masks(images, differential_operator)
    # Applying threshold to the generated masks
    differential_operator_mask_t = generate_thresholded_masks(differential_operator_mask, threshold)
    # Applying the thresholded masks to the images (result)
    differential_operator_result = apply_thresholded_mask(images, differential_operator_mask_t, differential_operator)
    # Converting the type to uint8
    differential_operator_mask = differential_operator_mask.astype(np.uint8)
    differential_operator_mask_t = differential_operator_mask_t.astype(np.uint8)
    differential_operator_result = differential_operator_result.astype(np.uint8)
    return differential_operator_result, differential_operator_mask

def apply_1D_gaussian_operator(images, gaussian_1D_operator, threshold):
    # Applying 1D Gaussian operator on the given images
    gaussian_1D_mask = generate_masks(images, gaussian_1D_operator)
    # Applying threshold to the generated masks
    gaussian_1D_mask_t = generate_thresholded_masks(gaussian_1D_mask,threshold)
    # Applying the thresholded masks to the images (result)
    gaussian_result = apply_thresholded_mask(images, gaussian_1D_mask_t, gaussian_1D_operator)
    # Converting the type to uint8
    gaussian_1D_mask = gaussian_1D_mask.astype(np.uint8)
    gaussian_1D_mask_t = gaussian_1D_mask_t.astype(np.uint8)
    gaussian_result = gaussian_result.astype(np.uint8)
    return gaussian_result, gaussian_1D_mask

def std_deviation(images, title):
    #standard Deviation
    std_dev = []
    std = []
    for i in images:
        std_dev.append(np.std(i))  
    x = list(range(0, 100, 1))
    std = std_dev[0:100]
    
    #plotting std
    plt.title(title)
    plt.xlabel("Image Number")
    plt.ylabel("Standard Deviation")
    plt.plot(x, std, 'o', color ="red")
    plt.show()


images = []
gray_scaled_images = []
threshold = float(input("Enter the Threshold value : "))
images, n = read_input_images()
print(f'No of Images in the given path is {n}')
gray_scaled_images = covert_to_grayScale(images)

# #display(images, np.zeros(np.shape(images)), 50, 'Original Images and Gray Scaled Images')
# for i in images:
#     cv.imshow('Input Images', i)
#     cv.waitKey(50)
# cv.destroyAllWindows()

# for i in gray_scaled_images:
#     cv.imshow('Gray Images', i)
#     cv.waitKey(50)
# cv.destroyAllWindows()

sigma = float(input('Enter the sigma Value : '))
x = np.linspace(-sigma, sigma, 3)
differential_operator = [-0.5, 0, 0.5]
gaussian_1D_operator = gaussian_derivative(x, sigma)

differential_operator_result, differential_operator_mask = apply_1D_differential_operator(gray_scaled_images, differential_operator, threshold)
gaussian_result, gaussian_1D_mask = apply_1D_gaussian_operator(gray_scaled_images, gaussian_1D_operator, threshold)

display(differential_operator_mask, gaussian_1D_mask, 50, 'Differential Mask, Gaussian Mask (No Smoothening)')
std_deviation(differential_operator_result, "differential_operator_result without smoothening")

display(differential_operator_result, gaussian_result, 50, 'Differential Result, Gaussian Result (No Smoothening)')
std_deviation(gaussian_result, "gaussian_result without smoothening")

# nr = []
# # concatenate image Horizontally
# for i in range(len(differential_operator_result)):
#     if (i % 3 == 0):
#        nr.append(differential_operator_result[i])

# nr_t = []
# for i in range(len(gaussian_result)):
#     if (i % 3 == 0):
#        nr_t.append(gaussian_result[i])

choice = input("Do you want to smoothen the input Images : \n Enter Y or N : ")

while (choice == 'Y' or choice == 'y'):
    c = int(input('Choose 1 for Box Smoothening \n Choose 2 for 2D Gaussian Smoothening : '))
    
    if (c == 1):
        size = int(input('Enter the Size of box filter for Smoothening : '))
        smoothened_images = box_filter(gray_scaled_images, size)
        differential_operator_result, differential_operator_mask = apply_1D_differential_operator(smoothened_images, differential_operator, threshold)
        gaussian_result, gaussian_1D_mask = apply_1D_gaussian_operator(smoothened_images, gaussian_1D_operator, threshold)
        display(differential_operator_mask, gaussian_1D_mask, 50, 'Differential Mask, Gaussian Mask (Smoothening)')
        std_deviation(differential_operator_result, "differential_operator_result with box smoothening")
        display(differential_operator_result, gaussian_result, 50, 'Differential Result, Gaussian Result (Smoothening)')
        std_deviation(gaussian_result, "gaussian_result with box smoothening")

    elif (c == 2):
        ksize = int(input("Enter the 2D Gaussian Filter Size : "))
        ksigma = float(input("Enter the standard deviation of 2D Gaussian Filter : "))
        gaussian_filter = generate_2D_gaussian_filter(ksize, ksigma)
        smoothened_images = apply_gaussian_2D_filter(gray_scaled_images, gaussian_filter)
        display(smoothened_images, gray_scaled_images, 50, 'Smoothened Images, Gray Scaled Images (Smoothening)')
        differential_operator_result, differential_operator_mask = apply_1D_differential_operator(smoothened_images, differential_operator, threshold)
        gaussian_result, gaussian_1D_mask = apply_1D_gaussian_operator(smoothened_images, gaussian_1D_operator, threshold)
        display(differential_operator_mask, gaussian_1D_mask, 50, 'Differential Mask, Gaussian Mask (Smoothening)')
        std_deviation(differential_operator_result, "differential_operator_result with Gaussian smoothening")
        display(differential_operator_result, gaussian_result, 50, 'Differential Result, Gaussian Result (Smoothening)')
        std_deviation(gaussian_result, "gaussian_result with Gaussian smoothening")

    choice = input("Do you want to smoothen the input Images : \n Enter Y or N : ")

print('Thank You!')


# In[ ]:




