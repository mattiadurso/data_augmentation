# This is Data Augmentation Library for images developed in Python and OpenCV

# TODO
# - Shearing the image


# SUCCESFULLY IMPLEMENTED:
# - Resizing and rotation in the same direction all images
# - Random color modification
# - Blurring images
# - Flipping images vertically/horizontally/mirror
# - Erasing part of the image randomly
# - Salt and Pepper adding
# - Change brigthness and contrast
# - Random rotation of he images
# - Edge Detection
# - Zoom in
# - Zoom out

import os
import cv2 as cv
import glob
import numpy as np
import random
from time import perf_counter_ns as tm
from copy import deepcopy
from scipy.ndimage import convolve, interpolation


def displayImg(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

# Load all .jpg images
#"Dataset/*.jpg"
def load_images(path):
    path = glob.glob(path)
    images = []
    for img in path:
        n = cv.imread(img)
        images.append(n)
    print("Loaded.")
    return images


# Resize to equal dimensions
def resize_images(images):
    rotate_images = deepcopy(images)
    for idx, _ in enumerate(rotate_images):
        #check the orientation
        w, h = _.shape[0], _.shape[1]
        if h <= w:
            _ = cv.rotate(_, cv.ROTATE_90_CLOCKWISE)
        _ = cv.resize(_, (1500, 1000))
        rotate_images[idx] = _
        cv.imwrite(f"Augmented/original_{idx}.jpg",_)
    print("Resized.")
    return rotate_images


# color modification
def color_modification(images):
    colorMod_images = deepcopy(images)
    for idx, _ in enumerate(colorMod_images):
        _ = (_ + random.randint(0,255))//2
        colorMod_images[idx] = _
        cv.imwrite(f"Augmented/colormod_{idx}.jpg",_)
    print("ColorModified.")
    return colorMod_images


# blurred images
def blurr_images(images):
    blurred_images = deepcopy(images)
    for idx, _ in enumerate(blurred_images):
        _ = cv.blur(_, (10,10))
        blurred_images[idx] = _
        cv.imwrite(f"Augmented/blurred_{idx}.jpg",_)
    print("Blurred.")
    return blurred_images


# flipping
def flip_images(images):
    flipped_images = deepcopy(images)
    counter = 0
    for idx, _ in enumerate(flipped_images):
        a = cv.flip(_, 0)
        b = cv.flip(_, 1)
        c = cv.flip(_, -1)
        counter += 1
        cv.imwrite(f"Augmented/flipped_a{counter}.jpg",a)
        cv.imwrite(f"Augmented/flipped_b{counter}.jpg",b)
        cv.imwrite(f"Augmented/flipped_c{counter}.jpg",c)
    print("Flipped.")
    return flipped_images


# random erasing
def random_erasing(images):
    images_erasing = deepcopy(images)
    for idx,_ in enumerate(images_erasing):
        x1 = random.randint( _.shape[1]*0.25, _.shape[1]*0.5)
        x2 = random.randint(x1, _.shape[1]*0.75)
        y1 = random.randint( _.shape[0]*0.25, _.shape[0]*0.5)
        y2 = random.randint(y1, _.shape[0]*0.75)

        images_erasing[idx] = cv.rectangle(_, (x1, y1), (x2, y2), (0,0,0), thickness = -1)
        cv.imwrite(f"Augmented/random_erased_{idx}.jpg",_)

    print("Erased.")
    return images_erasing
    

# add random noise:
def add_noise(images):
    salted_images = deepcopy(images)
    for idx,_ in enumerate(salted_images):
        
        row, col = _.shape[:2]
        number_of_pixels = random.randint(2000, 20000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            _[y_coord][x_coord] = 255
            
        number_of_pixels = random.randint(2000, 20000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            _[y_coord][x_coord] = 0        

        salted_images[idx] = _
        cv.imwrite(f"Augmented/salted_{idx}.jpg",_)

    print("Salt and pepper added.")
    return salted_images


# change brihtness
def brigth_and_contrast(images, alpha = 1.5, beta = 0, gamma = 50):
    #alpha = 1.5   #weigth of first source. 0: black, 1: original, 2: very brigth 
    #beta = 0    #weigth of second source
    #gamma = 50   #scalar added -> contrast
    ##(1-alpha)*img + (1-beta)*alpha + gamma
    brigth_and_contrast_images = deepcopy(images)
    for idx, _ in enumerate(brigth_and_contrast_images):

        _ = cv.addWeighted(_, alpha, np.zeros(_.shape, _.dtype), beta, -gamma) 
        brigth_and_contrast_images[idx] = _
        cv.imwrite(f"Augmented/brigth_and_contrast{idx}.jpg",_)
    print("Brighted and contrasted.")
    return brigth_and_contrast_images


# random rotated
def random_rotation(images, iterations=1):
    random_rotated_images = deepcopy(images)

    for i in range(1,iterations):
        for idx,_ in enumerate(random_rotated_images):
            angle = int(random.uniform(-180, 180))
            h, w = _.shape[:2]
            matrix = cv.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
            _ = cv.warpAffine(_, matrix, (w, h))
            random_rotated_images[idx] = _
            
            cv.imwrite(f"Augmented/random_rotated_{i*10+idx}.jpg",_)
    print("Random rotated.")
    return random_rotated_images


# edge detection
def detect_edges(images):
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ], float)

    edges_images = deepcopy(images)
    for idx,_ in enumerate(edges_images):
        # Convert to BW
        image_bw = cv.cvtColor(_, cv.COLOR_RGB2GRAY).astype(float)

        # Apply convolution
        image_h = convolve(image_bw, kernel)
        image_v = convolve(image_bw, kernel.T)

        # Combine edges
        image_hv2 = np.sqrt(np.square(image_h) + np.square(image_v))

        image_hv2 = cv.cvtColor(image_hv2.astype(np.uint8), cv.COLOR_GRAY2RGB)

        edges_images[idx] = image_hv2
        cv.imwrite(f"Augmented/edged_{idx}.jpg",image_hv2)
    print("Edged.")
    return detect_edges

# zoom in
def zoom_in(images):
    zoomed_in_images = deepcopy(images)
    for idx,_ in enumerate(zoomed_in_images):
        _ = _[300:1200, 200:800]
        _ = cv.resize(_, (1500, 1000))
        cv.imwrite(f"Augmented/zoomed_in_images{idx}.jpg",_)
        zoomed_in_images[idx] = _
    print("Zoomed in.")
    return zoomed_in_images

# zoom out
def zoom_out(images):
    zoomed_out_images = deepcopy(images)
    for idx,_ in enumerate(zoomed_out_images):
        _ = cv.resize(_, (1050, 700))
        _ = cv.copyMakeBorder(_,150,150,225,225, cv.BORDER_CONSTANT)
        cv.imwrite(f"Augmented/zoomed_out_images{idx}.jpg",_)
        zoomed_out_images[idx] = _
    print("Zoomed out.")
    return zoomed_out_images






#
#       |\  /| /\  | |\ |
#       | \/ |/--\ | | \|
# 


if __name__ == "__main__":
    os.system("clear")
    # Mount the rigth directory
    #os.system("cd /OpenCV/Project_1/")
    if os.path.exists('Augmented') == False:
        os.mkdir("Augmented")


    print("This library augements your dataset by 12 times.")
    print("Insert your dataset's path:")

    #   /Users/mattiadurso/Desktop/python Projects/OpenCV/Project_1/Dataset/*.jpg
    path = input()+"/*.jpg"

    print()
    start = tm()

    bool = True
    if bool == True:
        images = resize_images(load_images(path))
        color_modification(images)
        blurr_images(images)
        flip_images(images)
        brigth_and_contrast(images)
        random_rotation(images)
        random_erasing(images)
        add_noise(images)
        detect_edges(images)
        zoom_in(images)
        zoom_out(images)

    end = tm()
    print()
 
    print(f"Augmented in {round((end-start)/1_000_000_000, 2)}s")
    print()
    print()



