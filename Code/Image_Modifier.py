import cv2 as cv
import os


def resize_image(orig_path, new_path, target_size):
    img = cv.imread(orig_path)
    img = cv.resize(img, target_size)
    cv.imwrite(new_path, img)

def resize_images_in_dir(orig_dir, new_dir, target_size, progress=True):
    image_formats = ['.jpg', '.png', '.jpeg', '.JPEG', '.JPG', '.PNG']

    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for filename in os.listdir(orig_dir):
        # If the file is a directory, recursively resize the images in the subdirectory
        if os.path.isdir(orig_dir + filename):
            resize_images_in_dir(orig_dir + filename + '/', new_dir + filename + '/', target_size, progress)
            continue

        # If the file is not an image, skip it
        if not any([filename.endswith(image_format) for image_format in image_formats]):
            continue

        # Resize the image
        # Convert the directory to string

        resize_image(orig_dir + filename, new_dir + filename, target_size)

        if progress:
            print(filename + ' resized!')

path = 'D:\ImageNet\Original_Dataset\\ILSVRC\Data\CLS-LOC\\test'
new_path = 'D:\ImageNet\Original_Dataset\\ILSVRC\Data\CLS-LOC\\res_test'
path = os.path.abspath(path) + '/'
new_path = os.path.abspath(new_path) + '/'
resize_images_in_dir(path, new_path, (224, 224), progress=True)