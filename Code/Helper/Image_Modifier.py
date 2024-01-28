import cv2 as cv
import os


def resize_image(orig_path, new_path, target_size):
    """
    Resize an image and save the resized version to a new file.

    Parameters:
    - orig_path (str): The path to the original image file.
    - new_path (str): The path to save the resized image.
    - target_size (tuple): A tuple (width, height) specifying the target size for the resized image.

    Returns:
    None
    """
    img = cv.imread(orig_path)
    img = cv.resize(img, target_size)
    cv.imwrite(new_path, img)

def resize_images_in_dir(orig_dir, new_dir, target_size, progress=True):
    """
    Resize images in a directory and save the resized versions to a new directory.

    Parameters:
    - orig_dir (str): The path to the original directory containing images.
    - new_dir (str): The path to save the resized images.
    - target_size (tuple): A tuple (width, height) specifying the target size for the resized images.
    - progress (bool, optional): If True, print progress messages for each resized image. Default is True.

    Returns:
    None
    """
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

if __name__ == "__main__":
    path = os.path.abspath('D:\ImageNet\Original_Dataset\\ILSVRC\Data\CLS-LOC\\test') + '/'
    new_path = os.path.abspath('D:\ImageNet\Original_Dataset\\ILSVRC\Data\CLS-LOC\\res_test') + '/'
    resize_images_in_dir(path, new_path, (224, 224), progress=True)