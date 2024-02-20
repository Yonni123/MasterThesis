import matplotlib.pyplot as plt
import numpy as np
from model_trainer import preprocess_data
from PIL import Image

def save_image(image_array, path):
    # Ensure pixel values are in the range [0, 255]
    image_array = (image_array * 255).astype(np.uint8)

    # Create a PIL Image object from the numpy array
    image = Image.fromarray(image_array)

    # Save the image to the specified path
    image.save(path)

def visualize(obf_images, rec_images, orig_imgs, names, save=False):
    if save:
        save_image(orig_imgs, path='colored_MNIST_example.png')
        for obf, rec, bn in zip(obf_images, rec_images, [8, 16, 32, 64, 128, 256, 512, 1024]):
            save_image(obf, path=f'Obf/{bn}.png')
            save_image(rec, path=f'Rec/{bn}.png')

    N = len(obf_images)
    fig, axes = plt.subplots(nrows=N+1, ncols=2, figsize=(8, 2 * (N+1)))
    black_img = np.zeros(shape=(28, 280, 3))

    axes[0, 0].imshow(orig_imgs)
    axes[0, 0].set_title('Original images')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(black_img)
    axes[0, 1].set_title('Black image')
    axes[0, 1].axis('off')

    for i in range(N):
        name = names[i]
        axes[i+1, 0].imshow(obf_images[i])
        axes[i+1, 0].set_title(name[0])
        axes[i+1, 0].axis('off')

        axes[i+1, 1].imshow(rec_images[i])
        axes[i+1, 1].set_title(name[1])
        axes[i+1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_recnets(obf_models, rec_models, data, names, verbose=0):
    _, x_test, _, _, _, _ = preprocess_data(data['Obf'])  # We only need testing images
    indices = np.array([117, 97, 23, 64, 8, 43, 111, 120, 5, 33])
    test_imgs = x_test[indices]
    obf_imgs = []
    rec_imgs = []
    for obf, rec in zip(obf_models, rec_models):
        obfuscated = obf.predict(test_imgs, verbose=verbose)
        reconstructed = rec.predict(obfuscated, verbose=verbose)
        max_values = np.amax(obfuscated, axis=(1, 2, 3), keepdims=True)
        obfuscated /= max_values    # We need to normalize the pixel values, so they range from [0-1]
        max_values = np.amax(reconstructed, axis=(1, 2, 3), keepdims=True)
        reconstructed /= max_values
        obfuscated = np.concatenate(np.array(obfuscated), axis=1)
        reconstructed = np.concatenate(np.array(reconstructed), axis=1)
        obf_imgs.append(obfuscated)
        rec_imgs.append(reconstructed)
    orig_imgs = np.concatenate(np.array(test_imgs), axis=1)
    visualize(obf_imgs, rec_imgs, orig_imgs, names)

