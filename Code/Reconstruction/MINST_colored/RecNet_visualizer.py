import matplotlib.pyplot as plt
import numpy as np
from model_trainer import preprocess_data

def visualize(obf_images, rec_images, names):
    N = len(obf_images)
    fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(8, 2 * N))

    for i in range(N):
        name = names[i]
        axes[i, 0].imshow(obf_images[i])
        axes[i, 0].set_title(name[0])
        axes[i, 0].axis('off')

        axes[i, 1].imshow(rec_images[i])
        axes[i, 1].set_title(name[1])
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_recnets(obf_models, rec_models, data, names, verbose=0):
    _, x_test, _, _ = preprocess_data(data['Obf'])  # We only need testing images
    indices = np.random.randint(0, x_test.shape[0], 10)
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
    plt.imshow(orig_imgs)
    plt.axis('off')
    plt.title('Input images')
    plt.show(block=False)
    visualize(obf_imgs, rec_imgs, names)

