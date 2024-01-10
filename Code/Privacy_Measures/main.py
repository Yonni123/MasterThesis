import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import HaarPSI as hp

def get_measurements(original_image, distorted_image):
    mse = mean_squared_error(original_image, distorted_image)
    haar_psi = 1-hp.haar_psi_numpy(original_image, distorted_image)[0]

    ssim_r = ssim(original_image[:, :, 0], distorted_image[:, :, 0], data_range=distorted_image.max() - distorted_image.min())
    ssim_g = ssim(original_image[:, :, 1], distorted_image[:, :, 1], data_range=distorted_image.max() - distorted_image.min())
    ssim_b = ssim(original_image[:, :, 2], distorted_image[:, :, 2], data_range=distorted_image.max() - distorted_image.min())
    ssim_value = 1-(ssim_r + ssim_g + ssim_b) / 3

    return mse, ssim_value, haar_psi


# Define file paths for the images
fish_path = '../Random_Images/fish.jpg'
fish_noise_path = '../Random_Images/fish_noise.jpg'
lion_path = '../Random_Images/lion.jpg'

# Load the images using Matplotlib's imread function
fish = np.array(imread(fish_path))
fish_noise = np.array(imread(fish_noise_path))
lion = np.array(imread(lion_path))

# Create subplots to display the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the first image (fish)
axes[0].imshow(fish)
axes[0].set_title('Fish')

# Display the second image (fish with noise)
axes[1].imshow(fish_noise)
axes[1].set_title('Fish with Noise')

# Display the third image (lion)
axes[2].imshow(lion)
axes[2].set_title('Lion')

# Hide the axes ticks
for ax in axes:
    ax.axis('off')

# Do the similarity measures
mse, ssim_value, haar_psi = get_measurements(fish, fish)
axes[0].set_title(f'MSE: {mse:.2f}, SSIM: {ssim_value:.2f} \n HaarPSI: {haar_psi:.2f}')
mse, ssim_value, haar_psi = get_measurements(fish, fish_noise)
axes[1].set_title(f'MSE: {mse:.2f}, SSIM: {ssim_value:.2f} \n HaarPSI: {haar_psi:.2f}')
mse, ssim_value, haar_psi = get_measurements(fish, lion)
axes[2].set_title(f'MSE: {mse:.2f}, SSIM: {ssim_value:.2f} \n HaarPSI: {haar_psi:.2f}')

# Show the images
plt.tight_layout()
plt.show()
