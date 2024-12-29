import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from skimage.feature import canny
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
import numpy as np


class ClassicalCVModel():

    def __init__(self):
        pass


    def predict(self, image, generate_plots=False):

        gray_image = color.rgb2gray(image)

        # Apply Gaussian filter to reduce noise
        blurred_image = filters.gaussian(gray_image, sigma=1)

        # Use edge detection (Canny)
        edges = canny(blurred_image, sigma=1.5)

        # Fill holes to make solid shapes
        filled_image = binary_fill_holes(edges)

        # Remove small objects (noise)
        cleaned_image = morphology.remove_small_objects(filled_image, min_size=100)

        # Label connected components
        labeled_image = measure.label(cleaned_image)

        # Count the number of labeled regions
        num_coins = len(np.unique(labeled_image)) - 1  # Subtract 1 for the background label

        if generate_plots:
            fig, axs = plt.subplots(1, 6, figsize=(20, 10))
            axs[0].imshow(image)
            axs[0].set_title('Original image')
            axs[0].axis('off')
            axs[1].imshow(gray_image, cmap='gray')
            axs[1].set_title('Grayscale image')
            axs[1].axis('off')
            axs[2].imshow(blurred_image, cmap='gray')
            axs[2].set_title('Blurred image')
            axs[2].axis('off')
            axs[3].imshow(edges, cmap='gray')
            axs[3].set_title('Edges detector (Canny)')
            axs[3].axis('off')
            axs[4].imshow(filled_image, cmap='gray')
            axs[4].set_title('Fill holes to make solid shapes')
            axs[4].axis('off')
            axs[5].imshow(cleaned_image, cmap='gray')
            axs[5].set_title('Remove small objects (noise)')
            axs[5].axis('off')

        return num_coins