from scipy.signal import convolve2d
from imageio.v2 import imread
import numpy as np
import cv2
from align_image_code import align_images
from scipy.ndimage import gaussian_filter
import skimage.io as skio
from skimage import img_as_ubyte


def save_image(path, image):
    image = np.clip(image, -1, 1)
    skio.imsave("output/" + path + ".jpg", img_as_ubyte(image))


def gaussian_blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def difference(image_path):
    image = imread(image_path, pilmode="F") / 255.0

    gaussian_kernel = np.outer(
        cv2.getGaussianKernel(3, 1), cv2.getGaussianKernel(3, 1).T
    )

    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    # blurred_image = convolve2d(image, gaussian_kernel, mode="same", boundary="symm")

    image_dx = convolve2d(image, Dx, mode="same", boundary="symm")
    image_dy = convolve2d(image, Dy, mode="same", boundary="symm")

    gradient_magnitude = np.sqrt(image_dx**2 + image_dy**2)

    threshold = 0.2
    binary_gradient_magnitude = (
        gradient_magnitude > threshold * gradient_magnitude.max()
    )

    gaussian_dx = convolve2d(gaussian_kernel, Dx, mode="same", boundary="symm")
    gaussian_dy = convolve2d(gaussian_kernel, Dy, mode="same", boundary="symm")

    gaussian_image_dx = convolve2d(image, gaussian_dx, mode="same", boundary="symm")
    gaussian_image_dy = convolve2d(image, gaussian_dy, mode="same", boundary="symm")

    gaussian_conv = np.sqrt(gaussian_image_dx**2 + gaussian_image_dy**2)

    threshold = 0.25
    binary_gaussian_conv = gaussian_conv > threshold * gaussian_conv.max()

    return (
        image,
        image_dx,
        image_dy,
        gradient_magnitude,
        binary_gradient_magnitude,
        gaussian_dx,
        gaussian_dy,
        binary_gaussian_conv,
    )


def sharpen_image(image_path, alpha=1.2, blur=False):
    image = imread(image_path) / 255.0

    def unsharp_mask_kernel(size, sigma):
        gaussian_kernel = np.outer(
            cv2.getGaussianKernel(3, sigma), cv2.getGaussianKernel(3, sigma).T
        )

        identity = np.zeros_like(gaussian_kernel)
        identity[size // 2, size // 2] = 1

        return (1 + alpha) * identity - alpha * gaussian_kernel

    kernel = unsharp_mask_kernel(5, 1)

    if blur:
        blurred_image = gaussian_blur(image, sigma=1.5)

        r = blurred_image[:, :, 0]
        g = blurred_image[:, :, 1]
        b = blurred_image[:, :, 2]
    else:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

    r = convolve2d(r, kernel)
    g = convolve2d(g, kernel)
    b = convolve2d(b, kernel)

    sharpened_image = np.stack([r, g, b], axis=2)

    return image, blurred_image if blur else None, sharpened_image


def hybrid_image(image_path1, image_path2, pts=None, sigma1=3, sigma2=10):
    # high sf
    im1 = imread(image_path1) / 255.0

    # low sf
    im2 = imread(image_path2) / 255.0

    # crop 10px borders
    im1 = im1[10:-10, 10:-10]
    im2 = im2[10:-10, 10:-10]

    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2, pts)

    ## You will provide the code below. Sigma1 and sigma2 are arbitrary
    ## cutoff values for the high and low frequencies

    def hybrid_image(image1, image2, s1, s2):
        high_frequencies = image1 - gaussian_filter(image1, s1)
        low_frequencies = gaussian_filter(image2, s2)
        return high_frequencies + low_frequencies

    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    hybrid_grayscale = np.dot(hybrid, [0.299, 0.587, 0.114])

    log_magnitude = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid_grayscale))))

    # normalize log magnitude
    log_magnitude = (log_magnitude - log_magnitude.min()) / (
        log_magnitude.max() - log_magnitude.min()
    )

    return hybrid, log_magnitude


def get_gaussian_stack(image, N, sigma):
    stack = [image]
    for _ in range(N):
        stack.append(gaussian_filter(stack[-1], sigma))
    return stack


def get_laplacian_stack(gaussian_stack, N):
    stack = []
    for i in range(N):
        stack.append(gaussian_stack[i] - gaussian_stack[i + 1])
    stack.append(gaussian_stack[-1])
    return stack


def stack(image, N, sigma=3):
    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]

    gaussian_stack_r = get_gaussian_stack(image_r, N, sigma)
    gaussian_stack_g = get_gaussian_stack(image_g, N, sigma)
    gaussian_stack_b = get_gaussian_stack(image_b, N, sigma)

    laplacian_stack_r = get_laplacian_stack(gaussian_stack_r, N)
    laplacian_stack_g = get_laplacian_stack(gaussian_stack_g, N)
    laplacian_stack_b = get_laplacian_stack(gaussian_stack_b, N)

    laplacian_stack = [
        np.stack(
            [laplacian_stack_r[i], laplacian_stack_g[i], laplacian_stack_b[i]], axis=2
        )
        for i in range(N + 1)
    ]

    return laplacian_stack


def blend(image1, image2, d, horizontal=True):
    rows, cols = image1.shape[:2]

    blend_radius = int(cols * d) // 2

    mask = np.zeros((rows, cols, 1))

    if horizontal:
        mask[:, : cols // 2 - blend_radius] = 1
        mask[:, cols // 2 + blend_radius :] = 0

        for i in range(2 * blend_radius):
            alpha = 1 - i / (2 * blend_radius)
            mask[:, cols // 2 - blend_radius + i] = alpha
    else:
        mask[: rows // 2 - blend_radius] = 1
        mask[rows // 2 + blend_radius :] = 0

        for i in range(2 * blend_radius):
            alpha = 1 - i / (2 * blend_radius)
            mask[rows // 2 - blend_radius + i] = alpha

    blended_image = mask * image1 + (1 - mask) * image2

    return blended_image


def merge_images(image_path1, image_path2, pts=None, horizontal=True, N=5):
    im1 = imread(image_path1) / 255.0
    im2 = imread(image_path2) / 255.0

    if pts is not None:
        im1, im2 = align_images(im1, im2, pts)

    stack1 = stack(im1, N)
    stack2 = stack(im2, N)

    blank_image = np.zeros_like(im1)

    final_image = np.zeros_like(im1)
    im1_sum = np.zeros_like(im1)
    im2_sum = np.zeros_like(im2)

    normalized_im1s = []
    normalized_im2s = []
    normalized_sums = []

    for i in range(N + 1):
        im1_part = blend(stack1[i], blank_image, i * 0.05 + 0.1, horizontal)
        im2_part = blend(blank_image, stack2[i], i * 0.05 + 0.1, horizontal)
        summed_parts = im1_part + im2_part

        im1_sum += im1_part
        im2_sum += im2_part

        final_image += summed_parts

        if i != N:
            normalized_im1s.append(
                (im1_part - im1_part.min()) / (im1_part.max() - im1_part.min())
            )
            normalized_im2s.append(
                (im2_part - im2_part.min()) / (im2_part.max() - im2_part.min())
            )
            normalized_sums.append(
                (summed_parts - summed_parts.min())
                / (summed_parts.max() - summed_parts.min())
            )
        else:
            normalized_im1s.append(im1_sum)
            normalized_im2s.append(im2_sum)
            normalized_sums.append(im1_sum + im2_sum)

    return final_image, normalized_im1s, normalized_im2s, normalized_sums


def merge_images_mask(image_path1, image_path2, mask_path, pts=None, N=5):
    im1 = imread(image_path1) / 255.0
    im2 = imread(image_path2) / 255.0
    mask = imread(mask_path) / 255.0

    mask_stack = get_gaussian_stack(mask, N, 3)

    if pts is not None:
        im1, im2 = align_images(im1, im2, pts)

    stack1 = stack(im1, N)
    stack2 = stack(im2, N)

    final_image = np.zeros_like(im1)

    for i in range(N + 1):
        final_image += mask_stack[i] * stack1[i] + (1 - mask_stack[i]) * stack2[i]

    return final_image
