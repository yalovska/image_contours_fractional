import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження зображення
image = cv2.imread('image6.jpg', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

def fractional_1d_kernel(size=5, alpha=0.5):
    """Створення ядра дробової похідної на основі генералізованого бінома"""
    kernel = np.zeros(size, dtype=np.float32)
    kernel[0] = 1.0
    for i in range(1, size):
        kernel[i] = kernel[i - 1] * ((-alpha + i - 1) / i)
    return kernel

def compute_gradients(image, alpha=0.5, kernel_size=5):
    """Обчислення градієнтів із використанням дробових похідних"""
    kernel = fractional_1d_kernel(kernel_size, alpha)
    kernel = kernel[::-1]  # Інверсія ядра

    # Перетворюємо 1D ядро у 2D ядра для X та Y
    kernel_x = kernel.reshape(1, -1)
    kernel_y = kernel.reshape(-1, 1)

    grad_x = cv2.filter2D(image.astype(np.float32), -1, kernel_x)
    grad_y = cv2.filter2D(image.astype(np.float32), -1, kernel_y)

    return grad_x, grad_y

def compute_gradient_magnitude(grad_x, grad_y):
    return cv2.magnitude(grad_x, grad_y)

def apply_threshold(gradient_magnitude, threshold=100):
    _, contours = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    return contours.astype(np.uint8)

# Обчислення
grad_x, grad_y = compute_gradients(blurred_image, alpha=0.5)
gradient_magnitude = compute_gradient_magnitude(grad_x, grad_y)
contours = apply_threshold(gradient_magnitude, threshold=50)

# Візуалізація
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Gradient Magnitude')
plt.imshow(gradient_magnitude, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Contours')
plt.imshow(contours, cmap='gray')

plt.show()
