import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Завантажуємо зображення
image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

# Перетворюємо в відтінки сірого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 2. Функція для обчислення дробових похідних
def fractional_derivative_kernel(alpha, size=3):
    """Створює ядро для дробової похідної за допомогою фільтрації."""
    kernel = np.zeros((size, size), dtype=np.float32)

    # Заповнюємо ядро для дробових похідних на основі α
    for i in range(size):
        for j in range(size):
            # Використовуємо наближення дробових похідних
            kernel[i, j] = (i - size // 2) ** alpha - (j - size // 2) ** alpha

    return kernel


# 3. Обчислення градієнтів за допомогою дробових похідних
def compute_gradients(image, alpha=0.5):
    kernel = fractional_derivative_kernel(alpha)

    # Застосовуємо ядро до зображення по осях X та Y
    grad_x = cv2.filter2D(image, -1, kernel)
    grad_y = cv2.filter2D(image, -1, kernel.T)

    return grad_x, grad_y


# 4. Обчислення модуля градієнта
def compute_gradient_magnitude(grad_x, grad_y):
    return cv2.magnitude(grad_x, grad_y)


# 5. Визначення порогу для виділення контурів
def apply_threshold(gradient_magnitude, threshold=100):
    _, contours = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    return contours


# 6. Попереднє розмиття зображення для зменшення шуму
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 7. Обчислюємо градієнти
grad_x, grad_y = compute_gradients(blurred_image, alpha=0.5)

# 8. Обчислюємо модуль градієнта
gradient_magnitude = compute_gradient_magnitude(grad_x, grad_y)

# 9. Застосовуємо поріг для виділення контурів
contours = apply_threshold(gradient_magnitude, threshold=100)

# 10. Візуалізація результатів
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Fractional Derivative Gradients')
plt.imshow(gradient_magnitude, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Detected Contours')
plt.imshow(contours, cmap='gray')

plt.show()

# 11. Збереження результатів
cv2.imwrite('fractional_contours.jpg', contours)
