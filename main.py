# МЕТОДИ ВИЯВЛЕННЯ КОНТУРІВ У ЗОБРАЖЕННЯХ ІЗ ВИКОРИСТАННЯМ ДРОБОВИХ ПОХІДНИХ

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class FractionalDerivativeEdgeDetector:
    """
    Клас для виявлення контурів на зображенні з використанням дробових похідних

    Методи:
    - fractional_1d_kernel: Створює 1D ядро дробової похідної
    - compute_gradients: Обчислює градієнти за допомогою дробових похідних
    - compute_gradient_magnitude: Обчислює магнітуду градієнта
    - detect_edges: Виконує повний процес виявлення контурів
    - visualize_results: Візуалізує проміжні та кінцеві результати
    """

    def __init__(self, image_path: str):
        """
        Ініціалізація детектора контурів

        Параметри:
        - image_path: Шлях до вхідного зображення
        """
        self.image_path = image_path
        self.original_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.blurred_image: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_y: Optional[np.ndarray] = None
        self.gradient_magnitude: Optional[np.ndarray] = None
        self.contours: Optional[np.ndarray] = None

        self._load_image()

    def _load_image(self) -> None:
        """Завантажує та підготовлює зображення для обробки"""
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Не вдалося завантажити зображення за шляхом: {self.image_path}")

        self.original_image = img.copy()
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

    @staticmethod
    def fractional_1d_kernel(size: int = 5, alpha: float = 0.5) -> np.ndarray:
        """
        Створення ядра дробової похідної на основі генералізованого бінома

        Параметри:
        - size: Розмір ядра
        - alpha: Порядок дробової похідної (0 < alpha < 1)

        Повертає:
        - 1D масив NumPy, що представляє ядро дробової похідної
        """
        kernel = np.zeros(size, dtype=np.float32)
        kernel[0] = 1.0
        for i in range(1, size):
            kernel[i] = kernel[i - 1] * ((-alpha + i - 1) / i)
        return kernel

    def compute_gradients(self, alpha: float = 0.5, kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обчислення градієнтів з використанням дробових похідних

        Параметри:
        - alpha: Порядок дробової похідної
        - kernel_size: Розмір ядра фільтра

        Повертає:
        - Кортеж (grad_x, grad_y) з горизонтальною та вертикальною складовими градієнта
        """
        if self.blurred_image is None:
            raise RuntimeError("Зображення не було завантажено коректно")

        kernel = self.fractional_1d_kernel(kernel_size, alpha)[::-1]  # Інверсія ядра

        # Створення 2D ядер для обчислення градієнтів
        kernel_x = kernel.reshape(1, -1)
        kernel_y = kernel.reshape(-1, 1)

        self.grad_x = cv2.filter2D(self.blurred_image.astype(np.float32), -1, kernel_x)
        self.grad_y = cv2.filter2D(self.blurred_image.astype(np.float32), -1, kernel_y)

        return self.grad_x, self.grad_y

    def compute_gradient_magnitude(self) -> np.ndarray:
        """
        Обчислення магнітуди градієнта

        Повертає:
        - Масив магнітуд градієнта
        """
        if self.grad_x is None or self.grad_y is None:
            raise RuntimeError("Спочатку необхідно обчислити градієнти (викликати compute_gradients)")

        self.gradient_magnitude = np.sqrt(self.grad_x ** 2 + self.grad_y ** 2)
        return self.gradient_magnitude

    def detect_edges(self, threshold: float = 50) -> np.ndarray:
        """
        Виявлення контурів на зображенні

        Параметри:
        - threshold: Поріг для бінаризації магнітуди градієнта

        Повертає:
        - Бінарне зображення з виявленими контурами
        """
        if self.gradient_magnitude is None:
            self.compute_gradient_magnitude()

        _, binary = cv2.threshold(
            self.gradient_magnitude.astype(np.uint8),
            threshold,
            255,
            cv2.THRESH_BINARY
        )
        self.contours = binary
        return self.contours

    def visualize_results(self) -> None:
        """Візуалізація проміжних та кінцевих результатів"""
        if self.contours is None:
            self.detect_edges()

        plt.figure(figsize=(15, 5))

        # Вихідне зображення
        plt.subplot(1, 3, 1)
        plt.title('Вихідне зображення')
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # Магнітуда градієнта
        plt.subplot(1, 3, 2)
        plt.title('Магнітуда градієнта')
        plt.imshow(self.gradient_magnitude, cmap='gray')
        plt.axis('off')

        # Виявлені контури
        plt.subplot(1, 3, 3)
        plt.title('Виявлені контури')
        plt.imshow(self.contours, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# Приклад використання класу
if __name__ == "__main__":
    # Ініціалізація детектора з шляхом до зображення
    detector = FractionalDerivativeEdgeDetector(
        '/Images/image7.jpg'
    )

    # Обчислення градієнтів
    detector.compute_gradients(alpha=0.5, kernel_size=5)

    # Виявлення контурів
    detector.detect_edges(threshold=50)

    # Візуалізація результатів
    detector.visualize_results()
