"""Image preprocessing module."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .logger import ImageProcessingError, UnsupportedFormatError, setup_logger
from .models import PreprocessedImage, PreprocessingConfig

logger = setup_logger(__name__)


class ImagePreprocessor:
    """画像前処理コンポーネント"""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        """
        Args:
            config: 前処理設定（デフォルト値で初期化）
        """
        self.config = config

    def preprocess(self, image_path: str) -> PreprocessedImage:
        """
        画像を読み込み、グレースケール・二値化・ノイズ除去を実行

        Args:
            image_path: 入力画像ファイルパス（PNG/JPG）

        Returns:
            PreprocessedImage: 前処理済み画像データ

        Raises:
            FileNotFoundError: ファイルが存在しない
            UnsupportedFormatError: サポートされていないフォーマット
            ImageProcessingError: 画像処理が失敗
        """
        # ファイルの存在確認
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {image_path}")

        # フォーマット確認
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            # 画像読み込み（PILで読み込んでからOpenCVに変換）
            pil_image = Image.open(image_path)
            image_rgb = np.array(pil_image.convert("RGB"))
            original_shape = image_rgb.shape[:2]  # (height, width)

            # OpenCV形式に変換（RGB -> BGR）
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # グレースケール変換
            grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            logger.debug(f"Converted to grayscale: shape={grayscale.shape}")

            # ノイズ除去
            if self.config.noise_removal_method == "median":
                denoised = cv2.medianBlur(grayscale, self.config.blur_kernel_size)
            elif self.config.noise_removal_method == "gaussian":
                denoised = cv2.GaussianBlur(
                    grayscale, (self.config.blur_kernel_size, self.config.blur_kernel_size), 0
                )
            else:
                denoised = grayscale

            logger.debug(f"Applied {self.config.noise_removal_method} blur")

            # 二値化
            if self.config.binary_threshold_method == "otsu":
                _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.config.binary_threshold_method == "adaptive":
                binary = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            else:
                _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)

            logger.debug(f"Applied {self.config.binary_threshold_method} thresholding")

            logger.info(f"Preprocessing completed for {image_path}")

            return PreprocessedImage(
                grayscale=grayscale, binary=binary, original_shape=original_shape
            )

        except Exception as e:
            if isinstance(e, (FileNotFoundError, UnsupportedFormatError)):
                raise
            raise ImageProcessingError(f"Failed to process image: {e}") from e
