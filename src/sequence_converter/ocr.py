"""OCR Engine component for text extraction from images."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from sequence_converter.models import OCREngineType, OCRResult

logger = logging.getLogger(__name__)


# ============================================================================
# OCR Backend Abstract Base Class
# ============================================================================


class OCRBackend(ABC):
    """OCRバックエンドの抽象基底クラス"""

    @abstractmethod
    def extract_text_data(self, image: NDArray[np.uint8]) -> list[dict[str, any]]:
        """画像からテキストデータを抽出する（バウンディングボックスと信頼度を含む）

        Args:
            image: 入力画像 (grayscale or color, uint8)

        Returns:
            list[dict]: テキストデータのリスト
                各要素は以下のキーを持つ:
                - text: str
                - left: int
                - top: int
                - width: int
                - height: int
                - confidence: float
        """
        pass

    @abstractmethod
    def extract_text_from_roi(self, roi: NDArray[np.uint8]) -> str:
        """ROI（Region of Interest）からテキストを抽出する

        Args:
            roi: 入力ROI画像

        Returns:
            str: 抽出されたテキスト
        """
        pass


# ============================================================================
# Tesseract Backend Implementation
# ============================================================================


class TesseractBackend(OCRBackend):
    """Tesseract OCRバックエンド実装"""

    def __init__(self, config: str = "--oem 3 --psm 6"):
        """Initialize TesseractBackend with configuration.

        Args:
            config: Tesseract configuration string.
                Default: "--oem 3 --psm 6"
                - oem 3: Default OCR Engine Mode (LSTM-based)
                - psm 6: Assume a single uniform block of text
        """
        self.config = config
        logger.info(f"TesseractBackend initialized with config: {config}")

    def extract_text_data(self, image: NDArray[np.uint8]) -> list[dict[str, any]]:
        """画像からテキストデータを抽出する（Tesseract使用）"""
        import pytesseract

        try:
            ocr_data = pytesseract.image_to_data(
                image, config=self.config, output_type=pytesseract.Output.DICT
            )

            results = []
            n_boxes = len(ocr_data["text"])

            for i in range(n_boxes):
                text = ocr_data["text"][i].strip()
                confidence = float(ocr_data["conf"][i])

                if not text or confidence < 0:
                    continue

                results.append(
                    {
                        "text": text,
                        "left": int(ocr_data["left"][i]),
                        "top": int(ocr_data["top"][i]),
                        "width": int(ocr_data["width"][i]),
                        "height": int(ocr_data["height"][i]),
                        "confidence": confidence,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error during Tesseract OCR: {e}")
            return []

    def extract_text_from_roi(self, roi: NDArray[np.uint8]) -> str:
        """ROIからテキストを抽出する（Tesseract使用）"""
        import pytesseract

        try:
            text = pytesseract.image_to_string(roi, config=self.config).strip()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from ROI with Tesseract: {e}")
            return ""


# ============================================================================
# EasyOCR Backend Implementation
# ============================================================================


class EasyOCRBackend(OCRBackend):
    """EasyOCR バックエンド実装"""

    def __init__(self, languages: list[str] = None, gpu: bool = False):
        """Initialize EasyOCRBackend with configuration.

        Args:
            languages: List of language codes (e.g., ['ja', 'en']). Default: ['ja', 'en']
            gpu: Whether to use GPU acceleration. Default: False
        """
        if languages is None:
            languages = ["ja", "en"]

        self.languages = languages
        self.gpu = gpu
        self._reader = None  # Lazy initialization
        logger.info(f"EasyOCRBackend initialized with languages: {languages}, GPU: {gpu}")

    def _get_reader(self):
        """EasyOCR Readerのレイジー初期化"""
        if self._reader is None:
            import easyocr

            logger.info("Initializing EasyOCR Reader (this may take a moment)...")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR Reader initialized successfully")
        return self._reader

    def extract_text_data(self, image: NDArray[np.uint8]) -> list[dict[str, any]]:
        """画像からテキストデータを抽出する（EasyOCR使用）"""
        try:
            reader = self._get_reader()

            # EasyOCRでテキスト検出
            # result format: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
            ocr_results = reader.readtext(image)

            results = []
            for bbox_coords, text, confidence in ocr_results:
                text = text.strip()
                if not text:
                    continue

                # バウンディングボックスの座標を計算
                # bbox_coords: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [coord[0] for coord in bbox_coords]
                y_coords = [coord[1] for coord in bbox_coords]

                left = int(min(x_coords))
                top = int(min(y_coords))
                width = int(max(x_coords) - left)
                height = int(max(y_coords) - top)

                # EasyOCRの信頼度は0-1なので、0-100にスケーリング
                confidence_percent = confidence * 100.0

                results.append(
                    {
                        "text": text,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        "confidence": confidence_percent,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error during EasyOCR text extraction: {e}")
            return []

    def extract_text_from_roi(self, roi: NDArray[np.uint8]) -> str:
        """ROIからテキストを抽出する（EasyOCR使用）"""
        try:
            reader = self._get_reader()
            ocr_results = reader.readtext(roi)

            # すべてのテキストを結合
            texts = [text for _, text, _ in ocr_results]
            return " ".join(texts).strip()

        except Exception as e:
            logger.error(f"Error extracting text from ROI with EasyOCR: {e}")
            return ""


# ============================================================================
# OCR Engine (Facade)
# ============================================================================


class OCREngine:
    """OCR Engine component for text extraction from images.

    This component extracts text regions from images with bounding boxes
    and confidence scores, and can also detect text colors (red/blue).
    Supports multiple OCR backends: Tesseract and EasyOCR.
    """

    # Color thresholds for text color detection
    RED_THRESHOLD = 150  # Minimum R value for red text
    BLUE_THRESHOLD = 150  # Minimum B value for blue text
    OTHER_COLOR_MAX = 100  # Maximum value for other color channels

    # OCR confidence thresholds
    MIN_CONFIDENCE = 30  # Minimum confidence to accept OCR result
    LOW_CONFIDENCE_THRESHOLD = 60  # Threshold for low confidence warning

    def __init__(
        self,
        engine_type: OCREngineType = OCREngineType.TESSERACT,
        tesseract_config: str = "--oem 3 --psm 6",
        easyocr_languages: list[str] = None,
        easyocr_gpu: bool = False,
    ):
        """Initialize OCREngine with specified backend.

        Args:
            engine_type: OCR engine type (tesseract or easyocr)
            tesseract_config: Tesseract configuration string (used if engine_type is TESSERACT)
                Default: "--oem 3 --psm 6"
                - oem 3: Default OCR Engine Mode (LSTM-based)
                - psm 6: Assume a single uniform block of text
            easyocr_languages: List of language codes for EasyOCR (used if engine_type is EASYOCR)
                Default: ['ja', 'en']
            easyocr_gpu: Whether to use GPU for EasyOCR (used if engine_type is EASYOCR)
                Default: False
        """
        self.engine_type = engine_type

        # バックエンドを初期化
        if engine_type == OCREngineType.TESSERACT:
            self.backend = TesseractBackend(config=tesseract_config)
        elif engine_type == OCREngineType.EASYOCR:
            self.backend = EasyOCRBackend(languages=easyocr_languages, gpu=easyocr_gpu)
        else:
            raise ValueError(f"Unsupported OCR engine type: {engine_type}")

        logger.info(f"OCREngine initialized with backend: {engine_type.value}")

    def extract_text_regions(self, image: NDArray[np.uint8]) -> list[OCRResult]:
        """Extract all text regions from an image with bounding boxes.

        Uses the configured OCR backend to get word-level bounding boxes
        and confidence scores.

        Args:
            image: Input image (grayscale or color, uint8)

        Returns:
            list[OCRResult]: List of extracted text regions with metadata
        """
        logger.debug(f"Extracting text regions from image of shape {image.shape}")

        try:
            # バックエンドからテキストデータを取得
            text_data_list = self.backend.extract_text_data(image)

            results = []
            for text_data in text_data_list:
                text = text_data["text"]
                confidence = text_data["confidence"]

                # Skip low confidence results to reduce false positives
                if confidence < self.MIN_CONFIDENCE:
                    logger.debug(
                        f"Skipping text '{text}' with confidence {confidence:.1f}% "
                        f"(below minimum threshold {self.MIN_CONFIDENCE}%)"
                    )
                    continue

                # Extract bounding box coordinates
                left = text_data["left"]
                top = text_data["top"]
                width = text_data["width"]
                height = text_data["height"]
                bbox = (left, top, width, height)

                # Detect text color if image is color (3 channels)
                color = None
                if len(image.shape) == 3 and image.shape[2] == 3:
                    color = self._detect_text_color(image, bbox)

                # Log warning for low confidence results
                if confidence < self.LOW_CONFIDENCE_THRESHOLD:
                    logger.warning(
                        f"Low OCR confidence ({confidence:.1f}%) for text '{text}' "
                        f"at bbox {bbox}"
                    )

                results.append(
                    OCRResult(
                        text=text,
                        bounding_box=bbox,
                        confidence=confidence,
                        color=color,
                    )
                )

            logger.info(f"Extracted {len(results)} text regions from image")
            return results

        except Exception as e:
            logger.error(f"Error during OCR text extraction: {e}")
            return []

    def extract_from_region(self, image: NDArray[np.uint8], bbox: tuple[int, int, int, int]) -> str:
        """Extract text from a specific region of the image.

        Args:
            image: Input image (grayscale or color, uint8)
            bbox: Bounding box (left, top, width, height)

        Returns:
            str: Extracted text from the region
        """
        left, top, width, height = bbox

        # Extract the region of interest
        try:
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            left = max(0, min(left, w - 1))
            top = max(0, min(top, h - 1))
            right = min(left + width, w)
            bottom = min(top + height, h)

            if right <= left or bottom <= top:
                logger.warning(f"Invalid bbox {bbox} for image shape {image.shape}")
                return ""

            roi = image[top:bottom, left:right]

            # バックエンドを使用してROIからテキストを抽出
            text = self.backend.extract_text_from_roi(roi)

            logger.debug(f"Extracted text from region {bbox}: '{text}'")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from region {bbox}: {e}")
            return ""

    def _detect_text_color(
        self, image: NDArray[np.uint8], bbox: tuple[int, int, int, int]
    ) -> Optional[str]:
        """Detect text color (red/blue) from the bounding box region.

        Args:
            image: Color image in BGR format (OpenCV default)
            bbox: Bounding box (left, top, width, height)

        Returns:
            Optional[str]: "red", "blue", or None for default color
        """
        left, top, width, height = bbox

        try:
            # Extract ROI
            h, w = image.shape[:2]
            left = max(0, min(left, w - 1))
            top = max(0, min(top, h - 1))
            right = min(left + width, w)
            bottom = min(top + height, h)

            if right <= left or bottom <= top:
                return None

            roi = image[top:bottom, left:right]

            # Calculate mean BGR values
            mean_bgr = cv2.mean(roi)[:3]  # Get B, G, R values
            b, g, r = mean_bgr

            logger.debug(f"Mean BGR values for bbox {bbox}: B={b:.1f}, G={g:.1f}, R={r:.1f}")

            # Detect red text: high R, low G and B
            if r > self.RED_THRESHOLD and g < self.OTHER_COLOR_MAX and b < self.OTHER_COLOR_MAX:
                logger.debug(f"Detected red text at bbox {bbox}")
                return "red"

            # Detect blue text: high B, low G and R
            if b > self.BLUE_THRESHOLD and g < self.OTHER_COLOR_MAX and r < self.OTHER_COLOR_MAX:
                logger.debug(f"Detected blue text at bbox {bbox}")
                return "blue"

            # Default color (black, gray, etc.)
            return None

        except Exception as e:
            logger.error(f"Error detecting text color at bbox {bbox}: {e}")
            return None
