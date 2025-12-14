"""OCR Engine component for text extraction from images."""

import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract
from numpy.typing import NDArray

from sequence_converter.models import OCRResult

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR Engine component using Tesseract for text extraction.

    This component extracts text regions from images with bounding boxes
    and confidence scores, and can also detect text colors (red/blue).
    """

    # Color thresholds for text color detection
    RED_THRESHOLD = 150  # Minimum R value for red text
    BLUE_THRESHOLD = 150  # Minimum B value for blue text
    OTHER_COLOR_MAX = 100  # Maximum value for other color channels

    def __init__(self, config: str = "--oem 3 --psm 6"):
        """Initialize OCREngine with Tesseract configuration.

        Args:
            config: Tesseract configuration string.
                Default: "--oem 3 --psm 6"
                - oem 3: Default OCR Engine Mode (LSTM-based)
                - psm 6: Assume a single uniform block of text
        """
        self.config = config
        logger.info(f"OCREngine initialized with config: {config}")

    def extract_text_regions(self, image: NDArray[np.uint8]) -> list[OCRResult]:
        """Extract all text regions from an image with bounding boxes.

        Uses pytesseract's image_to_data to get word-level bounding boxes
        and confidence scores.

        Args:
            image: Input image (grayscale or color, uint8)

        Returns:
            list[OCRResult]: List of extracted text regions with metadata
        """
        logger.debug(f"Extracting text regions from image of shape {image.shape}")

        try:
            # Use image_to_data to get detailed OCR results
            ocr_data = pytesseract.image_to_data(
                image, config=self.config, output_type=pytesseract.Output.DICT
            )

            results = []
            n_boxes = len(ocr_data["text"])

            for i in range(n_boxes):
                text = ocr_data["text"][i].strip()
                confidence = float(ocr_data["conf"][i])

                # Skip empty text or invalid confidence
                if not text or confidence < 0:
                    continue

                # Extract bounding box coordinates
                left = int(ocr_data["left"][i])
                top = int(ocr_data["top"][i])
                width = int(ocr_data["width"][i])
                height = int(ocr_data["height"][i])
                bbox = (left, top, width, height)

                # Detect text color if image is color (3 channels)
                color = None
                if len(image.shape) == 3 and image.shape[2] == 3:
                    color = self._detect_text_color(image, bbox)

                # Log warning for low confidence results
                if confidence < 60:
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

            # Use pytesseract to extract text from ROI
            text = pytesseract.image_to_string(roi, config=self.config).strip()

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
