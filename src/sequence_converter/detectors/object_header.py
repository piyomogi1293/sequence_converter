"""Object Header Detector component for detecting participant objects in sequence diagrams."""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from sequence_converter.models import ObjectHeader, PreprocessedImage

if TYPE_CHECKING:
    from sequence_converter.ocr import OCREngine

logger = logging.getLogger(__name__)


class ObjectHeaderDetector:
    """Object Header Detector component.

    Detects rectangular object headers (participants) from the top portion
    of sequence diagram images and extracts object names using OCR.
    """

    def __init__(self, ocr_engine: "OCREngine"):
        """Initialize ObjectHeaderDetector with OCR engine.

        Args:
            ocr_engine: OCRエンジンインスタンス（依存性注入）
        """
        self.ocr_engine = ocr_engine
        logger.info("ObjectHeaderDetector initialized")

    def detect(self, preprocessed: PreprocessedImage) -> list[ObjectHeader]:
        """Detect object headers from the top portion of the image.

        Args:
            preprocessed: 前処理済み画像

        Returns:
            list[ObjectHeader]: 検出されたオブジェクトヘッダー（左から右へソート済み）
        """
        logger.info("Starting object header detection")

        # Use binary image for contour detection
        binary_image = preprocessed.binary
        h, w = binary_image.shape

        # Focus on the top 30% of the image where object headers typically appear
        top_region_height = int(h * 0.3)
        top_region = binary_image[:top_region_height, :]

        logger.debug(
            f"Searching for object headers in top region: "
            f"{top_region.shape[0]}x{top_region.shape[1]}"
        )

        # Find contours using RETR_EXTERNAL and CHAIN_APPROX_SIMPLE for optimization
        contours, _ = cv2.findContours(top_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.debug(f"Found {len(contours)} contours in top region")

        object_headers = []

        for contour in contours:
            try:
                # Get bounding rectangle
                x, y, width, height = cv2.boundingRect(contour)

                # Filter out very small or very large contours
                # Object headers should be reasonably sized
                min_width, min_height = 30, 15
                max_width, max_height = w * 0.3, h * 0.15

                if (
                    width < min_width
                    or height < min_height
                    or width > max_width
                    or height > max_height
                ):
                    logger.debug(
                        f"Skipping contour with bbox ({x}, {y}, {width}, {height}) "
                        f"- size out of range"
                    )
                    continue

                # Aspect ratio check: object headers are typically wider than tall
                aspect_ratio = width / height
                if aspect_ratio < 0.5 or aspect_ratio > 10:
                    logger.debug(
                        f"Skipping contour with bbox ({x}, {y}, {width}, {height}) "
                        f"- aspect ratio {aspect_ratio:.2f} out of range"
                    )
                    continue

                # Extract text from the bounding box using OCR
                # Use grayscale image for better OCR results
                roi = preprocessed.grayscale[y : y + height, x : x + width]

                # Extract text using OCR engine
                text = self.ocr_engine.extract_from_region(
                    preprocessed.grayscale, (x, y, width, height)
                )

                if not text or len(text.strip()) == 0:
                    logger.debug(
                        f"Skipping contour with bbox ({x}, {y}, {width}, {height}) "
                        f"- no text extracted"
                    )
                    continue

                # Calculate lifeline X coordinate (center of the bounding box)
                lifeline_x = x + width // 2

                # For confidence, we'll use a simple heuristic based on text length
                # In a real implementation, this would come from OCR confidence
                confidence = min(100.0, len(text.strip()) * 10.0)

                object_header = ObjectHeader(
                    name=text.strip(),
                    lifeline_x=lifeline_x,
                    bounding_box=(x, y, width, height),
                    confidence=confidence,
                )

                object_headers.append(object_header)
                logger.debug(f"Detected object header: '{text.strip()}' at lifeline_x={lifeline_x}")

            except Exception as e:
                logger.warning(
                    f"Failed to process contour at bbox ({x}, {y}, {width}, {height}): {e}. "
                    f"Continuing with remaining objects."
                )
                continue

        # Sort object headers by X coordinate (left to right)
        object_headers.sort(key=lambda oh: oh.lifeline_x)

        logger.info(
            f"Object header detection completed. Found {len(object_headers)} objects: "
            f"{[oh.name for oh in object_headers]}"
        )

        return object_headers
