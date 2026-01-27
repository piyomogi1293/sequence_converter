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

        # Apply morphological closing to connect thin lines and form complete rectangles
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        top_region_closed = cv2.morphologyEx(top_region, cv2.MORPH_CLOSE, kernel)

        logger.debug(f"Applied morphological closing with kernel size {kernel_size}")

        # Find contours using RETR_EXTERNAL and CHAIN_APPROX_SIMPLE for optimization
        contours, _ = cv2.findContours(
            top_region_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        logger.debug(f"Total contours found in top region: {len(contours)}")

        object_headers = []
        candidate_count = 0

        # First pass: contour-based detection
        contour_candidates = []
        for contour in contours:
            try:
                # Get bounding rectangle
                x, y, width, height = cv2.boundingRect(contour)

                # Filter out very small or very large contours
                # Reduced minimum size to catch smaller headers
                min_width, min_height = 20, 10
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
                if aspect_ratio < 0.3 or aspect_ratio > 15:
                    logger.debug(
                        f"Skipping contour with bbox ({x}, {y}, {width}, {height}) "
                        f"- aspect ratio {aspect_ratio:.2f} out of range"
                    )
                    continue

                contour_candidates.append((x, y, width, height))
                candidate_count += 1

            except Exception as e:
                logger.warning(
                    f"Failed to process contour: {e}. Continuing with remaining objects."
                )
                continue

        logger.debug(f"Contour-based candidates: {len(contour_candidates)}")

        # Second pass: OCR-based detection for text regions in top portion
        # Extract all text regions from the top area
        ocr_candidates = self._extract_text_regions_from_top(preprocessed, top_region_height)
        logger.debug(f"OCR-based candidates: {len(ocr_candidates)}")

        # Merge candidates from both methods
        all_candidates = self._merge_candidates(contour_candidates, ocr_candidates)
        logger.debug(f"Total merged candidates: {len(all_candidates)}")

        # Process each candidate
        for x, y, width, height in all_candidates:
            try:
                # Filter by vertical position: object headers should be in top 15% of the region
                # This helps exclude message labels that appear lower
                if y > top_region_height * 0.15:
                    logger.debug(
                        f"Skipping candidate at ({x}, {y}, {width}, {height}) "
                        f"- too far down (y={y} > {top_region_height * 0.15:.0f})"
                    )
                    continue

                # Extract text using OCR engine
                text = self.ocr_engine.extract_from_region(
                    preprocessed.grayscale, (x, y, width, height)
                )

                # Prepare name and confidence based on OCR result
                name = text.strip() if text else ""

                if name:
                    # Skip if text looks like a message label (starts with number followed by period)
                    import re

                    if re.match(r"^\d+\.", name):
                        logger.debug(
                            f"Skipping candidate at ({x}, {y}, {width}, {height}) "
                            f"- looks like message label: '{name}'"
                        )
                        continue

                    # Calculate confidence based on text length heuristic
                    confidence = min(100.0, len(name) * 10.0)
                else:
                    # Skip candidates without text
                    logger.debug(f"Skipping candidate at ({x}, {y}, {width}, {height}) - no text")
                    continue

                # Calculate lifeline X coordinate (center of the bounding box)
                lifeline_x = x + width // 2

                object_header = ObjectHeader(
                    name=name,
                    lifeline_x=lifeline_x,
                    bounding_box=(x, y, width, height),
                    confidence=confidence,
                )

                object_headers.append(object_header)

                logger.debug(
                    f"Detected object header candidate: "
                    f"name='{name}', bbox=({x}, {y}, {width}, {height}), confidence={confidence}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to process candidate at bbox ({x}, {y}, {width}, {height}): {e}. "
                    f"Continuing with remaining objects."
                )
                continue

        # Sort object headers by X coordinate (left to right)
        object_headers.sort(key=lambda oh: oh.lifeline_x)

        logger.debug(
            f"Total candidate object header boxes after size/aspect filtering: {candidate_count}"
        )
        logger.info(
            f"Object header detection completed. Found {len(object_headers)} objects: "
            f"{[oh.name for oh in object_headers]}"
        )

        return object_headers

    def _extract_text_regions_from_top(
        self, preprocessed: PreprocessedImage, top_region_height: int
    ) -> list[tuple[int, int, int, int]]:
        """Extract text regions from the top portion using OCR.

        Args:
            preprocessed: 前処理済み画像
            top_region_height: 上部領域の高さ

        Returns:
            list[tuple[int, int, int, int]]: テキスト領域のバウンディングボックスリスト (x, y, w, h)
        """
        candidates = []

        try:
            grayscale = preprocessed.grayscale
            top_region_gray = grayscale[:top_region_height, :]

            # Use OCR backend to get all text regions
            ocr_data = self.ocr_engine.backend.extract_text_data(top_region_gray)

            for data in ocr_data:
                text = data.get("text", "").strip()
                if not text:
                    continue

                x = data.get("left", 0)
                y = data.get("top", 0)
                w = data.get("width", 0)
                h = data.get("height", 0)

                # Filter by reasonable size for object headers
                if w >= 20 and h >= 10 and w <= top_region_gray.shape[1] * 0.3:
                    candidates.append((x, y, w, h))

        except Exception as e:
            logger.debug(f"OCR-based text region extraction failed: {e}")

        return candidates

    def _merge_candidates(
        self,
        contour_candidates: list[tuple[int, int, int, int]],
        ocr_candidates: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Merge candidates from contour and OCR detection, removing duplicates.

        Args:
            contour_candidates: 輪郭ベースの候補
            ocr_candidates: OCRベースの候補

        Returns:
            list[tuple[int, int, int, int]]: マージされた候補リスト
        """
        merged = []
        all_candidates = contour_candidates + ocr_candidates

        for candidate in all_candidates:
            # Check if this candidate overlaps significantly with any existing candidate
            is_duplicate = False
            x1, y1, w1, h1 = candidate

            for existing in merged:
                x2, y2, w2, h2 = existing

                # Calculate intersection over union (IoU)
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    union_area = area1 + area2 - intersection_area

                    iou = intersection_area / union_area if union_area > 0 else 0

                    # If IoU > 0.5, consider it a duplicate
                    if iou > 0.5:
                        is_duplicate = True
                        break

            if not is_duplicate:
                merged.append(candidate)

        return merged
