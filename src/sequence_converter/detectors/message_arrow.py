"""Message Arrow Detector component for detecting message arrows in sequence diagrams."""

import logging
from typing import Optional

import cv2
import numpy as np

from sequence_converter.models import ArrowDirection, MessageArrow, PreprocessedImage

logger = logging.getLogger(__name__)


class MessageArrowDetector:
    """Message Arrow Detector component.

    Detects horizontal line segments representing message arrows and determines
    their direction and lifeline mappings.
    """

    LIFELINE_MATCH_TOLERANCE: int = 100  # ピクセル（矢印ヘッドとテキストラベルの幅を考慮）

    def __init__(self):
        """Initialize MessageArrowDetector."""
        logger.info("MessageArrowDetector initialized")

    def detect(self, preprocessed: PreprocessedImage, lifelines: list[int]) -> list[MessageArrow]:
        """Detect message arrows from horizontal line segments.

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト（ObjectHeaderから取得）

        Returns:
            list[MessageArrow]: 検出されたメッセージ矢印（上から下へソート済み）
        """
        logger.info(f"Starting message arrow detection with {len(lifelines)} lifelines")
        logger.debug(f"Lifeline X coordinates: {lifelines}")

        # Use binary image for line detection
        # preprocessed.binary is already inverted (white lines on black background)
        # from preprocessing: cv2.THRESH_BINARY_INV makes black lines white
        # So we use it directly without additional inversion
        binary_image = preprocessed.binary.copy()

        # Remove text regions using vertical erosion
        # Text characters have significant vertical extent, while arrow lines are thin
        # Eroding vertically will remove text but preserve horizontal lines
        kernel_v_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        binary_image = cv2.erode(binary_image, kernel_v_erode, iterations=1)
        logger.debug("Applied vertical erosion to remove text (1x5 kernel)")

        # Apply strong horizontal morphological closing to connect arrow line fragments
        # Arrow lines are often fragmented in sequence diagrams
        # Use very large kernel to bridge gaps and connect fragments into complete lines
        # This is necessary because arrows can span long distances between lifelines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_h)
        logger.debug("Applied horizontal morphological closing (100x1 kernel)")

        # Detect horizontal lines using HoughLinesP
        # Use fine theta step to detect lines close to horizontal
        lines_detected = cv2.HoughLinesP(
            binary_image,
            rho=1,
            theta=np.pi / 180,  # Standard angle step
            threshold=20,  # Lower threshold to detect weaker/shorter lines
            minLineLength=80,  # Lower minimum to catch shorter arrows
            maxLineGap=50,  # Larger gap to connect highly fragmented lines
        )

        # Filter to keep only horizontal lines
        horizontal_lines = []
        if lines_detected is not None:
            for line in lines_detected:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Skip if not enough horizontal distance
                if dx < 80:
                    continue

                # Calculate angle from horizontal
                angle = np.degrees(np.arctan2(dy, dx)) if dx > 0 else 90.0

                # Only accept lines within 5 degrees of horizontal
                # Also check that vertical deviation is small relative to horizontal length
                if angle <= 5.0 and dy <= max(5, dx * 0.08):
                    horizontal_lines.append(line)

        lines = np.array(horizontal_lines) if horizontal_lines else None

        if lines is None:
            logger.warning("No lines detected in image")
            return []

        logger.debug(f"HoughLinesP detected {len(lines)} horizontal line segments")

        message_arrows = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Note: Lines are already filtered for horizontal orientation
            # Calculate average Y coordinate
            y_avg = (y1 + y2) // 2

            # Determine start and end points (ensure start_x < end_x for consistency)
            if x1 <= x2:
                start_x, end_x = x1, x2
                # start_y, end_y = y1, y2
            else:
                start_x, end_x = x2, x1
                # start_y, end_y = y2, y1

            # Check minimum arrow length (should span reasonable distance)
            # In sequence diagrams, arrows typically span between lifelines
            arrow_length = end_x - start_x
            if arrow_length < 50:  # Minimum 50 pixels for significant arrows
                logger.debug(f"Skipping short line at y={y_avg} (length={arrow_length}px)")
                continue

            # Determine arrow direction by checking for arrowhead at endpoints
            direction = self._detect_arrow_direction(binary_image, start_x, end_x, y_avg)

            if direction is None:
                # If no clear direction detected, skip this line
                logger.debug(f"Skipping line at y={y_avg} - no clear arrow direction detected")
                continue

            # Match endpoints to lifelines
            source_lifeline = self._match_to_lifeline(start_x, lifelines)
            dest_lifeline = self._match_to_lifeline(end_x, lifelines)

            # Skip arrows that don't match both lifelines (likely false positives)
            # In sequence diagrams, message arrows must connect lifelines
            if source_lifeline is None or dest_lifeline is None:
                logger.debug(
                    f"Skipping arrow at y={y_avg} (start_x={start_x}, end_x={end_x}) "
                    f"- failed to match both lifelines (source={source_lifeline}, "
                    f"dest={dest_lifeline})"
                )
                continue

            # Skip self-loops (these should be detected by SelfCallDetector)
            if source_lifeline == dest_lifeline:
                logger.debug(
                    f"Skipping arrow at y={y_avg} - appears to be self-loop "
                    f"(lifeline={source_lifeline})"
                )
                continue

            message_arrow = MessageArrow(
                start_x=start_x,
                end_x=end_x,
                y=y_avg,
                direction=direction,
                source_lifeline=source_lifeline,
                dest_lifeline=dest_lifeline,
            )

            message_arrows.append(message_arrow)
            logger.debug(
                f"Detected message arrow: {direction.value} at y={y_avg}, "
                f"start_x={start_x}, end_x={end_x}"
            )

        # Remove duplicate/overlapping arrows (keep the longest one at each Y position)
        message_arrows = self._remove_duplicates(message_arrows)

        # Sort by Y coordinate (top to bottom)
        message_arrows.sort(key=lambda arrow: arrow.y)

        logger.info(f"Message arrow detection completed. Found {len(message_arrows)} arrows")

        return message_arrows

    def _detect_arrow_direction(
        self, image: np.ndarray, start_x: int, end_x: int, y: int
    ) -> Optional[ArrowDirection]:
        """Detect arrow direction by analyzing endpoints.

        Looks for triangular or '>' shaped patterns at the endpoints
        to determine the arrow direction.

        Args:
            image: Binary image
            start_x: Start X coordinate (left endpoint)
            end_x: End X coordinate (right endpoint)
            y: Y coordinate of the arrow

        Returns:
            ArrowDirection or None if direction cannot be determined
        """
        h, w = image.shape

        # Define search regions around endpoints (±10 pixels in X, ±5 pixels in Y)
        search_radius_x = 10
        search_radius_y = 5

        # Check right endpoint for arrowhead (LEFT_TO_RIGHT arrow)
        right_x_min = max(0, end_x - search_radius_x)
        right_x_max = min(w, end_x + search_radius_x)
        right_y_min = max(0, y - search_radius_y)
        right_y_max = min(h, y + search_radius_y)

        right_region = image[right_y_min:right_y_max, right_x_min:right_x_max]

        # Check left endpoint for arrowhead (RIGHT_TO_LEFT arrow)
        left_x_min = max(0, start_x - search_radius_x)
        left_x_max = min(w, start_x + search_radius_x)
        left_y_min = max(0, y - search_radius_y)
        left_y_max = min(h, y + search_radius_y)

        left_region = image[left_y_min:left_y_max, left_x_min:left_x_max]

        # Count non-zero pixels in each region (simple heuristic)
        # More pixels at the endpoint suggests an arrowhead
        right_pixel_count = cv2.countNonZero(right_region) if right_region.size > 0 else 0
        left_pixel_count = cv2.countNonZero(left_region) if left_region.size > 0 else 0

        # Determine direction based on which endpoint has more pixels
        # This is a simple heuristic; a more sophisticated approach would
        # use contour detection to identify actual arrow shapes
        if right_pixel_count > left_pixel_count * 1.2:
            return ArrowDirection.LEFT_TO_RIGHT
        elif left_pixel_count > right_pixel_count * 1.2:
            return ArrowDirection.RIGHT_TO_LEFT
        else:
            # Default to LEFT_TO_RIGHT if unclear
            # In real diagrams, most arrows are left-to-right
            return ArrowDirection.LEFT_TO_RIGHT

    def _remove_duplicates(self, arrows: list[MessageArrow]) -> list[MessageArrow]:
        """Remove duplicate arrows that are too close in Y coordinate.

        Keep the longest arrow when multiple arrows are detected at similar Y positions.

        Args:
            arrows: List of detected arrows

        Returns:
            List of arrows with duplicates removed
        """
        if not arrows:
            return []

        # Group arrows by similar Y coordinate (within 10 pixels)
        groups = []
        Y_TOLERANCE = 10

        for arrow in arrows:
            # Find existing group for this arrow
            found_group = False
            for group in groups:
                if abs(group[0].y - arrow.y) <= Y_TOLERANCE:
                    group.append(arrow)
                    found_group = True
                    break

            if not found_group:
                groups.append([arrow])

        # For each group, keep only the longest arrow
        result = []
        for group in groups:
            if len(group) == 1:
                result.append(group[0])
            else:
                # Keep the arrow with the longest length
                longest = max(group, key=lambda a: a.end_x - a.start_x)
                result.append(longest)
                logger.debug(
                    f"Removed {len(group) - 1} duplicate arrows at y≈{longest.y}, "
                    f"kept longest ({longest.end_x - longest.start_x}px)"
                )

        return result

    def _match_to_lifeline(self, x: int, lifelines: list[int]) -> Optional[int]:
        """Match an X coordinate to the nearest lifeline within tolerance.

        Args:
            x: X coordinate to match
            lifelines: List of lifeline X coordinates

        Returns:
            Matched lifeline X coordinate or None if no match within tolerance
        """
        if not lifelines:
            return None

        # Find the closest lifeline
        min_distance = float("inf")
        closest_lifeline = None

        for lifeline_x in lifelines:
            distance = abs(x - lifeline_x)
            if distance < min_distance:
                min_distance = distance
                closest_lifeline = lifeline_x

        # Check if within tolerance
        if min_distance <= self.LIFELINE_MATCH_TOLERANCE:
            return closest_lifeline
        else:
            return None
