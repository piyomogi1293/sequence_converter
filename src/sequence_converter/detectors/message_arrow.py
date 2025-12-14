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

    LIFELINE_MATCH_TOLERANCE: int = 20  # ピクセル（制約条件より）

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
        binary_image = preprocessed.binary

        # Detect horizontal lines using HoughLinesP
        lines = cv2.HoughLinesP(
            binary_image,
            rho=1,  # Distance resolution in pixels
            theta=np.pi / 180,  # Angle resolution in radians
            threshold=50,  # Minimum number of intersections to detect a line
            minLineLength=30,  # Minimum line length
            maxLineGap=10,  # Maximum gap between line segments
        )

        if lines is None:
            logger.warning("No lines detected in image")
            return []

        logger.debug(f"HoughLinesP detected {len(lines)} line segments")

        message_arrows = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line is approximately horizontal
            # Allow small vertical deviation (max 5 pixels)
            if abs(y2 - y1) > 5:
                continue

            # Calculate average Y coordinate
            y_avg = (y1 + y2) // 2

            # Determine start and end points (ensure start_x < end_x for consistency)
            if x1 <= x2:
                start_x, end_x = x1, x2
                # start_y, end_y = y1, y2
            else:
                start_x, end_x = x2, x1
                # start_y, end_y = y2, y1

            # Determine arrow direction by checking for arrowhead at endpoints
            direction = self._detect_arrow_direction(binary_image, start_x, end_x, y_avg)

            if direction is None:
                # If no clear direction detected, skip this line
                logger.debug(f"Skipping line at y={y_avg} - no clear arrow direction detected")
                continue

            # Match endpoints to lifelines
            source_lifeline = self._match_to_lifeline(start_x, lifelines)
            dest_lifeline = self._match_to_lifeline(end_x, lifelines)

            # Log warning if lifeline matching failed
            if source_lifeline is None or dest_lifeline is None:
                logger.warning(
                    f"Failed to match arrow at y={y_avg} (start_x={start_x}, end_x={end_x}) "
                    f"to lifelines. source_lifeline={source_lifeline}, "
                    f"dest_lifeline={dest_lifeline}. Continuing with remaining arrows."
                )
                # Continue processing even if matching failed
                # The arrow will have None for unmatched lifelines

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
