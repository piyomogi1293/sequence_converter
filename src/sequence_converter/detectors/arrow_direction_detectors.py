"""Arrow direction detection methods for message arrows."""

import logging
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from sequence_converter.models import ArrowDirection

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Arrow direction detection methods."""

    GEOMETRIC = "geometric"
    TEMPLATE = "template"
    CLASSIFIER = "classifier"
    COMBINED = "combined"


class GeometricArrowDetector:
    """Geometric rule-based arrow direction detector.

    Detects arrowheads by looking for:
    1. Two diagonal lines forming '>' or '<' shape
    2. Filled triangular regions
    """

    def __init__(
        self,
        roi_size: int = 35,
        min_diagonal_lines: int = 2,
        angle_threshold: float = 20.0,
    ):
        """Initialize geometric detector.

        Args:
            roi_size: Size of ROI around endpoints (pixels)
            min_diagonal_lines: Minimum diagonal lines for arrowhead
            angle_threshold: Max angle deviation from expected diagonal (degrees)
        """
        self.roi_size = roi_size
        self.min_diagonal_lines = min_diagonal_lines
        self.angle_threshold = angle_threshold

    def detect_direction(
        self, image: np.ndarray, start_x: int, end_x: int, y: int
    ) -> Optional[ArrowDirection]:
        """Detect arrow direction using geometric rules.

        Args:
            image: Binary image (white pixels on black background)
            start_x: Left endpoint X
            end_x: Right endpoint X
            y: Y coordinate

        Returns:
            Detected arrow direction or None
        """
        h, w = image.shape

        # Extract ROIs around both endpoints
        left_roi = self._extract_roi(image, start_x, y, h, w)
        right_roi = self._extract_roi(image, end_x, y, h, w)

        # Analyze both endpoints
        left_score = self._analyze_arrowhead(left_roi, pointing_right=False)
        right_score = self._analyze_arrowhead(right_roi, pointing_right=True)

        logger.debug(f"Geometric scores at y={y}: left={left_score:.2f}, right={right_score:.2f}")

        # Determine direction based on scores
        # Reduced threshold for more sensitive detection
        if right_score > left_score * 1.15:
            return ArrowDirection.LEFT_TO_RIGHT
        elif left_score > right_score * 1.15:
            return ArrowDirection.RIGHT_TO_LEFT
        elif right_score > 20:  # Absolute threshold
            return ArrowDirection.LEFT_TO_RIGHT
        elif left_score > 20:
            return ArrowDirection.RIGHT_TO_LEFT
        else:
            # Unclear - return None or default
            return None

    def _extract_roi(self, image: np.ndarray, x: int, y: int, h: int, w: int) -> np.ndarray:
        """Extract ROI around endpoint.

        Args:
            image: Source image
            x: Center X
            y: Center Y
            h: Image height
            w: Image width

        Returns:
            ROI image
        """
        half_size = self.roi_size // 2
        y_min = max(0, y - half_size)
        y_max = min(h, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(w, x + half_size)

        roi = image[y_min:y_max, x_min:x_max]
        return roi

    def _analyze_arrowhead(self, roi: np.ndarray, pointing_right: bool) -> float:
        """Analyze ROI for arrowhead pattern.

        Looks for:
        1. Diagonal lines forming > or < shape
        2. Triangular filled region

        Args:
            roi: ROI image
            pointing_right: True if looking for '>', False for '<'

        Returns:
            Confidence score (0-100)
        """
        if roi.size == 0:
            return 0.0

        score = 0.0

        # Method 1: Detect diagonal lines using HoughLinesP
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=8,  # Lower threshold for better sensitivity
            minLineLength=4,
            maxLineGap=4,
        )

        if lines is not None:
            diagonal_count = 0
            angle_sum = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

                # Expected angles for diagonal lines in arrowhead: 15-75 degrees
                if 15 <= angle <= 75:
                    # Check if line points in correct direction
                    if pointing_right:
                        # For '>' shape: lines should converge towards right
                        # Lines should go from left to right
                        if abs(dx) > 2:  # Significant horizontal component
                            diagonal_count += 1
                            angle_sum += angle
                    else:
                        # For '<' shape: lines should converge towards left
                        if abs(dx) > 2:
                            diagonal_count += 1
                            angle_sum += angle

            if diagonal_count >= self.min_diagonal_lines:
                score += 50.0  # Increased weight for diagonal lines
            elif diagonal_count >= 1:
                score += 25.0  # Partial credit

        # Method 2: Check for filled triangular region
        # Use contour analysis to find triangular shapes
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_triangle_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8:  # Too small
                continue
            if area > roi.size * 0.7:  # Too large (probably the whole ROI)
                continue

            # Approximate contour to polygon
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if triangle-like (3-5 vertices)
            if len(approx) in [3, 4, 5]:
                # Check aspect ratio and orientation
                x, y, w_c, h_c = cv2.boundingRect(contour)

                # Triangular arrowheads can vary in aspect ratio
                aspect_ratio = w_c / h_c if h_c > 0 else 0

                if 0.3 < aspect_ratio < 5.0:
                    # Higher score for more triangle-like shapes
                    triangle_score = 35.0
                    if len(approx) == 3:
                        triangle_score = 45.0  # Perfect triangle
                    max_triangle_score = max(max_triangle_score, triangle_score)

        score += max_triangle_score

        # Method 3: Check pixel density pattern
        # Arrowheads have higher density at the tip
        if roi.shape[1] > 4 and roi.shape[0] > 4:
            if pointing_right:
                # Check if right side has more pixels (arrowhead at right)
                mid = roi.shape[1] // 2
                left_half = roi[:, :mid]
                right_half = roi[:, mid:]

                left_density = np.sum(left_half) / (left_half.size + 1)
                right_density = np.sum(right_half) / (right_half.size + 1)

                density_ratio = right_density / (left_density + 1e-6)
                if density_ratio > 1.3:
                    score += 35.0
                elif density_ratio > 1.1:
                    score += 20.0
            else:
                # Check if left side has more pixels (arrowhead at left)
                mid = roi.shape[1] // 2
                left_half = roi[:, :mid]
                right_half = roi[:, mid:]

                left_density = np.sum(left_half) / (left_half.size + 1)
                right_density = np.sum(right_half) / (right_half.size + 1)

                density_ratio = left_density / (right_density + 1e-6)
                if density_ratio > 1.3:
                    score += 35.0
                elif density_ratio > 1.1:
                    score += 20.0

        return score


class TemplateArrowDetector:
    """Template matching-based arrow direction detector.

    Uses pre-defined arrow templates to match arrowheads.
    """

    def __init__(self, roi_size: int = 25, match_threshold: float = 0.6):
        """Initialize template detector.

        Args:
            roi_size: Size of ROI and templates
            match_threshold: Minimum matching score (0-1)
        """
        self.roi_size = roi_size
        self.match_threshold = match_threshold
        self.templates_right = self._create_right_arrow_templates()
        self.templates_left = self._create_left_arrow_templates()

    def _create_right_arrow_templates(self) -> list[np.ndarray]:
        """Create templates for right-pointing arrows ('>').

        Returns:
            List of template images
        """
        templates = []
        size = self.roi_size

        # Template 1: Simple '>' shape with lines
        template1 = np.zeros((size, size), dtype=np.uint8)
        center_y = size // 2
        # Draw two diagonal lines forming '>'
        cv2.line(
            template1,
            (size // 4, center_y - size // 3),
            (3 * size // 4, center_y),
            255,
            2,
        )
        cv2.line(
            template1,
            (size // 4, center_y + size // 3),
            (3 * size // 4, center_y),
            255,
            2,
        )
        templates.append(template1)

        # Template 2: Filled triangle pointing right
        template2 = np.zeros((size, size), dtype=np.uint8)
        pts = np.array(
            [
                [size // 4, center_y - size // 3],
                [size // 4, center_y + size // 3],
                [3 * size // 4, center_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(template2, [pts], 255)
        templates.append(template2)

        # Template 3: Thinner arrow
        template3 = np.zeros((size, size), dtype=np.uint8)
        cv2.line(
            template3,
            (size // 3, center_y - size // 4),
            (2 * size // 3, center_y),
            255,
            1,
        )
        cv2.line(
            template3,
            (size // 3, center_y + size // 4),
            (2 * size // 3, center_y),
            255,
            1,
        )
        templates.append(template3)

        return templates

    def _create_left_arrow_templates(self) -> list[np.ndarray]:
        """Create templates for left-pointing arrows ('<').

        Returns:
            List of template images
        """
        # Simply flip the right-pointing templates
        return [cv2.flip(t, 1) for t in self.templates_right]

    def detect_direction(
        self, image: np.ndarray, start_x: int, end_x: int, y: int
    ) -> Optional[ArrowDirection]:
        """Detect arrow direction using template matching.

        Args:
            image: Binary image
            start_x: Left endpoint X
            end_x: Right endpoint X
            y: Y coordinate

        Returns:
            Detected arrow direction or None
        """
        h, w = image.shape

        # Extract ROIs
        left_roi = self._extract_roi(image, start_x, y, h, w)
        right_roi = self._extract_roi(image, end_x, y, h, w)

        # Match against templates
        left_score = self._match_templates(left_roi, self.templates_left)
        right_score = self._match_templates(right_roi, self.templates_right)

        logger.debug(f"Template scores at y={y}: left={left_score:.3f}, right={right_score:.3f}")

        # Determine direction based on best match
        if right_score > self.match_threshold and right_score > left_score * 1.2:
            return ArrowDirection.LEFT_TO_RIGHT
        elif left_score > self.match_threshold and left_score > right_score * 1.2:
            return ArrowDirection.RIGHT_TO_LEFT
        else:
            return None

    def _extract_roi(self, image: np.ndarray, x: int, y: int, h: int, w: int) -> np.ndarray:
        """Extract and resize ROI to template size."""
        half_size = self.roi_size // 2
        y_min = max(0, y - half_size)
        y_max = min(h, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(w, x + half_size)

        roi = image[y_min:y_max, x_min:x_max]

        # Resize to template size if needed
        if roi.shape[0] != self.roi_size or roi.shape[1] != self.roi_size:
            roi = cv2.resize(roi, (self.roi_size, self.roi_size))

        return roi

    def _match_templates(self, roi: np.ndarray, templates: list[np.ndarray]) -> float:
        """Match ROI against templates.

        Args:
            roi: ROI image
            templates: List of templates to match

        Returns:
            Best matching score (0-1)
        """
        if roi.size == 0:
            return 0.0

        best_score = 0.0

        for template in templates:
            # Ensure same size
            if roi.shape != template.shape:
                continue

            # Use normalized cross-correlation
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0] if result.size > 0 else 0.0

            # Normalize to 0-1 range
            score = (score + 1) / 2  # TM_CCOEFF_NORMED ranges from -1 to 1

            best_score = max(best_score, score)

        return best_score


class CombinedArrowDetector:
    """Combined detector using multiple methods."""

    def __init__(self):
        """Initialize combined detector."""
        self.geometric = GeometricArrowDetector()
        self.template = TemplateArrowDetector()

    def detect_direction(
        self, image: np.ndarray, start_x: int, end_x: int, y: int
    ) -> Optional[ArrowDirection]:
        """Detect direction using voting from multiple methods.

        Args:
            image: Binary image
            start_x: Left endpoint X
            end_x: Right endpoint X
            y: Y coordinate

        Returns:
            Detected arrow direction or None
        """
        # Get predictions from each method
        geometric_dir = self.geometric.detect_direction(image, start_x, end_x, y)
        template_dir = self.template.detect_direction(image, start_x, end_x, y)

        # Voting logic
        votes = {}
        if geometric_dir is not None:
            votes[geometric_dir] = votes.get(geometric_dir, 0) + 2  # Higher weight
        if template_dir is not None:
            votes[template_dir] = votes.get(template_dir, 0) + 1

        if not votes:
            return None

        # Return direction with most votes
        best_dir = max(votes.items(), key=lambda x: x[1])
        logger.debug(f"Combined voting at y={y}: {votes}, selected={best_dir[0].value}")

        return best_dir[0]
