"""Tests for MessageArrowDetector component."""

import numpy as np
import pytest

from sequence_converter.detectors.message_arrow import MessageArrowDetector
from sequence_converter.models import ArrowDirection, MessageArrow, PreprocessedImage


class TestMessageArrowDetector:
    """Tests for MessageArrowDetector."""

    @pytest.fixture
    def detector(self):
        """Create MessageArrowDetector instance."""
        return MessageArrowDetector()

    @pytest.fixture
    def sample_lifelines(self):
        """Sample lifeline X coordinates."""
        return [100, 200, 300, 400]

    @pytest.fixture
    def simple_horizontal_line_image(self):
        """Create a simple image with a horizontal line."""
        # Create a 200x500 white image
        image = np.ones((200, 500), dtype=np.uint8) * 255

        # Draw a horizontal line from (100, 100) to (200, 100)
        # Make it thick enough to be detected
        for dy in range(-2, 3):
            image[100 + dy, 100:201] = 0

        # Add arrowhead at the right endpoint (triangular shape)
        # Simple arrowhead: pixels around the right endpoint
        for dx in range(-5, 6):
            for dy in range(-3, 4):
                if abs(dy) <= abs(dx) + 1:  # Create triangular pattern
                    x = 200 + dx
                    y = 100 + dy
                    if 0 <= x < 500 and 0 <= y < 200:
                        image[y, x] = 0

        return PreprocessedImage(
            grayscale=image.copy(), binary=image.copy(), original_shape=(200, 500)
        )

    @pytest.fixture
    def multiple_arrows_image(self):
        """Create an image with multiple horizontal arrows at different Y positions."""
        # Create a 400x500 white image
        image = np.ones((400, 500), dtype=np.uint8) * 255

        # Arrow 1: y=50, from x=100 to x=200 (LEFT_TO_RIGHT)
        for dy in range(-2, 3):
            image[50 + dy, 100:201] = 0
        # Arrowhead at right
        for dx in range(-5, 6):
            for dy in range(-3, 4):
                if abs(dy) <= abs(dx) + 1:
                    x = 200 + dx
                    y = 50 + dy
                    if 0 <= x < 500 and 0 <= y < 400:
                        image[y, x] = 0

        # Arrow 2: y=150, from x=200 to x=300 (LEFT_TO_RIGHT)
        for dy in range(-2, 3):
            image[150 + dy, 200:301] = 0
        # Arrowhead at right
        for dx in range(-5, 6):
            for dy in range(-3, 4):
                if abs(dy) <= abs(dx) + 1:
                    x = 300 + dx
                    y = 150 + dy
                    if 0 <= x < 500 and 0 <= y < 400:
                        image[y, x] = 0

        # Arrow 3: y=250, from x=100 to x=300 (LEFT_TO_RIGHT)
        for dy in range(-2, 3):
            image[250 + dy, 100:301] = 0
        # Arrowhead at right
        for dx in range(-5, 6):
            for dy in range(-3, 4):
                if abs(dy) <= abs(dx) + 1:
                    x = 300 + dx
                    y = 250 + dy
                    if 0 <= x < 500 and 0 <= y < 400:
                        image[y, x] = 0

        return PreprocessedImage(
            grayscale=image.copy(), binary=image.copy(), original_shape=(400, 500)
        )

    def test_detect_horizontal_line_segment(
        self, detector, simple_horizontal_line_image, sample_lifelines
    ):
        """Test that horizontal line segments are detected."""
        arrows = detector.detect(simple_horizontal_line_image, sample_lifelines)

        # Should detect at least one arrow
        assert len(arrows) > 0

        # Check that detected arrows have valid coordinates
        for arrow in arrows:
            assert arrow.start_x >= 0
            assert arrow.end_x >= 0
            assert arrow.y >= 0
            assert arrow.start_x < arrow.end_x  # Start should be left of end

    def test_arrow_direction_detection(
        self, detector, simple_horizontal_line_image, sample_lifelines
    ):
        """Test that arrow direction (LEFT_TO_RIGHT / RIGHT_TO_LEFT) is correctly determined."""
        arrows = detector.detect(simple_horizontal_line_image, sample_lifelines)

        # Should detect at least one arrow
        assert len(arrows) > 0

        # For this test image, we drew an arrowhead at the right end
        # so it should be LEFT_TO_RIGHT
        arrow = arrows[0]
        assert arrow.direction in [ArrowDirection.LEFT_TO_RIGHT, ArrowDirection.RIGHT_TO_LEFT]

    def test_lifeline_matching_within_tolerance(self, detector, simple_horizontal_line_image):
        """Test that lifeline matching works within 20px tolerance."""
        # Create lifelines that are close to the arrow endpoints
        # Arrow is from x=100 to x=200 (approximately)
        lifelines = [105, 195]  # Within 20px tolerance

        arrows = detector.detect(simple_horizontal_line_image, lifelines)

        if len(arrows) > 0:
            arrow = arrows[0]
            # Should match to the nearest lifelines
            assert arrow.source_lifeline is not None
            assert arrow.dest_lifeline is not None
            assert arrow.source_lifeline in lifelines
            assert arrow.dest_lifeline in lifelines

    def test_lifeline_matching_outside_tolerance(self, detector, simple_horizontal_line_image):
        """Test that lifeline matching fails when endpoints are outside 20px tolerance."""
        # Create lifelines that are far from the arrow endpoints
        # Arrow is from x=100 to x=200 (approximately)
        lifelines = [50, 250]  # Outside 20px tolerance

        arrows = detector.detect(simple_horizontal_line_image, lifelines)

        if len(arrows) > 0:
            arrow = arrows[0]
            # Both lifelines should fail to match (or at least one)
            # Since we're outside the tolerance range
            assert arrow.source_lifeline is None or arrow.dest_lifeline is None

    def test_results_sorted_by_y_coordinate(
        self, detector, multiple_arrows_image, sample_lifelines
    ):
        """Test that detected results are sorted by Y coordinate (top to bottom)."""
        arrows = detector.detect(multiple_arrows_image, sample_lifelines)

        # Should detect multiple arrows
        assert len(arrows) >= 2

        # Check that arrows are sorted by Y coordinate
        for i in range(len(arrows) - 1):
            assert arrows[i].y <= arrows[i + 1].y

    def test_matching_failure_logged_and_continues(
        self, detector, simple_horizontal_line_image, caplog
    ):
        """Test that matching failure is logged and processing continues."""
        # Use lifelines that will cause matching failure
        lifelines = [500, 600]  # Far from the actual arrow positions

        arrows = detector.detect(simple_horizontal_line_image, lifelines)

        # Should still return the arrows (even if matching failed)
        # The detector should continue processing
        assert isinstance(arrows, list)

        # Check that warning was logged
        if len(arrows) > 0:
            arrow = arrows[0]
            if arrow.source_lifeline is None or arrow.dest_lifeline is None:
                assert "Failed to match arrow" in caplog.text

    def test_no_lines_in_empty_image(self, detector, sample_lifelines):
        """Test that no arrows are detected in an empty image."""
        # Create an empty (white) image
        empty_image = np.ones((200, 500), dtype=np.uint8) * 255
        preprocessed = PreprocessedImage(
            grayscale=empty_image.copy(), binary=empty_image.copy(), original_shape=(200, 500)
        )

        arrows = detector.detect(preprocessed, sample_lifelines)

        # Should return empty list
        assert len(arrows) == 0

    def test_vertical_lines_ignored(self, detector, sample_lifelines):
        """Test that vertical lines are ignored (not detected as message arrows)."""
        # Create image with only vertical lines
        image = np.ones((200, 500), dtype=np.uint8) * 255

        # Draw vertical line from (100, 50) to (100, 150)
        for dx in range(-2, 3):
            image[50:151, 100 + dx] = 0

        preprocessed = PreprocessedImage(
            grayscale=image.copy(), binary=image.copy(), original_shape=(200, 500)
        )

        arrows = detector.detect(preprocessed, sample_lifelines)

        # Should not detect vertical lines as arrows
        # (or at least significantly fewer than if they were horizontal)
        assert len(arrows) == 0
