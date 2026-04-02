"""Tests for ObjectHeaderDetector component."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from sequence_converter.detectors.object_header import ObjectHeaderDetector
from sequence_converter.models import ObjectHeader, PreprocessedImage
from sequence_converter.ocr import OCREngine


@pytest.fixture
def mock_ocr_engine():
    """Create a mock OCR engine for testing."""
    mock_ocr = MagicMock(spec=OCREngine)
    return mock_ocr


@pytest.fixture
def sample_preprocessed_image():
    """Create a sample preprocessed image with simple rectangles."""
    # Create a 400x600 white image
    grayscale = np.ones((400, 600), dtype=np.uint8) * 255
    binary = np.ones((400, 600), dtype=np.uint8) * 255

    # Draw three black rectangles in the top region (simulating object headers)
    # Rectangle 1: x=50, y=20, width=80, height=30
    binary[20:50, 50:130] = 0
    grayscale[20:50, 50:130] = 0

    # Rectangle 2: x=200, y=25, width=90, height=35
    binary[25:60, 200:290] = 0
    grayscale[25:60, 200:290] = 0

    # Rectangle 3: x=400, y=22, width=85, height=32
    binary[22:54, 400:485] = 0
    grayscale[22:54, 400:485] = 0

    return PreprocessedImage(grayscale=grayscale, binary=binary, original_shape=(400, 600))


class TestObjectHeaderDetector:
    """Test suite for ObjectHeaderDetector."""

    def test_initialization_with_ocr_engine(self, mock_ocr_engine):
        """Test that ObjectHeaderDetector initializes with OCR engine."""
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)
        assert detector.ocr_engine == mock_ocr_engine

    def test_detect_returns_list(self, mock_ocr_engine, sample_preprocessed_image):
        """Test that detect returns a list of ObjectHeader."""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "TestObject"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, ObjectHeader)

    def test_detect_multiple_rectangles_sorted_left_to_right(
        self, mock_ocr_engine, sample_preprocessed_image
    ):
        """複数の矩形が検出され、左から右へソートされることをテスト"""
        # Arrange - Mock OCR to return different names for each rectangle
        mock_ocr_engine.extract_from_region.side_effect = [
            "ObjectA",  # First rectangle (x=50)
            "ObjectB",  # Second rectangle (x=200)
            "ObjectC",  # Third rectangle (x=400)
        ]
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert - Should be sorted by lifeline_x (left to right)
        assert len(results) >= 3
        # Verify sorting: each subsequent object should have larger or equal lifeline_x
        for i in range(len(results) - 1):
            assert results[i].lifeline_x <= results[i + 1].lifeline_x

        # Verify the names are in the correct order (based on x position)
        object_names = [obj.name for obj in results]
        assert "ObjectA" in object_names
        assert "ObjectB" in object_names
        assert "ObjectC" in object_names

        # Find indices of each object
        idx_a = object_names.index("ObjectA")
        idx_b = object_names.index("ObjectB")
        idx_c = object_names.index("ObjectC")

        # Verify they are in left-to-right order
        assert idx_a < idx_b < idx_c

    def test_ocr_extracts_text_from_rectangle_region(
        self, mock_ocr_engine, sample_preprocessed_image
    ):
        """OCRが矩形領域から正しくテキストを抽出することをテスト"""
        # Arrange
        expected_text = "ParticipantName"
        mock_ocr_engine.extract_from_region.return_value = expected_text
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert len(results) > 0
        assert mock_ocr_engine.extract_from_region.called
        assert results[0].name == expected_text

    def test_lifeline_x_is_rectangle_center(self, mock_ocr_engine):
        """ライフラインX座標が矩形中心として記録されることをテスト"""
        # Arrange - Create simple image with one known rectangle
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        # Draw rectangle: x=100, y=20, width=60, height=30
        # Expected center x = 100 + 60//2 = 100 + 30 = 130
        binary[20:50, 100:160] = 0
        grayscale[20:50, 100:160] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "TestObject"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert
        assert len(results) == 1
        # Center X should be approximately 130 (x + width//2 = 100 + 30)
        assert results[0].lifeline_x == 130
        assert results[0].bounding_box[0] == 100  # x position
        assert results[0].bounding_box[2] == 60  # width

    def test_partial_detection_failure_continues_processing(
        self, mock_ocr_engine, sample_preprocessed_image, caplog
    ):
        """一部矩形の検出失敗時もログを記録し処理を継続することをテスト"""
        # Arrange - Make OCR fail for some rectangles
        mock_ocr_engine.extract_from_region.side_effect = [
            "Object1",  # Success
            "",  # Failure: empty text
            "Object3",  # Success
        ]
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert - Should continue processing despite failures
        assert len(results) >= 2  # At least Object1 and Object3
        object_names = [obj.name for obj in results]
        assert "Object1" in object_names
        assert "Object3" in object_names

        # Verify logging occurred (debug level for skipped contours)
        # Note: This will depend on logger configuration

    def test_empty_text_extraction_skips_rectangle(
        self, mock_ocr_engine, sample_preprocessed_image
    ):
        """OCRがテキストを抽出できない場合、矩形をスキップすることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = ""
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert - No results since all OCR extractions return empty
        assert len(results) == 0

    def test_whitespace_only_text_skips_rectangle(self, mock_ocr_engine, sample_preprocessed_image):
        """OCRが空白のみを返す場合、矩形をスキップすることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "   \t\n  "
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert len(results) == 0

    def test_object_header_has_required_fields(self, mock_ocr_engine, sample_preprocessed_image):
        """検出されたObjectHeaderが必須フィールドを持つことをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "TestObject"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert len(results) > 0
        obj = results[0]
        assert isinstance(obj.name, str)
        assert len(obj.name) > 0
        assert isinstance(obj.lifeline_x, int)
        assert obj.lifeline_x >= 0
        assert isinstance(obj.bounding_box, tuple)
        assert len(obj.bounding_box) == 4
        assert isinstance(obj.confidence, float)
        assert -1.0 <= obj.confidence <= 100.0

    def test_bounding_box_format(self, mock_ocr_engine, sample_preprocessed_image):
        """バウンディングボックスが(x, y, width, height)形式であることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "TestObject"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert len(results) > 0
        x, y, width, height = results[0].bounding_box
        assert x >= 0
        assert y >= 0
        assert width > 0
        assert height > 0

    def test_filters_very_small_rectangles(self, mock_ocr_engine):
        """非常に小さい矩形がフィルタリングされることをテスト"""
        # Arrange - Create image with very small rectangle
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        # Draw very small rectangle: width=5, height=5 (below min threshold)
        binary[20:25, 100:105] = 0
        grayscale[20:25, 100:105] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "Small"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert - Should be filtered out
        assert len(results) == 0

    def test_filters_very_large_rectangles(self, mock_ocr_engine):
        """非常に大きい矩形がフィルタリングされることをテスト"""
        # Arrange - Create image with very large rectangle
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        # Draw very large rectangle: width=200, height=100 (above max threshold)
        binary[10:110, 50:250] = 0
        grayscale[10:110, 50:250] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "Large"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert - Should be filtered out
        assert len(results) == 0

    def test_focuses_on_top_region_only(self, mock_ocr_engine):
        """画像上部30%のみを検索することをテスト"""
        # Arrange - Create image with rectangles in top and bottom regions
        grayscale = np.ones((400, 600), dtype=np.uint8) * 255
        binary = np.ones((400, 600), dtype=np.uint8) * 255

        # Top region rectangle (should be detected): y=20
        binary[20:50, 100:180] = 0
        grayscale[20:50, 100:180] = 0

        # Bottom region rectangle (should NOT be detected): y=300 (below 30% = 120)
        binary[300:330, 100:180] = 0
        grayscale[300:330, 100:180] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 600)
        )

        mock_ocr_engine.extract_from_region.return_value = "Object"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert - Should only detect top rectangle
        assert len(results) == 1
        # Y coordinate should be in top 30% (< 120)
        assert results[0].bounding_box[1] < 120

    def test_handles_ocr_exception_gracefully(
        self, mock_ocr_engine, sample_preprocessed_image, caplog
    ):
        """OCRエンジンが例外を発生させた場合の処理をテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.side_effect = [
            "Object1",  # Success
            Exception("OCR failed"),  # Failure
            "Object3",  # Success
        ]
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert - Should continue processing despite exception
        assert len(results) >= 2  # At least Object1 and Object3
        object_names = [obj.name for obj in results]
        assert "Object1" in object_names
        assert "Object3" in object_names

        # Verify warning was logged
        assert any("Failed to process contour" in record.message for record in caplog.records)

    def test_confidence_calculation(self, mock_ocr_engine, sample_preprocessed_image):
        """信頼度スコアが適切に計算されることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "TestObject"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert len(results) > 0
        # Confidence should be within valid range
        for obj in results:
            assert -1.0 <= obj.confidence <= 100.0

    def test_text_is_stripped(self, mock_ocr_engine, sample_preprocessed_image):
        """抽出されたテキストがトリミングされることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "  TestObject  \n"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(sample_preprocessed_image)

        # Assert
        assert len(results) > 0
        assert results[0].name == "TestObject"  # Should be stripped


class TestObjectHeaderDetectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_image(self, mock_ocr_engine):
        """空の画像（矩形なし）の処理をテスト"""
        # Arrange
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert
        assert len(results) == 0

    def test_image_with_noise(self, mock_ocr_engine):
        """ノイズを含む画像の処理をテスト"""
        # Arrange - Create image with small noise dots
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        # Add valid rectangle
        binary[20:50, 100:180] = 0
        grayscale[20:50, 100:180] = 0

        # Add noise (very small dots that should be filtered)
        binary[30, 50] = 0
        binary[35, 60] = 0
        binary[40, 70] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "Object"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert - Should detect only the valid rectangle, not noise
        assert len(results) == 1

    def test_very_narrow_rectangles_filtered(self, mock_ocr_engine):
        """アスペクト比が極端に縦長の矩形がフィルタリングされることをテスト"""
        # Arrange - Create narrow vertical rectangle
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        # Very narrow rectangle: width=20, height=100 (aspect ratio = 0.2)
        binary[20:120, 100:120] = 0
        grayscale[20:120, 100:120] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "Narrow"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert - Should be filtered due to aspect ratio
        assert len(results) == 0

    def test_very_wide_rectangles_filtered(self, mock_ocr_engine):
        """アスペクト比が極端に横長の矩形がフィルタリングされることをテスト"""
        # Arrange - Create very wide rectangle
        grayscale = np.ones((200, 400), dtype=np.uint8) * 255
        binary = np.ones((200, 400), dtype=np.uint8) * 255

        # Very wide rectangle: width=200, height=10 (aspect ratio = 20)
        binary[20:30, 50:250] = 0
        grayscale[20:30, 50:250] = 0

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(200, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "Wide"
        detector = ObjectHeaderDetector(ocr_engine=mock_ocr_engine)

        # Act
        results = detector.detect(preprocessed)

        # Assert - Should be filtered due to aspect ratio
        assert len(results) == 0
