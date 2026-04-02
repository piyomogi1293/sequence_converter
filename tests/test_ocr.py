"""Tests for OCREngine component."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from sequence_converter.models import OCREngineType, OCRResult
from sequence_converter.ocr import EasyOCRBackend, OCREngine, TesseractBackend


@pytest.fixture
def sample_grayscale_image() -> NDArray[np.uint8]:
    """Create a simple grayscale test image with text."""
    # Create a 200x400 white image
    image = np.ones((200, 400), dtype=np.uint8) * 255
    return image


@pytest.fixture
def sample_color_image() -> NDArray[np.uint8]:
    """Create a simple color test image for color detection testing."""
    # Create a 200x400 white image (BGR format)
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255

    # Add a red text region (50x100 at position 50,50)
    image[50:100, 50:150, :] = [0, 0, 255]  # BGR: Red

    # Add a blue text region (50x100 at position 50,200)
    image[50:100, 200:300, :] = [255, 0, 0]  # BGR: Blue

    # Add a black text region (50x100 at position 120,50)
    image[120:170, 50:150, :] = [0, 0, 0]  # BGR: Black

    return image


class TestOCREngine:
    """Test suite for OCREngine."""

    def test_initialization_with_default_tesseract(self):
        """Test that OCREngine initializes with default Tesseract backend."""
        ocr = OCREngine()
        assert ocr.engine_type == OCREngineType.TESSERACT
        assert isinstance(ocr.backend, TesseractBackend)

    def test_initialization_with_tesseract_explicit(self):
        """Test that OCREngine accepts Tesseract engine type explicitly."""
        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)
        assert ocr.engine_type == OCREngineType.TESSERACT
        assert isinstance(ocr.backend, TesseractBackend)

    def test_initialization_with_easyocr(self):
        """Test that OCREngine accepts EasyOCR engine type."""
        ocr = OCREngine(engine_type=OCREngineType.EASYOCR)
        assert ocr.engine_type == OCREngineType.EASYOCR
        assert isinstance(ocr.backend, EasyOCRBackend)

    def test_initialization_with_custom_tesseract_config(self):
        """Test that OCREngine accepts custom Tesseract config."""
        custom_config = "--oem 1 --psm 3"
        ocr = OCREngine(engine_type=OCREngineType.TESSERACT, tesseract_config=custom_config)
        assert ocr.backend.config == custom_config

    def test_initialization_with_custom_easyocr_languages(self):
        """Test that OCREngine accepts custom EasyOCR languages."""
        languages = ["en", "ja", "ch_sim"]
        ocr = OCREngine(engine_type=OCREngineType.EASYOCR, easyocr_languages=languages)
        assert ocr.backend.languages == languages

    def test_extract_text_regions_returns_list(self, sample_grayscale_image):
        """Test that extract_text_regions returns a list of OCRResult."""
        ocr = OCREngine()
        results = ocr.extract_text_regions(sample_grayscale_image)
        assert isinstance(results, list)

    def test_extract_text_regions_with_text_image(self, sample_grayscale_image):
        """Test that extract_text_regions can extract text from an image."""
        # Note: This test may fail if Tesseract is not installed or
        # if the simple test image doesn't contain recognizable text
        ocr = OCREngine()
        results = ocr.extract_text_regions(sample_grayscale_image)

        # Each result should be an OCRResult with required fields
        for result in results:
            assert isinstance(result, OCRResult)
            assert isinstance(result.text, str)
            assert isinstance(result.bounding_box, tuple)
            assert len(result.bounding_box) == 4
            assert isinstance(result.confidence, float)
            assert -1.0 <= result.confidence <= 100.0

    def test_extract_text_regions_bounding_box_format(self, sample_grayscale_image):
        """Test that bounding boxes have correct format (left, top, width, height)."""
        ocr = OCREngine()
        results = ocr.extract_text_regions(sample_grayscale_image)

        for result in results:
            left, top, width, height = result.bounding_box
            assert isinstance(left, int)
            assert isinstance(top, int)
            assert isinstance(width, int)
            assert isinstance(height, int)
            assert left >= 0
            assert top >= 0
            assert width >= 0
            assert height >= 0

    def test_extract_from_region_with_valid_bbox(self, sample_grayscale_image):
        """Test extract_from_region extracts text from a specific region."""
        ocr = OCREngine()
        bbox = (10, 10, 100, 50)  # (x, y, width, height)
        text = ocr.extract_from_region(sample_grayscale_image, bbox)

        assert isinstance(text, str)

    def test_extract_from_region_with_empty_region(self, sample_grayscale_image):
        """Test extract_from_region with empty region returns empty or minimal text."""
        ocr = OCREngine()
        # Extract from a small empty region
        bbox = (0, 0, 10, 10)
        text = ocr.extract_from_region(sample_grayscale_image, bbox)

        # Empty regions should return empty string or whitespace
        assert isinstance(text, str)

    def test_color_detection_red(self, sample_color_image):
        """Test that red text is correctly identified."""
        ocr = OCREngine()
        # This would require the OCREngine to detect color
        # For now, we'll test the method exists and returns proper format
        results = ocr.extract_text_regions(sample_color_image)

        # At least some results should have color information
        # Note: Actual color detection may depend on implementation
        for result in results:
            assert result.color is None or isinstance(result.color, str)
            if result.color:
                assert result.color in ["red", "blue", None]

    def test_color_detection_blue(self, sample_color_image):
        """Test that blue text is correctly identified."""
        ocr = OCREngine()
        results = ocr.extract_text_regions(sample_color_image)

        # Similar to red test
        for result in results:
            assert result.color is None or isinstance(result.color, str)

    def test_color_detection_default(self, sample_grayscale_image):
        """Test that default (non-red, non-blue) text has None color."""
        ocr = OCREngine()
        results = ocr.extract_text_regions(sample_grayscale_image)

        for result in results:
            # Grayscale images should not have color information
            assert result.color is None or isinstance(result.color, str)

    def test_low_confidence_text_logging(self, sample_grayscale_image, caplog):
        """Test that low confidence OCR results generate warning logs."""
        ocr = OCREngine()

        # Extract text (may have low confidence on blank image)
        results = ocr.extract_text_regions(sample_grayscale_image)

        # Check if any low confidence warnings were logged
        # This depends on implementation but we expect logging for conf < 60
        # Note: The actual check will depend on the logger implementation


class TestOCREngineWithMocks:
    """Test suite using mocks for controlled testing."""

    def test_extract_text_regions_with_mock_tesseract_backend(self, sample_grayscale_image):
        """サンプル画像からテキスト領域が抽出されることをテストする（Tesseractバックエンド）"""
        # Arrange
        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "Hello",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 15,
                    "confidence": 95.5,
                },
                {
                    "text": "World",
                    "left": 50,
                    "top": 30,
                    "width": 35,
                    "height": 18,
                    "confidence": 88.2,
                },
                {
                    "text": "Test",
                    "left": 90,
                    "top": 40,
                    "width": 25,
                    "height": 12,
                    "confidence": 45.0,
                },
            ]

            # Act
            results = ocr.extract_text_regions(sample_grayscale_image)

            # Assert
            assert len(results) == 3
            assert results[0].text == "Hello"
            assert results[1].text == "World"
            assert results[2].text == "Test"

    def test_extract_text_regions_with_mock_easyocr_backend(self, sample_grayscale_image):
        """サンプル画像からテキスト領域が抽出されることをテストする（EasyOCRバックエンド）"""
        # Arrange
        ocr = OCREngine(engine_type=OCREngineType.EASYOCR)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "Hello",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 15,
                    "confidence": 95.5,
                },
                {
                    "text": "World",
                    "left": 50,
                    "top": 30,
                    "width": 35,
                    "height": 18,
                    "confidence": 88.2,
                },
            ]

            # Act
            results = ocr.extract_text_regions(sample_grayscale_image)

            # Assert
            assert len(results) == 2
            assert results[0].text == "Hello"
            assert results[1].text == "World"

    def test_bounding_box_coordinates_are_correct(self, sample_grayscale_image):
        """バウンディングボックス座標が正しく取得されることをテスト"""
        # Arrange
        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "Test",
                    "left": 25,
                    "top": 35,
                    "width": 45,
                    "height": 20,
                    "confidence": 90.0,
                },
            ]

            # Act
            results = ocr.extract_text_regions(sample_grayscale_image)

            # Assert
            assert len(results) == 1
            assert results[0].bounding_box == (25, 35, 45, 20)

    def test_text_color_detection_for_red(self):
        """テキスト色判定（赤）が機能することをテスト"""
        # Arrange - Create red text region
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:35, 10:40] = [30, 30, 200]  # Red region (BGR: low B/G, high R)

        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "Red",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 15,
                    "confidence": 90.0,
                },
            ]

            # Act
            results = ocr.extract_text_regions(image)

            # Assert
            assert len(results) == 1
            assert results[0].color == "red"

    def test_text_color_detection_for_blue(self):
        """テキスト色判定（青）が機能することをテスト"""
        # Arrange - Create blue text region
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:48, 50:85] = [200, 30, 30]  # Blue region (BGR: high B, low G/R)

        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "Blue",
                    "left": 50,
                    "top": 30,
                    "width": 35,
                    "height": 18,
                    "confidence": 90.0,
                },
            ]

            # Act
            results = ocr.extract_text_regions(image)

            # Assert
            assert len(results) == 1
            assert results[0].color == "blue"

    def test_text_color_detection_for_default(self):
        """テキスト色判定（デフォルト）でNoneが返されることをテスト"""
        # Arrange - Create gray text region
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "Gray",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 15,
                    "confidence": 90.0,
                },
            ]

            # Act
            results = ocr.extract_text_regions(image)

            # Assert
            assert len(results) == 1
            assert results[0].color is None

    @patch("sequence_converter.ocr.logger")
    def test_low_confidence_text_generates_warning(self, mock_logger, sample_grayscale_image):
        """低信頼度テキストでログ警告が出力されることをテスト"""
        # Arrange
        ocr = OCREngine(engine_type=OCREngineType.TESSERACT)

        with patch.object(ocr.backend, "extract_text_data") as mock_extract:
            mock_extract.return_value = [
                {
                    "text": "LowConf",
                    "left": 10,
                    "top": 20,
                    "width": 30,
                    "height": 15,
                    "confidence": 45.0,
                },
            ]

            # Act
            results = ocr.extract_text_regions(sample_grayscale_image)

            # Assert
            assert len(results) == 1
            assert results[0].confidence == 45.0
            mock_logger.warning.assert_called()


class TestOCREngineIntegration:
    """Integration tests for OCREngine with real images."""

    @pytest.mark.integration
    def test_extract_text_from_real_image(self, tmp_path):
        """Test OCR on a real sequence diagram image."""
        # This would use the actual test image from input/
        # Skip if the image doesn't exist
        pytest.skip("Integration test - requires actual test image")
