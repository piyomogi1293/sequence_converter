"""Tests for NoteAnnotationDetector component."""

import sys
from pathlib import Path

# Force src to be at position 0 to avoid test directory shadowing
src_path = Path(__file__).parent.parent.parent / "src"
src_str = str(src_path)
# Remove if already present
if src_str in sys.path:
    sys.path.remove(src_str)
# Insert at position 0
sys.path.insert(0, src_str)

from unittest.mock import Mock

import numpy as np
import pytest

from sequence_converter.detectors.note_annotation import NoteAnnotationDetector
from sequence_converter.models import NoteAnnotation, OCRResult, PreprocessedImage
from sequence_converter.ocr import OCREngine


class TestNoteAnnotationDetector:
    """NoteAnnotationDetectorコンポーネントのテストクラス"""

    @pytest.fixture
    def mock_ocr_engine(self):
        """モックOCRエンジンを生成"""
        return Mock(spec=OCREngine)

    @pytest.fixture
    def detector(self, mock_ocr_engine):
        """NoteAnnotationDetectorインスタンスを生成"""
        return NoteAnnotationDetector(mock_ocr_engine)

    @pytest.fixture
    def sample_image_with_rectangle(self):
        """矩形を含むサンプル画像を生成"""
        # 400x300のグレースケール画像
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # 矩形を描画 (x=50, y=100, width=100, height=60)
        cv2.rectangle(grayscale, (50, 100), (150, 160), 0, 2)
        cv2.rectangle(binary, (50, 100), (150, 160), 0, 2)

        return PreprocessedImage(grayscale=grayscale, binary=binary, original_shape=(400, 300))

    def test_detect_rectangles_overlapping_or_independent_of_lifelines(
        self, detector, sample_image_with_rectangle, mock_ocr_engine
    ):
        """ライフラインと重なる／独立する矩形が検出されることをテスト"""
        # Arrange
        lifelines = [100, 200]
        # 矩形内にテキストが存在する
        mock_ocr_engine.extract_from_region.return_value = "Note content"

        # Act
        result = detector.detect(sample_image_with_rectangle, lifelines)

        # Assert
        assert len(result) > 0, "矩形が検出されること"
        assert isinstance(result[0], NoteAnnotation), "NoteAnnotation型であること"

    def test_extract_text_content_using_ocr(
        self, detector, sample_image_with_rectangle, mock_ocr_engine
    ):
        """ノート内のテキストがOCRで抽出されることをテスト"""
        # Arrange
        lifelines = [100, 200]
        expected_text = "Important note content"
        mock_ocr_engine.extract_from_region.return_value = expected_text

        # Act
        result = detector.detect(sample_image_with_rectangle, lifelines)

        # Assert
        assert len(result) > 0, "ノート注釈が検出されること"
        assert result[0].text == expected_text, "OCRでテキストが抽出されること"
        # extract_from_regionが呼ばれたことを確認
        assert mock_ocr_engine.extract_from_region.called, "OCRエンジンが呼ばれること"

    def test_associate_note_with_nearest_lifeline(self, detector, mock_ocr_engine):
        """ノートが最も近いライフラインに関連付けられることをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # X=110付近に矩形を描画（ライフライン100に最も近い）
        cv2.rectangle(grayscale, (105, 100), (205, 160), 0, 2)
        cv2.rectangle(binary, (105, 100), (205, 160), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [100, 250]  # 100の方が近い
        mock_ocr_engine.extract_from_region.return_value = "Note text"

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) > 0, "ノートが検出されること"
        assert result[0].related_lifeline == 100, "最も近いライフライン（100）に関連付けられること"

    def test_sort_results_by_y_coordinate(self, detector, mock_ocr_engine):
        """検出結果がY座標順にソートされることをテスト"""
        # Arrange
        grayscale = np.ones((500, 300), dtype=np.uint8) * 255
        binary = np.ones((500, 300), dtype=np.uint8) * 255

        import cv2

        # 3つの矩形を異なるY座標に描画
        cv2.rectangle(grayscale, (50, 300), (150, 360), 0, 2)  # 下
        cv2.rectangle(binary, (50, 300), (150, 360), 0, 2)

        cv2.rectangle(grayscale, (50, 50), (150, 110), 0, 2)  # 上
        cv2.rectangle(binary, (50, 50), (150, 110), 0, 2)

        cv2.rectangle(grayscale, (50, 180), (150, 240), 0, 2)  # 中
        cv2.rectangle(binary, (50, 180), (150, 240), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 300)
        )

        lifelines = [100]
        mock_ocr_engine.extract_from_region.return_value = "Note"

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) == 3, "3つのノートが検出されること"
        # Y座標はbounding_boxの2番目の要素（top）
        y_coords = [note.bounding_box[1] for note in result]
        assert y_coords == sorted(y_coords), "Y座標順（上から下へ）にソートされること"

    def test_no_rectangles_detected_returns_empty_list(self, detector, mock_ocr_engine):
        """矩形が検出されない場合、空のリストを返すことをテスト"""
        # Arrange
        # 矩形のない画像
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [100]

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert result == [], "矩形がない場合は空のリストを返すこと"

    def test_rectangles_without_text_are_ignored(self, detector, mock_ocr_engine):
        """テキストコンテンツがない矩形は無視されることをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # 矩形を描画
        cv2.rectangle(grayscale, (50, 100), (150, 160), 0, 2)
        cv2.rectangle(binary, (50, 100), (150, 160), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [100]
        # OCRが空文字列またはホワイトスペースのみを返す
        mock_ocr_engine.extract_from_region.return_value = "   "

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert result == [], "テキストがない矩形は検出されないこと"

    def test_empty_lifelines_list(self, detector, sample_image_with_rectangle, mock_ocr_engine):
        """ライフラインリストが空の場合でもノートを検出できることをテスト"""
        # Arrange
        lifelines = []
        mock_ocr_engine.extract_from_region.return_value = "Note content"

        # Act
        result = detector.detect(sample_image_with_rectangle, lifelines)

        # Assert
        # ライフラインが空でもノートは検出される（related_lifelineはNone）
        assert len(result) > 0, "ライフラインが空でもノートは検出されること"
        assert result[0].related_lifeline is None, "related_lifelineはNoneであること"

    def test_note_annotation_has_correct_bounding_box(
        self, detector, sample_image_with_rectangle, mock_ocr_engine
    ):
        """NoteAnnotationのbounding_boxが正しく設定されることをテスト"""
        # Arrange
        lifelines = [100]
        mock_ocr_engine.extract_from_region.return_value = "Note"

        # Act
        result = detector.detect(sample_image_with_rectangle, lifelines)

        # Assert
        assert len(result) > 0, "ノートが検出されること"
        bbox = result[0].bounding_box
        assert isinstance(bbox, tuple), "bounding_boxはタプルであること"
        assert len(bbox) == 4, "bounding_boxは4要素のタプルであること (x, y, width, height)"
        assert all(isinstance(v, (int, np.integer)) for v in bbox), "すべての要素が整数であること"

    def test_confidence_is_set_correctly(
        self, detector, sample_image_with_rectangle, mock_ocr_engine
    ):
        """信頼度が正しく設定されることをテスト"""
        # Arrange
        lifelines = [100]
        mock_ocr_engine.extract_from_region.return_value = "Note content"

        # Act
        result = detector.detect(sample_image_with_rectangle, lifelines)

        # Assert
        assert len(result) > 0, "ノートが検出されること"
        assert -1.0 <= result[0].confidence <= 100.0, "信頼度が-1.0から100.0の範囲であること"
