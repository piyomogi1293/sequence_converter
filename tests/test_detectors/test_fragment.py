"""Tests for FragmentDetector component."""

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

from sequence_converter.detectors.fragment import FragmentDetector
from sequence_converter.models import Fragment, FragmentType, PreprocessedImage
from sequence_converter.ocr import OCREngine


class TestFragmentDetector:
    """FragmentDetectorコンポーネントのテストクラス"""

    @pytest.fixture
    def mock_ocr_engine(self):
        """モックOCRエンジンを生成"""
        return Mock(spec=OCREngine)

    @pytest.fixture
    def detector(self, mock_ocr_engine):
        """FragmentDetectorインスタンスを生成"""
        return FragmentDetector(mock_ocr_engine)

    @pytest.fixture
    def sample_image_with_large_rect(self):
        """大きな矩形フレームを含むサンプル画像を生成"""
        # 500x400のグレースケール画像
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        # 大きな矩形フレーム (x=50, y=100, width=300, height=200)
        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        return PreprocessedImage(grayscale=grayscale, binary=binary, original_shape=(500, 400))

    def test_detect_large_rectangular_frames(
        self, detector, sample_image_with_large_rect, mock_ocr_engine
    ):
        """大きな矩形フレームが検出されることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "alt"

        # Act
        result = detector.detect(sample_image_with_large_rect)

        # Assert
        assert len(result) > 0, "大きな矩形フレームが検出されること"
        assert isinstance(result[0], Fragment), "Fragment型であること"

    def test_extract_label_from_top_left_corner(
        self, detector, sample_image_with_large_rect, mock_ocr_engine
    ):
        """左上隅の領域からラベルが抽出されることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "loop"

        # Act
        result = detector.detect(sample_image_with_large_rect)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        # OCRエンジンが左上隅の領域で呼ばれたことを確認
        assert mock_ocr_engine.extract_from_region.called, "OCRエンジンが呼ばれること"

    def test_identify_fragment_type_alt(self, detector, mock_ocr_engine):
        """フラグメントタイプ'alt'が識別されることをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "alt condition"

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert result[0].type == FragmentType.ALT, "タイプが'alt'として識別されること"

    def test_identify_fragment_type_loop(self, detector, mock_ocr_engine):
        """フラグメントタイプ'loop'が識別されることをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "loop until done"

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert result[0].type == FragmentType.LOOP, "タイプが'loop'として識別されること"

    def test_identify_fragment_type_opt(self, detector, mock_ocr_engine):
        """フラグメントタイプ'opt'が識別されることをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "opt"

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert result[0].type == FragmentType.OPT, "タイプが'opt'として識別されること"

    def test_identify_fragment_type_par(self, detector, mock_ocr_engine):
        """フラグメントタイプ'par'が識別されることをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "par"

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert result[0].type == FragmentType.PAR, "タイプが'par'として識別されること"

    def test_no_label_treated_as_group(self, detector, mock_ocr_engine):
        """ラベルがない場合、グループとして処理されることをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        # OCRが空文字列またはラベルなしを返す
        mock_ocr_engine.extract_from_region.return_value = ""

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert result[0].type == FragmentType.GROUP, "ラベルがない場合はGROUPとして処理されること"

    def test_extract_title_text(self, detector, mock_ocr_engine):
        """タイトルテキストが抽出されることをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        cv2.rectangle(grayscale, (50, 100), (350, 300), 0, 2)
        cv2.rectangle(binary, (50, 100), (350, 300), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "alt [condition is true]"

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert result[0].title is not None, "タイトルが設定されること"
        # タイトルにはラベル以降のテキストが含まれることを確認
        assert "[condition is true]" in result[0].title, "タイトルテキストが含まれること"

    def test_sort_results_by_y_coordinate(self, detector, mock_ocr_engine):
        """検出結果がY座標順にソートされることをテスト"""
        # Arrange
        grayscale = np.ones((600, 400), dtype=np.uint8) * 255
        binary = np.ones((600, 400), dtype=np.uint8) * 255

        import cv2

        # 3つの矩形フレームを異なるY座標に描画
        cv2.rectangle(grayscale, (50, 400), (350, 550), 0, 2)  # 下
        cv2.rectangle(binary, (50, 400), (350, 550), 0, 2)

        cv2.rectangle(grayscale, (50, 50), (350, 150), 0, 2)  # 上
        cv2.rectangle(binary, (50, 50), (350, 150), 0, 2)

        cv2.rectangle(grayscale, (50, 220), (350, 350), 0, 2)  # 中
        cv2.rectangle(binary, (50, 220), (350, 350), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(600, 400)
        )

        mock_ocr_engine.extract_from_region.return_value = "alt"

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) == 3, "3つのフラグメントが検出されること"
        # Y座標はbounding_boxの2番目の要素（top）
        y_coords = [frag.bounding_box[1] for frag in result]
        assert y_coords == sorted(y_coords), "Y座標順（上から下へ）にソートされること"

    def test_no_large_rectangles_returns_empty_list(self, detector, mock_ocr_engine):
        """大きな矩形がない場合、空のリストを返すことをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert result == [], "大きな矩形がない場合は空のリストを返すこと"

    def test_small_rectangles_not_detected_as_fragments(self, detector, mock_ocr_engine):
        """小さな矩形はフラグメントとして検出されないことをテスト"""
        # Arrange
        grayscale = np.ones((500, 400), dtype=np.uint8) * 255
        binary = np.ones((500, 400), dtype=np.uint8) * 255

        import cv2

        # 小さな矩形 (width=50, height=30)
        cv2.rectangle(grayscale, (100, 100), (150, 130), 0, 2)
        cv2.rectangle(binary, (100, 100), (150, 130), 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 400)
        )

        # Act
        result = detector.detect(preprocessed)

        # Assert
        assert len(result) == 0, "小さな矩形はフラグメントとして検出されないこと"

    def test_fragment_bounding_box_values(
        self, detector, sample_image_with_large_rect, mock_ocr_engine
    ):
        """Fragmentのbounding_box値が正しく設定されることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "alt"

        # Act
        result = detector.detect(sample_image_with_large_rect)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        frag = result[0]
        assert isinstance(frag.bounding_box, tuple), "bounding_boxはタプルであること"
        assert len(frag.bounding_box) == 4, "bounding_boxは4要素であること (x, y, width, height)"
        assert all(
            isinstance(v, (int, np.integer)) for v in frag.bounding_box
        ), "すべての要素が整数であること"

    def test_confidence_is_set(self, detector, sample_image_with_large_rect, mock_ocr_engine):
        """信頼度が設定されることをテスト"""
        # Arrange
        mock_ocr_engine.extract_from_region.return_value = "alt"

        # Act
        result = detector.detect(sample_image_with_large_rect)

        # Assert
        assert len(result) > 0, "フラグメントが検出されること"
        assert -1.0 <= result[0].confidence <= 100.0, "信頼度が-1.0から100.0の範囲であること"
