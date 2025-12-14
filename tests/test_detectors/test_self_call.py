"""Tests for SelfCallDetector component."""

import sys
from pathlib import Path

# テストのためにsrcをsys.pathに追加
test_dir = Path(__file__).parent.parent
project_root = test_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from sequence_converter.detectors.self_call import SelfCallDetector
from sequence_converter.models import OCRResult, PreprocessedImage, SelfCall
from sequence_converter.ocr import OCREngine


class TestSelfCallDetector:
    """SelfCallDetectorコンポーネントのテストクラス"""

    @pytest.fixture
    def mock_ocr_engine(self):
        """モックOCRエンジンを生成"""
        return Mock(spec=OCREngine)

    @pytest.fixture
    def detector(self, mock_ocr_engine):
        """SelfCallDetectorインスタンスを生成"""
        return SelfCallDetector(mock_ocr_engine)

    @pytest.fixture
    def sample_image(self):
        """サンプル画像を生成"""
        # 300x300のグレースケール画像
        grayscale = np.ones((300, 300), dtype=np.uint8) * 255
        binary = np.ones((300, 300), dtype=np.uint8) * 255

        # 中央に円を描画 (center=(150, 150), radius=20)
        import cv2

        cv2.circle(grayscale, (150, 150), 20, 0, 2)
        cv2.circle(binary, (150, 150), 20, 0, 2)

        return PreprocessedImage(grayscale=grayscale, binary=binary, original_shape=(300, 300))

    def test_detect_circles_using_hough_transform(self, detector, sample_image, mock_ocr_engine):
        """Hough変換で円形状が検出されることをテスト"""
        # Arrange
        lifelines = [150]  # 円の中心X座標と一致
        mock_ocr_engine.extract_text_regions.return_value = []

        # Act
        result = detector.detect(sample_image, lifelines)

        # Assert
        assert len(result) > 0, "円形状が検出されること"
        assert result[0].lifeline_x == 150, "ライフラインX座標が正しく設定されること"

    def test_match_circle_to_nearby_lifeline(self, detector, mock_ocr_engine):
        """円の中心X座標がライフライン近傍であることを判定できることをテスト"""
        # Arrange
        grayscale = np.ones((300, 300), dtype=np.uint8) * 255
        binary = np.ones((300, 300), dtype=np.uint8) * 255

        import cv2

        # X=155に円を描画（ライフライン150から5px離れている）
        cv2.circle(grayscale, (155, 150), 20, 0, 2)
        cv2.circle(binary, (155, 150), 20, 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(300, 300)
        )

        lifelines = [150, 250]  # 複数のライフライン
        mock_ocr_engine.extract_text_regions.return_value = []

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) > 0, "円が検出されること"
        assert result[0].lifeline_x == 150, "最も近いライフライン（150）にマッチすること"

    def test_extract_label_from_right_side_of_circle(self, detector, sample_image, mock_ocr_engine):
        """円の右側テキストが自己呼び出しラベルとして取得されることをテスト"""
        # Arrange
        lifelines = [150]

        # 円の右側にテキストを配置
        mock_ocr_engine.extract_text_regions.return_value = [
            OCRResult(
                text="Self Call Label",
                bounding_box=(180, 145, 100, 10),  # 円の右側 (x=180)
                confidence=95.0,
                color=None,
            ),
            OCRResult(
                text="Other Text",
                bounding_box=(50, 145, 80, 10),  # 円の左側 (x=50)
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = detector.detect(sample_image, lifelines)

        # Assert
        assert len(result) > 0, "自己呼び出しが検出されること"
        assert (
            result[0].label == "Self Call Label"
        ), "円の右側のテキストがラベルとして取得されること"

    def test_sort_results_by_y_coordinate(self, detector, mock_ocr_engine):
        """検出結果がY座標順にソートされることをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # 3つの円を異なるY座標に描画
        cv2.circle(grayscale, (150, 300), 20, 0, 2)  # 下
        cv2.circle(binary, (150, 300), 20, 0, 2)

        cv2.circle(grayscale, (150, 100), 20, 0, 2)  # 上
        cv2.circle(binary, (150, 100), 20, 0, 2)

        cv2.circle(grayscale, (150, 200), 20, 0, 2)  # 中
        cv2.circle(binary, (150, 200), 20, 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [150]
        mock_ocr_engine.extract_text_regions.return_value = []

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) == 3, "3つの円が検出されること"
        assert result[0].y < result[1].y < result[2].y, "Y座標順（上から下へ）にソートされること"

    def test_no_circles_detected(self, detector, mock_ocr_engine):
        """円が検出されない場合、空のリストを返すことをテスト"""
        # Arrange
        # 円のない画像
        grayscale = np.ones((300, 300), dtype=np.uint8) * 255
        binary = np.ones((300, 300), dtype=np.uint8) * 255

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(300, 300)
        )

        lifelines = [150]
        mock_ocr_engine.extract_text_regions.return_value = []

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert result == [], "円がない場合は空のリストを返すこと"

    def test_circle_far_from_lifelines_not_detected(self, detector, mock_ocr_engine):
        """ライフラインから遠い円は検出されないことをテスト"""
        # Arrange
        grayscale = np.ones((300, 300), dtype=np.uint8) * 255
        binary = np.ones((300, 300), dtype=np.uint8) * 255

        import cv2

        # ライフラインから遠い位置（X=50）に円を描画
        cv2.circle(grayscale, (50, 150), 20, 0, 2)
        cv2.circle(binary, (50, 150), 20, 0, 2)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(300, 300)
        )

        # ライフラインはX=150と250のみ（X=50から遠い）
        lifelines = [150, 250]
        mock_ocr_engine.extract_text_regions.return_value = []

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        # ライフラインから50px離れているため検出されないはず
        assert len(result) == 0, "ライフラインから遠い円は検出されないこと"

    def test_empty_lifelines_list(self, detector, sample_image, mock_ocr_engine):
        """ライフラインリストが空の場合、空のリストを返すことをテスト"""
        # Arrange
        lifelines = []
        mock_ocr_engine.extract_text_regions.return_value = []

        # Act
        result = detector.detect(sample_image, lifelines)

        # Assert
        assert result == [], "ライフラインが空の場合は空のリストを返すこと"

    def test_label_none_when_no_text_nearby(self, detector, sample_image, mock_ocr_engine):
        """円の近くにテキストがない場合、labelがNoneになることをテスト"""
        # Arrange
        lifelines = [150]
        # 円から遠い位置のテキストのみ
        mock_ocr_engine.extract_text_regions.return_value = [
            OCRResult(
                text="Far Text",
                bounding_box=(50, 50, 80, 10),  # 円から遠い位置
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = detector.detect(sample_image, lifelines)

        # Assert
        assert len(result) > 0, "円は検出されること"
        assert result[0].label is None, "近くにテキストがない場合、labelはNoneであること"
