"""Tests for ActivationBarDetector component."""

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

import numpy as np
import pytest

from sequence_converter.detectors.activation_bar import ActivationBarDetector
from sequence_converter.models import ActivationBar, PreprocessedImage


class TestActivationBarDetector:
    """ActivationBarDetectorコンポーネントのテストクラス"""

    @pytest.fixture
    def detector(self):
        """ActivationBarDetectorインスタンスを生成"""
        return ActivationBarDetector()

    @pytest.fixture
    def sample_image_with_vertical_rect(self):
        """垂直矩形を含むサンプル画像を生成"""
        # 400x300のグレースケール画像
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # 垂直矩形を描画 (x=100, y=100からy=250まで、width=10)
        cv2.rectangle(grayscale, (100, 100), (110, 250), 0, -1)
        cv2.rectangle(binary, (100, 100), (110, 250), 0, -1)

        return PreprocessedImage(grayscale=grayscale, binary=binary, original_shape=(400, 300))

    def test_detect_vertical_rectangles_near_lifelines(
        self, detector, sample_image_with_vertical_rect
    ):
        """ライフライン近傍の垂直矩形が検出されることをテスト"""
        # Arrange
        lifelines = [105]  # 矩形中心付近

        # Act
        result = detector.detect(sample_image_with_vertical_rect, lifelines)

        # Assert
        assert len(result) > 0, "垂直矩形が検出されること"
        assert isinstance(result[0], ActivationBar), "ActivationBar型であること"

    def test_record_y_coordinate_range(self, detector, sample_image_with_vertical_rect):
        """Y座標範囲が記録されることをテスト"""
        # Arrange
        lifelines = [105]

        # Act
        result = detector.detect(sample_image_with_vertical_rect, lifelines)

        # Assert
        assert len(result) > 0, "アクティベーションバーが検出されること"
        bar = result[0]
        assert bar.y_start >= 0, "y_startが0以上であること"
        assert bar.y_end > bar.y_start, "y_endがy_startより大きいこと"
        # 概ね100〜250の範囲に収まることを確認（多少の誤差を許容）
        assert 90 <= bar.y_start <= 110, "y_startが100付近であること"
        assert 240 <= bar.y_end <= 260, "y_endが250付近であること"

    def test_match_to_nearest_lifeline(self, detector):
        """最も近いライフラインに関連付けられることをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # X=152付近に垂直矩形を描画（ライフライン150に最も近い）
        cv2.rectangle(grayscale, (150, 100), (160, 250), 0, -1)
        cv2.rectangle(binary, (150, 100), (160, 250), 0, -1)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [150, 250]  # 150の方が近い

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) > 0, "アクティベーションバーが検出されること"
        assert result[0].lifeline_x == 150, "最も近いライフライン（150）に関連付けられること"

    def test_high_aspect_ratio_rectangles_detected(self, detector):
        """アスペクト比が高い矩形が検出されることをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # 高いアスペクト比の矩形 (height=200, width=15, ratio=13.3)
        cv2.rectangle(grayscale, (100, 50), (115, 250), 0, -1)
        cv2.rectangle(binary, (100, 50), (115, 250), 0, -1)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [107]

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) > 0, "高アスペクト比の矩形が検出されること"

    def test_wide_rectangles_not_detected(self, detector):
        """横長矩形はアクティベーションバーとして検出されないことをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # 横長矩形 (height=20, width=100, ratio=0.2)
        cv2.rectangle(grayscale, (50, 100), (150, 120), 0, -1)
        cv2.rectangle(binary, (50, 100), (150, 120), 0, -1)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [100]

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) == 0, "横長矩形はアクティベーションバーとして検出されないこと"

    def test_no_vertical_rectangles_returns_empty_list(self, detector):
        """垂直矩形がない場合、空のリストを返すことをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        lifelines = [100]

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert result == [], "垂直矩形がない場合は空のリストを返すこと"

    def test_empty_lifelines_returns_empty_list(self, detector, sample_image_with_vertical_rect):
        """ライフラインリストが空の場合、空のリストを返すことをテスト"""
        # Arrange
        lifelines = []

        # Act
        result = detector.detect(sample_image_with_vertical_rect, lifelines)

        # Assert
        assert result == [], "ライフラインが空の場合は空のリストを返すこと"

    def test_multiple_activation_bars_on_same_lifeline(self, detector):
        """同じライフラインに複数のアクティベーションバーが存在する場合をテスト"""
        # Arrange
        grayscale = np.ones((500, 300), dtype=np.uint8) * 255
        binary = np.ones((500, 300), dtype=np.uint8) * 255

        import cv2

        # 同じライフライン上に2つの垂直矩形
        cv2.rectangle(grayscale, (100, 50), (110, 150), 0, -1)  # 上
        cv2.rectangle(binary, (100, 50), (110, 150), 0, -1)

        cv2.rectangle(grayscale, (100, 250), (110, 400), 0, -1)  # 下
        cv2.rectangle(binary, (100, 250), (110, 400), 0, -1)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(500, 300)
        )

        lifelines = [105]

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) == 2, "2つのアクティベーションバーが検出されること"
        assert all(
            bar.lifeline_x == 105 for bar in result
        ), "すべて同じライフラインに関連付けられること"

    def test_activation_bar_far_from_lifelines_not_detected(self, detector):
        """ライフラインから遠い垂直矩形は検出されないことをテスト"""
        # Arrange
        grayscale = np.ones((400, 300), dtype=np.uint8) * 255
        binary = np.ones((400, 300), dtype=np.uint8) * 255

        import cv2

        # ライフラインから遠い位置（X=50）に垂直矩形を描画
        cv2.rectangle(grayscale, (50, 100), (60, 250), 0, -1)
        cv2.rectangle(binary, (50, 100), (60, 250), 0, -1)

        preprocessed = PreprocessedImage(
            grayscale=grayscale, binary=binary, original_shape=(400, 300)
        )

        # ライフラインはX=200のみ（X=50から150px離れている）
        lifelines = [200]

        # Act
        result = detector.detect(preprocessed, lifelines)

        # Assert
        assert len(result) == 0, "ライフラインから遠い垂直矩形は検出されないこと"

    def test_activation_bar_bounding_box_values(self, detector, sample_image_with_vertical_rect):
        """ActivationBarのフィールド値が正しく設定されることをテスト"""
        # Arrange
        lifelines = [105]

        # Act
        result = detector.detect(sample_image_with_vertical_rect, lifelines)

        # Assert
        assert len(result) > 0, "アクティベーションバーが検出されること"
        bar = result[0]
        assert isinstance(bar.lifeline_x, int), "lifeline_xは整数であること"
        assert isinstance(bar.y_start, int), "y_startは整数であること"
        assert isinstance(bar.y_end, int), "y_endは整数であること"
        assert bar.lifeline_x >= 0, "lifeline_xは0以上であること"
        assert bar.y_start >= 0, "y_startは0以上であること"
        assert bar.y_end >= 0, "y_endは0以上であること"
