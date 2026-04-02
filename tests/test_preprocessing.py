"""Unit tests for image preprocessing."""

import sys
from pathlib import Path

# テストのためにsrcをsys.pathに追加
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import pytest

from sequence_converter.logger import ImageProcessingError, UnsupportedFormatError
from sequence_converter.models import PreprocessedImage, PreprocessingConfig
from sequence_converter.preprocessing import ImagePreprocessor


class TestImagePreprocessor:
    """ImagePreprocessorのテストクラス"""

    def test_preprocess_png_image(self, tmp_path):
        """PNG画像が正しく読み込まれることをテスト"""
        # RED: テストを先に書く（実装前なので失敗する）
        from PIL import Image

        # テスト用PNG画像を作成
        test_image = Image.new("RGB", (100, 100), color="white")
        image_path = tmp_path / "test.png"
        test_image.save(image_path)

        # 前処理を実行
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(str(image_path))

        # 検証
        assert isinstance(result, PreprocessedImage)
        assert result.grayscale is not None
        assert result.binary is not None
        assert result.original_shape == (100, 100)

    def test_preprocess_jpg_image(self, tmp_path):
        """JPG画像が正しく読み込まれることをテスト"""
        from PIL import Image

        # テスト用JPG画像を作成
        test_image = Image.new("RGB", (100, 100), color="white")
        image_path = tmp_path / "test.jpg"
        test_image.save(image_path)

        # 前処理を実行
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(str(image_path))

        # 検証
        assert isinstance(result, PreprocessedImage)
        assert result.grayscale is not None
        assert result.binary is not None

    def test_unsupported_format_raises_error(self, tmp_path):
        """非サポート形式でValueErrorが発生することをテスト"""
        # テスト用テキストファイルを作成
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")

        # 検証
        preprocessor = ImagePreprocessor()
        with pytest.raises(UnsupportedFormatError):
            preprocessor.preprocess(str(text_file))

    def test_file_not_found_raises_error(self):
        """存在しないファイルでFileNotFoundErrorが発生することをテスト"""
        preprocessor = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess("/nonexistent/file.png")

    def test_preprocessed_image_has_grayscale_and_binary(self, tmp_path):
        """前処理後の画像がgrayscale, binary属性を持つことを検証"""
        from PIL import Image

        # テスト用画像を作成
        test_image = Image.new("RGB", (100, 100), color="white")
        image_path = tmp_path / "test.png"
        test_image.save(image_path)

        # 前処理を実行
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(str(image_path))

        # 検証
        assert hasattr(result, "grayscale")
        assert hasattr(result, "binary")
        assert hasattr(result, "original_shape")
        assert isinstance(result.grayscale, np.ndarray)
        assert isinstance(result.binary, np.ndarray)
        assert result.grayscale.dtype == np.uint8
        assert result.binary.dtype == np.uint8

    def test_custom_preprocessing_config(self, tmp_path):
        """カスタム前処理設定が適用されることをテスト"""
        from PIL import Image

        # テスト用画像を作成
        test_image = Image.new("RGB", (100, 100), color="white")
        image_path = tmp_path / "test.png"
        test_image.save(image_path)

        # カスタム設定で前処理
        config = PreprocessingConfig(
            blur_kernel_size=7, binary_threshold_method="otsu", noise_removal_method="gaussian"
        )
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.preprocess(str(image_path))

        # 検証
        assert isinstance(result, PreprocessedImage)
