"""Image preprocessing module."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .logger import ImageProcessingError, UnsupportedFormatError, setup_logger
from .models import PreprocessedImage, PreprocessingConfig

logger = setup_logger(__name__)


class ImagePreprocessor:
    """画像前処理コンポーネント"""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        """
        Args:
            config: 前処理設定（デフォルト値で初期化）
        """
        self.config = config

    def _enhance_lines(
        self, grayscale: NDArray[np.uint8], output_dir: Path | None, base_name: str | None
    ) -> NDArray[np.uint8]:
        """
        ライフラインと矢印の線を太くする処理

        戦略:
        1. 適応的二値化で線と文字を分離（局所的な照明条件に対応）
        2. 水平・垂直線のみを検出（文字は複雑な形状なので除外）
        3. 線のみにdilationを適用
        4. 元画像と合成して線だけを太くする

        Args:
            grayscale: グレースケール画像
            output_dir: 中間画像の出力ディレクトリ（Noneの場合は保存しない）
            base_name: 中間画像のベースファイル名（Noneの場合は保存しない）

        Returns:
            線を強調したグレースケール画像
        """
        # Otsu法による二値化
        # THRESH_BINARY_INV: 黒い線を白に、白い背景を黒に反転
        _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        logger.debug("Applied Otsu threshold for binarization")

        # 中間画像保存: 二値化（反転）
        if output_dir and base_name:
            binary_inv_path = output_dir / f"{base_name}_03_binary_inv.png"
            cv2.imwrite(str(binary_inv_path), binary)
            logger.debug(f"Saved intermediate image: {binary_inv_path}")

        # 水平線を検出（ライフラインと矢印の水平部分）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # 垂直線を検出（ライフラインの垂直部分）
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # 斜め線を検出（矢印の先端など）- より小さいカーネルで検出
        diagonal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        diagonal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, diagonal_kernel)

        # すべての線を統合
        all_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
        all_lines = cv2.bitwise_or(all_lines, diagonal_lines)

        # 中間画像保存: 線検出結果
        if output_dir and base_name:
            lines_path = output_dir / f"{base_name}_04_detected_lines.png"
            cv2.imwrite(str(lines_path), all_lines)
            logger.debug(f"Saved intermediate image: {lines_path}")

        # 線のみをdilation（太く）する
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_lines = cv2.dilate(all_lines, dilation_kernel, iterations=2)

        # 中間画像保存: dilation後の線
        if output_dir and base_name:
            dilated_path = output_dir / f"{base_name}_05_dilated_lines.png"
            cv2.imwrite(str(dilated_path), dilated_lines)
            logger.debug(f"Saved intermediate image: {dilated_path}")

        # 元画像に線を合成（線の部分を黒く、それ以外はグレースケールのまま）
        enhanced = grayscale.copy()
        enhanced[dilated_lines > 0] = 0  # 線の部分を黒に

        # 中間画像保存: 最終結果
        if output_dir and base_name:
            enhanced_path = output_dir / f"{base_name}_06_enhanced.png"
            cv2.imwrite(str(enhanced_path), enhanced)
            logger.debug(f"Saved intermediate image: {enhanced_path}")

        logger.debug("Applied line enhancement (dilation)")

        return enhanced

    def preprocess(self, image_path: str) -> PreprocessedImage:
        """
        画像を読み込み、グレースケール・二値化・ノイズ除去を実行

        Args:
            image_path: 入力画像ファイルパス（PNG/JPG）

        Returns:
            PreprocessedImage: 前処理済み画像データ

        Raises:
            FileNotFoundError: ファイルが存在しない
            UnsupportedFormatError: サポートされていないフォーマット
            ImageProcessingError: 画像処理が失敗
        """
        # ファイルの存在確認
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {image_path}")

        # フォーマット確認
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            # 中間画像保存用のディレクトリを準備
            if self.config.save_intermediate_images and self.config.intermediate_output_dir:
                output_dir = Path(self.config.intermediate_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                base_name = path.stem

            # 画像読み込み（PILで読み込んでからOpenCVに変換）
            pil_image = Image.open(image_path)

            # 透過背景を白背景に変換
            if pil_image.mode in ("RGBA", "LA", "P"):
                # 白背景のRGB画像を作成
                background = Image.new("RGB", pil_image.size, (255, 255, 255))

                # アルファチャンネルがある場合はマスクとして使用
                if pil_image.mode == "RGBA":
                    background.paste(pil_image, mask=pil_image.split()[3])
                elif pil_image.mode == "LA":
                    background.paste(pil_image, mask=pil_image.split()[1])
                elif pil_image.mode == "P" and "transparency" in pil_image.info:
                    # パレットモードで透過情報がある場合
                    pil_image = pil_image.convert("RGBA")
                    background.paste(pil_image, mask=pil_image.split()[3])
                else:
                    background.paste(pil_image)

                pil_image = background
                logger.debug(f"Converted transparent background to white background")

            image_rgb = np.array(pil_image.convert("RGB"))
            original_shape = image_rgb.shape[:2]  # (height, width)

            # 中間画像保存: RGB変換後
            if self.config.save_intermediate_images and self.config.intermediate_output_dir:
                rgb_path = output_dir / f"{base_name}_01_rgb.png"
                Image.fromarray(image_rgb).save(rgb_path)
                logger.debug(f"Saved intermediate image: {rgb_path}")

            # OpenCV形式に変換（RGB -> BGR）
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # グレースケール変換
            grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            logger.debug(f"Converted to grayscale: shape={grayscale.shape}")

            # 中間画像保存: グレースケール
            if self.config.save_intermediate_images and self.config.intermediate_output_dir:
                grayscale_path = output_dir / f"{base_name}_02_grayscale.png"
                cv2.imwrite(str(grayscale_path), grayscale)
                logger.debug(f"Saved intermediate image: {grayscale_path}")

            # TODO: ノイズ除去処理はOCR時のみ適用するように変更する
            # 現在は矢印などの細い線が消えてしまうため、前処理パイプラインからは削除
            # OCR精度向上のため、OCR実行時にのみdenoise処理を適用すること

            # TODO: 二値化処理は検出タイプごとに選択的に適用するように変更する
            # 理由:
            # - 矢印検出: グレースケールのまま処理する方が細い線を保持できる
            # - OCR: 二値化により文字認識精度が向上する
            # - ライフライン検出: グレースケールまたは適応的二値化が必要
            # 現在はグレースケール画像をそのまま使用し、必要に応じて各検出器で二値化を行う

            # ライフラインと矢印の線を太くする処理
            # 戦略: 細い線形状のみを検出してdilationを適用し、文字は保持する
            enhanced = self._enhance_lines(
                grayscale,
                output_dir if self.config.save_intermediate_images else None,
                base_name if self.config.save_intermediate_images else None,
            )

            logger.info(f"Preprocessing completed for {image_path}")

            return PreprocessedImage(
                grayscale=enhanced, binary=enhanced, original_shape=original_shape
            )

        except Exception as e:
            if isinstance(e, (FileNotFoundError, UnsupportedFormatError)):
                raise
            raise ImageProcessingError(f"Failed to process image: {e}") from e
