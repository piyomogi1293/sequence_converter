"""Activation bar detector component."""

import logging
from typing import Optional

import cv2
import numpy as np

from sequence_converter.models import ActivationBar, PreprocessedImage

logger = logging.getLogger(__name__)


class ActivationBarDetector:
    """アクティベーションバー検出コンポーネント"""

    MIN_ASPECT_RATIO: float = 2.0  # 最小アスペクト比 (height/width)
    MIN_HEIGHT: int = 30  # 最小高さ（ピクセル）
    MAX_WIDTH: int = 30  # 最大幅（ピクセル）
    LIFELINE_MATCH_TOLERANCE: int = 50  # ライフライン近傍判定の許容範囲（ピクセル）

    def detect(self, preprocessed: PreprocessedImage, lifelines: list[int]) -> list[ActivationBar]:
        """
        ライフライン近傍の垂直矩形を検出

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト

        Returns:
            list[ActivationBar]: 検出されたアクティベーションバー
        """
        if not lifelines:
            logger.info("No lifelines provided, skipping activation bar detection")
            return []

        # findContoursで矩形を検出
        vertical_rects = self._detect_vertical_rectangles(preprocessed.binary)

        if not vertical_rects:
            logger.info("No vertical rectangles detected for activation bars")
            return []

        # 検出された垂直矩形をActivationBarオブジェクトに変換
        activation_bars: list[ActivationBar] = []

        for rect in vertical_rects:
            x, y, width, height = rect

            # 矩形の中心X座標を計算
            rect_center_x = x + width // 2

            # 最も近いライフラインを見つける
            matched_lifeline = self._match_to_lifeline(rect_center_x, lifelines)

            if matched_lifeline is None:
                logger.debug(f"Vertical rectangle at x={x} is too far from any lifeline, skipping")
                continue

            # ActivationBarオブジェクトを作成
            activation_bar = ActivationBar(lifeline_x=matched_lifeline, y_start=y, y_end=y + height)

            activation_bars.append(activation_bar)

        logger.info(f"Detected {len(activation_bars)} activation bar(s)")
        return activation_bars

    def _detect_vertical_rectangles(
        self, binary_image: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        findContoursで垂直矩形を検出

        Args:
            binary_image: 二値化画像

        Returns:
            list[tuple[int, int, int, int]]: 検出された垂直矩形のリスト [(x, y, width, height), ...]
        """
        # 二値画像を反転（黒い図形を検出するため）
        # シーケンス図は通常、白地に黒い線/図形なので、findContoursで検出するには反転が必要
        inverted = cv2.bitwise_not(binary_image)

        # 輪郭検出
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vertical_rects = []

        for contour in contours:
            # バウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(contour)

            # 高さフィルタリング
            if h < self.MIN_HEIGHT:
                continue

            # 幅フィルタリング
            if w > self.MAX_WIDTH:
                continue

            # アスペクト比チェック（height/width）
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < self.MIN_ASPECT_RATIO:
                continue

            vertical_rects.append((x, y, w, h))

        logger.debug(f"Detected {len(vertical_rects)} potential vertical rectangles")
        return vertical_rects

    def _match_to_lifeline(self, rect_center_x: int, lifelines: list[int]) -> Optional[int]:
        """
        矩形に最も近いライフラインを見つける

        Args:
            rect_center_x: 矩形の中心X座標
            lifelines: ライフラインX座標リスト

        Returns:
            Optional[int]: 最も近いライフラインのX座標、またはNone（許容範囲外の場合）
        """
        if not lifelines:
            return None

        # 最も近いライフラインを見つける
        nearest_lifeline = min(lifelines, key=lambda ll: abs(ll - rect_center_x))

        # 許容範囲内かチェック
        distance = abs(nearest_lifeline - rect_center_x)
        if distance <= self.LIFELINE_MATCH_TOLERANCE:
            return nearest_lifeline

        # 許容範囲外の場合はNoneを返す
        logger.debug(
            f"Rectangle at x={rect_center_x} is {distance}px from nearest lifeline, "
            f"beyond tolerance ({self.LIFELINE_MATCH_TOLERANCE}px)"
        )
        return None
