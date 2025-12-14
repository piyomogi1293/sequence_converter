"""SelfCallDetector component for detecting self-call circles."""

import logging
from typing import Optional

import cv2
import numpy as np

from sequence_converter.models import PreprocessedImage, SelfCall
from sequence_converter.ocr import OCREngine

logger = logging.getLogger(__name__)


class SelfCallDetector:
    """自己呼び出し検出コンポーネント"""

    LIFELINE_MATCH_TOLERANCE: int = 30  # ピクセル（ライフライン近傍判定の許容範囲）
    TEXT_SEARCH_HORIZONTAL_RANGE: int = 100  # 円の右側テキスト検索の水平範囲（ピクセル）
    TEXT_SEARCH_VERTICAL_RANGE: int = 30  # 円の右側テキスト検索の垂直範囲（ピクセル）

    def __init__(self, ocr_engine: OCREngine):
        """
        Args:
            ocr_engine: OCRエンジンインスタンス（依存性注入）
        """
        self.ocr_engine = ocr_engine

    def detect(self, preprocessed: PreprocessedImage, lifelines: list[int]) -> list[SelfCall]:
        """
        Hough変換で円形状を検出し、ライフライン近傍のものを自己呼び出しとして識別

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト

        Returns:
            list[SelfCall]: 検出された自己呼び出し（Y座標ソート済み）
        """
        if not lifelines:
            return []

        # HoughCirclesで円を検出
        circles = self._detect_circles(preprocessed.grayscale)

        if circles is None or len(circles) == 0:
            return []

        # OCRでテキスト領域を取得
        ocr_results = self.ocr_engine.extract_text_regions(preprocessed.grayscale)

        # 検出された円をSelfCallオブジェクトに変換
        self_calls: list[SelfCall] = []

        for circle in circles[0]:
            x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])

            # 最も近いライフラインを見つける
            matched_lifeline = self._match_to_lifeline(x, lifelines)

            if matched_lifeline is None:
                logger.debug(f"Circle at ({x}, {y}) is too far from any lifeline, skipping")
                continue

            # 円の右側からテキストを検索
            label = self._find_label_text(x, y, radius, ocr_results)

            # SelfCallオブジェクトを作成
            self_call = SelfCall(
                lifeline_x=matched_lifeline,
                y=y,
                label=label,
                confidence=90.0,  # Hough変換の信頼度（固定値）
            )

            self_calls.append(self_call)

        # Y座標順にソート（上から下へ）
        self_calls.sort(key=lambda sc: sc.y)

        logger.info(f"Detected {len(self_calls)} self-call(s)")
        return self_calls

    def _detect_circles(self, grayscale: np.ndarray) -> Optional[np.ndarray]:
        """
        HoughCirclesで円を検出

        Args:
            grayscale: グレースケール画像

        Returns:
            検出された円の配列 [[x, y, radius], ...] または None
        """
        # HoughCirclesのパラメータ
        # dp: 解像度の比率（1 = 入力画像と同じ解像度）
        # minDist: 検出される円の中心間の最小距離
        # param1: Cannyエッジ検出の高い閾値
        # param2: 円検出の閾値（小さいほど多く検出）
        # minRadius, maxRadius: 検出する円の半径範囲
        circles = cv2.HoughCircles(
            grayscale,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=40,
        )

        return circles

    def _match_to_lifeline(self, circle_x: int, lifelines: list[int]) -> Optional[int]:
        """
        円のX座標を最も近いライフラインにマッチング

        Args:
            circle_x: 円の中心X座標
            lifelines: ライフラインX座標リスト

        Returns:
            マッチしたライフラインX座標、またはNone（許容範囲外の場合）
        """
        min_distance = float("inf")
        matched_lifeline = None

        for lifeline_x in lifelines:
            distance = abs(circle_x - lifeline_x)

            if distance < min_distance and distance <= self.LIFELINE_MATCH_TOLERANCE:
                min_distance = distance
                matched_lifeline = lifeline_x

        return matched_lifeline

    def _find_label_text(
        self, circle_x: int, circle_y: int, radius: int, ocr_results: list
    ) -> Optional[str]:
        """
        円の右側からラベルテキストを検索

        Args:
            circle_x: 円の中心X座標
            circle_y: 円の中心Y座標
            radius: 円の半径
            ocr_results: OCR結果リスト

        Returns:
            見つかったラベルテキスト、またはNone
        """
        # 円の右側の検索範囲
        search_x_min = circle_x + radius
        search_x_max = circle_x + radius + self.TEXT_SEARCH_HORIZONTAL_RANGE
        search_y_min = circle_y - self.TEXT_SEARCH_VERTICAL_RANGE
        search_y_max = circle_y + self.TEXT_SEARCH_VERTICAL_RANGE

        # 検索範囲内のテキストを探す
        candidates = []

        for ocr in ocr_results:
            ocr_left, ocr_top, ocr_width, ocr_height = ocr.bounding_box
            ocr_center_x = ocr_left + ocr_width // 2
            ocr_center_y = ocr_top + ocr_height // 2

            # X座標が円の右側の範囲内か
            if search_x_min <= ocr_center_x <= search_x_max:
                # Y座標が円の中心近傍か
                if search_y_min <= ocr_center_y <= search_y_max:
                    # 円の中心からの距離を計算
                    distance = abs(ocr_center_x - circle_x) + abs(ocr_center_y - circle_y)
                    candidates.append((distance, ocr.text))

        # 最も近いテキストを選択
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        return None
