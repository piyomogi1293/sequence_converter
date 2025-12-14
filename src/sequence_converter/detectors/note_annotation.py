"""Note annotation detector component."""

import logging
from typing import Optional

import cv2
import numpy as np

from sequence_converter.models import NoteAnnotation, PreprocessedImage
from sequence_converter.ocr import OCREngine

logger = logging.getLogger(__name__)


class NoteAnnotationDetector:
    """ノート注釈検出コンポーネント"""

    MIN_RECT_AREA: int = 500  # 最小矩形面積（ピクセル²）
    MAX_RECT_AREA: int = 50000  # 最大矩形面積（ピクセル²）
    LIFELINE_MATCH_TOLERANCE: int = 100  # ライフライン関連付けの許容範囲（ピクセル）

    def __init__(self, ocr_engine: OCREngine):
        """
        Args:
            ocr_engine: OCRエンジンインスタンス（依存性注入）
        """
        self.ocr_engine = ocr_engine

    def detect(self, preprocessed: PreprocessedImage, lifelines: list[int]) -> list[NoteAnnotation]:
        """
        矩形領域を検出し、重要なテキストを含むものをノートとして識別

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト

        Returns:
            list[NoteAnnotation]: 検出されたノート注釈（Y座標ソート済み）
        """
        # findContoursで矩形を検出
        rectangles = self._detect_rectangles(preprocessed.binary)

        if not rectangles:
            logger.info("No rectangles detected for note annotations")
            return []

        # 検出された矩形からノートを識別
        notes: list[NoteAnnotation] = []

        for rect in rectangles:
            x, y, width, height = rect

            # OCRでテキストを抽出
            text = self.ocr_engine.extract_from_region(
                preprocessed.grayscale, (x, y, width, height)
            )

            # テキストが空またはホワイトスペースのみの場合はスキップ
            if not text or text.strip() == "":
                logger.debug(f"Rectangle at ({x}, {y}) has no text content, skipping")
                continue

            # 最も近いライフラインを見つける
            related_lifeline = self._find_nearest_lifeline(x, width, lifelines)

            # NoteAnnotationオブジェクトを作成
            note = NoteAnnotation(
                bounding_box=(x, y, width, height),
                text=text.strip(),
                related_lifeline=related_lifeline,
                confidence=85.0,  # OCR信頼度（固定値）
            )

            notes.append(note)

        # Y座標順にソート（上から下へ）
        notes.sort(key=lambda n: n.bounding_box[1])

        logger.info(f"Detected {len(notes)} note annotation(s)")
        return notes

    def _detect_rectangles(self, binary_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        findContoursで矩形を検出

        Args:
            binary_image: 二値化画像

        Returns:
            list[tuple[int, int, int, int]]: 検出された矩形のリスト [(x, y, width, height), ...]
        """
        # 輪郭検出
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []

        for contour in contours:
            # バウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # 面積フィルタリング
            if area < self.MIN_RECT_AREA or area > self.MAX_RECT_AREA:
                continue

            # アスペクト比チェック（極端に細長い矩形を除外）
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 10.0:
                continue

            rectangles.append((x, y, w, h))

        logger.debug(f"Detected {len(rectangles)} potential note rectangles")
        return rectangles

    def _find_nearest_lifeline(self, x: int, width: int, lifelines: list[int]) -> Optional[int]:
        """
        矩形に最も近いライフラインを見つける

        Args:
            x: 矩形のX座標
            width: 矩形の幅
            lifelines: ライフラインX座標リスト

        Returns:
            Optional[int]: 最も近いライフラインのX座標、または None（ライフラインが空の場合）
        """
        if not lifelines:
            return None

        # 矩形の中心X座標
        rect_center_x = x + width // 2

        # 最も近いライフラインを見つける
        nearest_lifeline = min(lifelines, key=lambda ll: abs(ll - rect_center_x))

        # 許容範囲内かチェック
        distance = abs(nearest_lifeline - rect_center_x)
        if distance <= self.LIFELINE_MATCH_TOLERANCE:
            return nearest_lifeline

        # 許容範囲外の場合はNoneを返す
        logger.debug(
            f"Rectangle at x={x} is {distance}px from nearest lifeline, "
            f"beyond tolerance ({self.LIFELINE_MATCH_TOLERANCE}px)"
        )
        return None
