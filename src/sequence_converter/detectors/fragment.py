"""Fragment detector component."""

import logging
import re
from typing import Optional

import cv2
import numpy as np

from sequence_converter.models import Fragment, FragmentType, PreprocessedImage
from sequence_converter.ocr import OCREngine

logger = logging.getLogger(__name__)


class FragmentDetector:
    """フラグメント検出コンポーネント"""

    MIN_FRAGMENT_AREA: int = 10000  # 最小矩形面積（ピクセル²）
    MIN_FRAGMENT_WIDTH: int = 80  # 最小幅（ピクセル）
    MIN_FRAGMENT_HEIGHT: int = 60  # 最小高さ（ピクセル）
    LABEL_REGION_HEIGHT: int = 40  # 左上ラベル領域の高さ（ピクセル）
    LABEL_REGION_WIDTH: int = 150  # 左上ラベル領域の幅（ピクセル）

    def __init__(self, ocr_engine: OCREngine):
        """
        Args:
            ocr_engine: OCRエンジンインスタンス（依存性注入）
        """
        self.ocr_engine = ocr_engine

    def detect(self, preprocessed: PreprocessedImage) -> list[Fragment]:
        """
        大きな矩形フレームを検出し、左上ラベルからタイプを識別

        Args:
            preprocessed: 前処理済み画像

        Returns:
            list[Fragment]: 検出されたフラグメント（Y座標ソート済み）
        """
        # findContoursで大きな矩形フレームを検出
        large_rects = self._detect_large_rectangles(preprocessed.binary)

        if not large_rects:
            logger.info("No large rectangles detected for fragments")
            return []

        # 検出された矩形フレームをFragmentオブジェクトに変換
        fragments: list[Fragment] = []

        for rect in large_rects:
            x, y, width, height = rect

            # 左上隅の領域からラベルを抽出
            label_text = self._extract_label_from_top_left(
                preprocessed.grayscale, x, y, width, height
            )

            # ラベルテキストからフラグメントタイプを識別
            fragment_type, title = self._identify_fragment_type(label_text)

            # Fragmentオブジェクトを作成
            fragment = Fragment(
                type=fragment_type,
                title=title,
                bounding_box=(x, y, width, height),
                confidence=80.0,  # 矩形検出の信頼度（固定値）
            )

            fragments.append(fragment)

        # Y座標順にソート（上から下へ）
        fragments.sort(key=lambda f: f.bounding_box[1])

        logger.info(f"Detected {len(fragments)} fragment(s)")
        return fragments

    def _detect_large_rectangles(self, binary_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        findContoursで大きな矩形フレームを検出

        Args:
            binary_image: 二値化画像

        Returns:
            list[tuple[int, int, int, int]]: 検出された大きな矩形のリスト [(x, y, width, height), ...]
        """
        # 二値画像を反転（黒い図形を検出するため）
        inverted = cv2.bitwise_not(binary_image)

        # 輪郭検出
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_rects = []

        for contour in contours:
            # バウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # 面積フィルタリング
            if area < self.MIN_FRAGMENT_AREA:
                continue

            # 幅と高さのフィルタリング
            if w < self.MIN_FRAGMENT_WIDTH or h < self.MIN_FRAGMENT_HEIGHT:
                continue

            large_rects.append((x, y, w, h))

        logger.debug(f"Detected {len(large_rects)} potential fragment rectangles")
        return large_rects

    def _extract_label_from_top_left(
        self, grayscale: np.ndarray, x: int, y: int, width: int, height: int
    ) -> str:
        """
        左上隅の領域からラベルテキストを抽出

        Args:
            grayscale: グレースケール画像
            x: 矩形のX座標
            y: 矩形のY座標
            width: 矩形の幅
            height: 矩形の高さ

        Returns:
            str: 抽出されたラベルテキスト
        """
        # 左上隅の領域を定義
        label_width = min(self.LABEL_REGION_WIDTH, width)
        label_height = min(self.LABEL_REGION_HEIGHT, height)

        # OCRでテキストを抽出
        label_text = self.ocr_engine.extract_from_region(
            grayscale, (x, y, label_width, label_height)
        )

        logger.debug(f"Extracted label text: '{label_text}' from fragment at ({x}, {y})")
        return label_text.strip() if label_text else ""

    def _identify_fragment_type(self, label_text: str) -> tuple[FragmentType, Optional[str]]:
        """
        ラベルテキストからフラグメントタイプとタイトルを識別

        Args:
            label_text: ラベルテキスト

        Returns:
            tuple[FragmentType, Optional[str]]: (フラグメントタイプ, タイトルテキスト)
        """
        if not label_text:
            logger.debug("No label text found, treating as GROUP")
            return FragmentType.GROUP, None

        # 小文字に変換して比較
        label_lower = label_text.lower()

        # フラグメントタイプキーワードを検索
        if "alt" in label_lower:
            # "alt"以降のテキストをタイトルとして抽出
            title = self._extract_title_after_keyword(label_text, "alt")
            return FragmentType.ALT, title

        if "loop" in label_lower:
            title = self._extract_title_after_keyword(label_text, "loop")
            return FragmentType.LOOP, title

        if "opt" in label_lower:
            title = self._extract_title_after_keyword(label_text, "opt")
            return FragmentType.OPT, title

        if "par" in label_lower:
            title = self._extract_title_after_keyword(label_text, "par")
            return FragmentType.PAR, title

        # キーワードが見つからない場合はGROUPとして扱う
        logger.debug(f"No fragment type keyword found in '{label_text}', treating as GROUP")
        return FragmentType.GROUP, label_text

    def _extract_title_after_keyword(self, text: str, keyword: str) -> Optional[str]:
        """
        キーワード以降のテキストをタイトルとして抽出

        Args:
            text: 元のテキスト
            keyword: フラグメントタイプキーワード（alt, loop等）

        Returns:
            Optional[str]: タイトルテキスト、またはNone（タイトルがない場合）
        """
        # 大文字小文字を区別せずにキーワードを検索
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(text)

        if match:
            # キーワード以降のテキストを取得
            title = text[match.end() :].strip()
            return title if title else None

        return None
