"""MessageLabelMapper component for mapping OCR labels to message arrows."""

import logging
from typing import Optional

from sequence_converter.models import MessageArrow, OCRResult

logger = logging.getLogger(__name__)


class MessageLabelMapper:
    """メッセージラベルマッピングコンポーネント"""

    LABEL_SEARCH_RANGE: tuple[int, int] = (5, 30)  # 上下のピクセル範囲（制約条件より）

    def map_labels_to_messages(
        self, messages: list[MessageArrow], ocr_results: list[OCRResult]
    ) -> dict[int, OCRResult]:
        """
        矢印Y座標から上下5〜30ピクセル範囲のテキストを検索してマッピング

        Args:
            messages: 検出されたメッセージ矢印（Y座標ソート済み）
            ocr_results: OCRで抽出されたテキスト領域

        Returns:
            dict[int, OCRResult]: メッセージY座標 → OCRラベルのマッピング

        Notes:
            - 複数のテキストが範囲内に存在する場合、最も近いものを選択
            - マッチしなかった矢印は空のマッピングとして扱う（ログに記録）
        """
        if not messages or not ocr_results:
            return {}

        result: dict[int, OCRResult] = {}
        min_distance, max_distance = self.LABEL_SEARCH_RANGE

        for message in messages:
            best_match: Optional[OCRResult] = None
            best_score: float = float("inf")

            for ocr in ocr_results:
                # OCRバウンディングボックス: (left, top, width, height)
                ocr_left, ocr_top, ocr_width, ocr_height = ocr.bounding_box

                # Y座標の距離を計算（矢印のY座標とOCRテキストのtop座標の差）
                y_distance = abs(message.y - ocr_top)

                # 範囲チェック: min_distance <= y_distance <= max_distance
                if y_distance < min_distance or y_distance > max_distance:
                    continue

                # X座標の考慮: OCRテキストが矢印のX範囲内にあるかチェック
                # 矢印のX範囲: min(start_x, end_x) 〜 max(start_x, end_x)
                arrow_x_min = min(message.start_x, message.end_x)
                arrow_x_max = max(message.start_x, message.end_x)

                # OCRテキストの中心X座標
                ocr_center_x = ocr_left + ocr_width // 2

                # X座標の距離を計算（範囲内なら0、範囲外ならその距離）
                if arrow_x_min <= ocr_center_x <= arrow_x_max:
                    x_distance = 0
                else:
                    x_distance = min(
                        abs(ocr_center_x - arrow_x_min), abs(ocr_center_x - arrow_x_max)
                    )

                # スコア計算: Y距離を優先し、X距離も考慮
                # Y距離が同じ場合、X距離が小さい方を優先
                score = y_distance + x_distance * 0.5

                if score < best_score:
                    best_score = score
                    best_match = ocr

            if best_match:
                result[message.y] = best_match
            else:
                logger.warning(
                    f"No label found for message at y={message.y} "
                    f"(start_x={message.start_x}, end_x={message.end_x})"
                )

        return result
