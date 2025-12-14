"""Tests for MessageLabelMapper component."""

import sys
from pathlib import Path

# テストのためにsrcをsys.pathに追加
test_dir = Path(__file__).parent.parent
project_root = test_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest

from sequence_converter.detectors.label_mapper import MessageLabelMapper
from sequence_converter.models import ArrowDirection, MessageArrow, OCRResult


class TestMessageLabelMapper:
    """MessageLabelMapperコンポーネントのテストクラス"""

    @pytest.fixture
    def mapper(self):
        """MessageLabelMapperインスタンスを生成"""
        return MessageLabelMapper()

    def test_map_labels_within_range(self, mapper):
        """矢印Y座標から上下5〜30px範囲のテキストが検索されることをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=150,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        ocr_results = [
            # Y座標が矢印から10px上にあるテキスト（範囲内）
            OCRResult(
                text="Message Label",
                bounding_box=(120, 140, 80, 10),  # (left, top, width, height)
                confidence=95.0,
                color=None,
            ),
            # Y座標が矢印から50px上にあるテキスト（範囲外）
            OCRResult(
                text="Out of Range",
                bounding_box=(120, 100, 80, 10),
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert 150 in result, "矢印Y座標150に対するマッピングが存在すること"
        assert result[150].text == "Message Label", "範囲内のテキストがマッピングされること"

    def test_select_nearest_text_when_multiple_in_range(self, mapper):
        """複数テキストが範囲内に存在する場合、最も近いものが選択されることをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=150,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        ocr_results = [
            # Y座標が矢印から5px上（範囲内、より近い）
            OCRResult(
                text="Nearest Label",
                bounding_box=(120, 145, 80, 10),
                confidence=95.0,
                color=None,
            ),
            # Y座標が矢印から20px上（範囲内、より遠い）
            OCRResult(
                text="Farther Label",
                bounding_box=(120, 130, 80, 10),
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert result[150].text == "Nearest Label", "最も近いテキストが選択されること"

    def test_no_match_when_no_text_in_range(self, mapper):
        """範囲内にテキストがない場合、マッチしないことをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=150,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        ocr_results = [
            # すべて範囲外
            OCRResult(
                text="Too Far Above",
                bounding_box=(120, 50, 80, 10),
                confidence=95.0,
                color=None,
            ),
            OCRResult(
                text="Too Far Below",
                bounding_box=(120, 250, 80, 10),
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert 150 not in result, "範囲外のテキストはマッピングされないこと"

    def test_consider_x_coordinate_for_better_matching(self, mapper):
        """X座標も考慮してマッチング精度が向上することをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=150,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        ocr_results = [
            # Y距離は同じだが、X座標が矢印の範囲内（100〜200）
            OCRResult(
                text="In X Range",
                bounding_box=(120, 145, 80, 10),  # left=120 は 100〜200の範囲内
                confidence=95.0,
                color=None,
            ),
            # Y距離は同じだが、X座標が矢印の範囲外
            OCRResult(
                text="Out of X Range",
                bounding_box=(300, 145, 80, 10),  # left=300 は範囲外
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert result[150].text == "In X Range", "X座標も考慮して最適なテキストが選択されること"

    def test_handle_multiple_messages(self, mapper):
        """複数のメッセージに対して正しくマッピングされることをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=100,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            ),
            MessageArrow(
                start_x=200,
                end_x=300,
                y=200,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=200,
                dest_lifeline=300,
            ),
        ]
        ocr_results = [
            OCRResult(
                text="First Message",
                bounding_box=(120, 95, 80, 10),
                confidence=95.0,
                color=None,
            ),
            OCRResult(
                text="Second Message",
                bounding_box=(220, 195, 80, 10),
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert len(result) == 2, "2つのメッセージがマッピングされること"
        assert result[100].text == "First Message"
        assert result[200].text == "Second Message"

    def test_empty_messages_list(self, mapper):
        """メッセージリストが空の場合、空の辞書を返すことをテスト"""
        # Arrange
        messages = []
        ocr_results = [
            OCRResult(
                text="Some Text",
                bounding_box=(120, 95, 80, 10),
                confidence=95.0,
                color=None,
            )
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert result == {}, "メッセージが空の場合は空の辞書を返すこと"

    def test_empty_ocr_results_list(self, mapper):
        """OCR結果リストが空の場合、空の辞書を返すことをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=150,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        ocr_results = []

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        assert result == {}, "OCR結果が空の場合は空の辞書を返すこと"

    def test_search_range_boundaries(self, mapper):
        """検索範囲の境界（5px〜30px）が正しく機能することをテスト"""
        # Arrange
        messages = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=150,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        ocr_results = [
            # ちょうど5px上（範囲内、下限）
            OCRResult(
                text="At Lower Bound",
                bounding_box=(120, 145, 80, 10),  # top=145, 矢印y=150, 差=5
                confidence=95.0,
                color=None,
            ),
            # 4px上（範囲外）
            OCRResult(
                text="Below Lower Bound",
                bounding_box=(120, 146, 80, 10),  # top=146, 差=4
                confidence=95.0,
                color=None,
            ),
            # ちょうど30px上（範囲内、上限）
            OCRResult(
                text="At Upper Bound",
                bounding_box=(120, 120, 80, 10),  # top=120, 差=30
                confidence=95.0,
                color=None,
            ),
            # 31px上（範囲外）
            OCRResult(
                text="Above Upper Bound",
                bounding_box=(120, 119, 80, 10),  # top=119, 差=31
                confidence=95.0,
                color=None,
            ),
        ]

        # Act
        result = mapper.map_labels_to_messages(messages, ocr_results)

        # Assert
        # 最も近いのは "At Lower Bound" (5px差)
        assert result[150].text == "At Lower Bound", "境界値のテキストが正しく選択されること"
