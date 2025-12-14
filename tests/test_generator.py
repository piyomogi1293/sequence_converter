"""Tests for PlantUMLGenerator component."""

import sys
from pathlib import Path

# Force src to be at position 0 to avoid test directory shadowing
src_path = Path(__file__).parent.parent / "src"
src_str = str(src_path)
# Remove if already present
if src_str in sys.path:
    sys.path.remove(src_str)
# Insert at position 0
sys.path.insert(0, src_str)

import pytest

from sequence_converter.generator import PlantUMLGenerator
from sequence_converter.models import (
    ActivationBar,
    ArrowDirection,
    Fragment,
    FragmentType,
    MessageArrow,
    NoteAnnotation,
    ObjectHeader,
    OCRResult,
    SelfCall,
    SequenceDiagramElements,
)


class TestPlantUMLGenerator:
    """PlantUMLGeneratorコンポーネントのテストクラス"""

    @pytest.fixture
    def generator(self):
        """PlantUMLGeneratorインスタンスを生成"""
        return PlantUMLGenerator()

    @pytest.fixture
    def sample_objects(self):
        """サンプルオブジェクトヘッダーを生成"""
        return [
            ObjectHeader(name="Client", lifeline_x=100),
            ObjectHeader(name="Server", lifeline_x=300),
        ]

    def test_generate_basic_structure(self, generator):
        """基本的なPlantUML構造(@startuml/@enduml)が生成されることをテスト"""
        # Arrange
        elements = SequenceDiagramElements(
            objects=[],
            messages=[],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert result.startswith("@startuml"), "@startumlで始まること"
        assert result.endswith("@enduml"), "@endumlで終わること"

    def test_generate_object_declarations(self, generator, sample_objects):
        """オブジェクト宣言が生成されることをテスト"""
        # Arrange
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert 'participant "Client"' in result, "Clientのparticipant宣言が含まれること"
        assert 'participant "Server"' in result, "Serverのparticipant宣言が含まれること"

    def test_generate_message_left_to_right(self, generator, sample_objects):
        """左から右へのメッセージが生成されることをテスト"""
        # Arrange
        message = MessageArrow(
            source_lifeline=100,
            dest_lifeline=300,
            y=150,
            direction=ArrowDirection.LEFT_TO_RIGHT,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert '"Client" -> "Server"' in result, "左から右への矢印が含まれること"

    def test_generate_message_right_to_left(self, generator, sample_objects):
        """右から左へのメッセージが生成されることをテスト"""
        # Arrange
        message = MessageArrow(
            source_lifeline=300,
            dest_lifeline=100,
            y=150,
            direction=ArrowDirection.RIGHT_TO_LEFT,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert '"Server" <- "Client"' in result, "右から左への矢印が含まれること"

    def test_generate_message_with_label(self, generator, sample_objects):
        """ラベル付きメッセージが生成されることをテスト"""
        # Arrange
        message = MessageArrow(
            source_lifeline=100,
            dest_lifeline=300,
            y=150,
            direction=ArrowDirection.LEFT_TO_RIGHT,
        )
        message_labels = {150: OCRResult(text="Request", color=None)}
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels=message_labels,
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert ": Request" in result, "メッセージラベルが含まれること"

    def test_generate_message_with_color(self, generator, sample_objects):
        """色情報付きメッセージが生成されることをテスト"""
        # Arrange
        message = MessageArrow(
            source_lifeline=100,
            dest_lifeline=300,
            y=150,
            direction=ArrowDirection.LEFT_TO_RIGHT,
        )
        message_labels = {150: OCRResult(text="Request", color="red")}
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels=message_labels,
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert "[#red]Request" in result, "色情報が含まれること"

    def test_generate_self_call(self, generator, sample_objects):
        """自己呼び出しが生成されることをテスト"""
        # Arrange
        self_call = SelfCall(lifeline_x=100, y=150, label="Process")
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[self_call],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert '"Client" -> "Client" : Process' in result, "自己呼び出しが含まれること"

    def test_generate_note_with_related_lifeline(self, generator, sample_objects):
        """関連ライフライン付きノートが生成されることをテスト"""
        # Arrange
        note = NoteAnnotation(
            bounding_box=(90, 200, 80, 40),
            text="Important note",
            related_lifeline=100,
            confidence=85.0,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[],
            notes=[note],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert 'note over "Client" : Important note' in result, "ノートが含まれること"

    def test_generate_note_without_related_lifeline(self, generator, sample_objects):
        """関連ライフラインなしのノートが生成されることをテスト"""
        # Arrange
        note = NoteAnnotation(
            bounding_box=(50, 200, 80, 40),
            text="General note",
            related_lifeline=None,
            confidence=85.0,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[],
            notes=[note],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert "note left : General note" in result, "一般的なノートが含まれること"

    def test_generate_fragment_alt(self, generator, sample_objects):
        """altフラグメントが生成されることをテスト"""
        # Arrange
        fragment = Fragment(
            bounding_box=(50, 100, 300, 200),
            type=FragmentType.ALT,
            title="condition",
            confidence=90.0,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[],
            notes=[],
            fragments=[fragment],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert "alt condition" in result, "altフラグメント開始が含まれること"
        assert result.count("end") >= 1, "endが含まれること"

    def test_generate_fragment_loop(self, generator, sample_objects):
        """loopフラグメントが生成されることをテスト"""
        # Arrange
        fragment = Fragment(
            bounding_box=(50, 100, 300, 200),
            type=FragmentType.LOOP,
            title="until done",
            confidence=90.0,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[],
            notes=[],
            fragments=[fragment],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert "loop until done" in result, "loopフラグメントが含まれること"

    def test_generate_fragment_without_title(self, generator, sample_objects):
        """タイトルなしフラグメントが生成されることをテスト"""
        # Arrange
        fragment = Fragment(
            bounding_box=(50, 100, 300, 200),
            type=FragmentType.OPT,
            title=None,
            confidence=90.0,
        )
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[],
            self_calls=[],
            notes=[],
            fragments=[fragment],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert "opt\n" in result or result.endswith("opt"), "タイトルなしoptが含まれること"

    def test_events_sorted_by_y_coordinate(self, generator, sample_objects):
        """イベントがY座標順にソートされることをテスト"""
        # Arrange
        # Y座標が逆順になるように配置
        message1 = MessageArrow(
            source_lifeline=100, dest_lifeline=300, y=300, direction=ArrowDirection.LEFT_TO_RIGHT
        )
        message2 = MessageArrow(
            source_lifeline=100, dest_lifeline=300, y=100, direction=ArrowDirection.LEFT_TO_RIGHT
        )
        message_labels = {
            300: OCRResult(text="Second", color=None),
            100: OCRResult(text="First", color=None),
        }
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message1, message2],  # 逆順で渡す
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels=message_labels,
        )

        # Act
        result = generator.generate(elements)

        # Assert
        lines = result.split("\n")
        first_index = next(i for i, line in enumerate(lines) if "First" in line)
        second_index = next(i for i, line in enumerate(lines) if "Second" in line)
        assert first_index < second_index, "Y座標が小さいイベントが先に出力されること"

    def test_activation_suffix_start(self, generator, sample_objects):
        """アクティベーション開始構文(++)が付与されることをテスト"""
        # Arrange
        message = MessageArrow(
            source_lifeline=100,
            dest_lifeline=300,
            y=150,
            direction=ArrowDirection.LEFT_TO_RIGHT,
        )
        activation = ActivationBar(lifeline_x=300, y_start=150, y_end=250)
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[activation],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert " ++" in result, "アクティベーション開始構文が含まれること"

    def test_activation_suffix_end(self, generator, sample_objects):
        """アクティベーション終了構文(--)が付与されることをテスト"""
        # Arrange
        message = MessageArrow(
            source_lifeline=300,
            dest_lifeline=100,
            y=250,
            direction=ArrowDirection.RIGHT_TO_LEFT,
        )
        activation = ActivationBar(lifeline_x=300, y_start=150, y_end=250)
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[activation],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert " --" in result, "アクティベーション終了構文が含まれること"

    def test_save_to_file(self, generator, tmp_path):
        """PlantUMLコードがファイルに保存されることをテスト"""
        # Arrange
        plantuml_code = '@startuml\nparticipant "Test"\n@enduml'
        output_path = tmp_path / "output.puml"

        # Act
        generator.save_to_file(plantuml_code, output_path)

        # Assert
        assert output_path.exists(), "ファイルが作成されること"
        content = output_path.read_text(encoding="utf-8")
        assert content == plantuml_code, "内容が正しく保存されること"

    def test_save_to_file_creates_parent_directory(self, generator, tmp_path):
        """親ディレクトリが存在しない場合に作成されることをテスト"""
        # Arrange
        plantuml_code = '@startuml\nparticipant "Test"\n@enduml'
        output_path = tmp_path / "subdir" / "output.puml"

        # Act
        generator.save_to_file(plantuml_code, output_path)

        # Assert
        assert output_path.parent.exists(), "親ディレクトリが作成されること"
        assert output_path.exists(), "ファイルが作成されること"

    def test_complex_scenario(self, generator, sample_objects):
        """複雑なシナリオ（メッセージ、ノート、フラグメント混在）が正しく生成されることをテスト"""
        # Arrange
        message1 = MessageArrow(
            source_lifeline=100, dest_lifeline=300, y=150, direction=ArrowDirection.LEFT_TO_RIGHT
        )
        note = NoteAnnotation(
            bounding_box=(90, 200, 80, 40),
            text="Note",
            related_lifeline=100,
            confidence=85.0,
        )
        fragment = Fragment(
            bounding_box=(50, 100, 300, 250),
            type=FragmentType.ALT,
            title="condition",
            confidence=90.0,
        )
        message_labels = {150: OCRResult(text="Request", color="red")}
        elements = SequenceDiagramElements(
            objects=sample_objects,
            messages=[message1],
            self_calls=[],
            notes=[note],
            fragments=[fragment],
            activations=[],
            message_labels=message_labels,
        )

        # Act
        result = generator.generate(elements)

        # Assert
        assert 'participant "Client"' in result
        assert 'participant "Server"' in result
        assert "alt condition" in result
        assert "[#red]Request" in result
        assert 'note over "Client" : Note' in result
        assert "end" in result
        assert result.startswith("@startuml")
        assert result.endswith("@enduml")

    def test_unmatched_lifeline_message_returns_none(self, generator):
        """ライフラインにマッチしないメッセージがスキップされることをテスト"""
        # Arrange
        # オブジェクトはlifeline_x=100のみ
        objects = [ObjectHeader(name="Client", lifeline_x=100)]
        # メッセージのdest_lifelineが存在しない500
        message = MessageArrow(
            source_lifeline=100,
            dest_lifeline=500,
            y=150,
            direction=ArrowDirection.LEFT_TO_RIGHT,
        )
        elements = SequenceDiagramElements(
            objects=objects,
            messages=[message],
            self_calls=[],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        # メッセージ行が含まれないことを確認（@startumlとparticipantとendのみ）
        lines = [line for line in result.split("\n") if line.strip() and not line.startswith("@")]
        message_lines = [line for line in lines if "->" in line or "<-" in line]
        assert len(message_lines) == 0, "マッチしないメッセージはスキップされること"

    def test_unmatched_lifeline_self_call_returns_none(self, generator):
        """ライフラインにマッチしない自己呼び出しがスキップされることをテスト"""
        # Arrange
        objects = [ObjectHeader(name="Client", lifeline_x=100)]
        self_call = SelfCall(lifeline_x=500, y=150, label="Process")
        elements = SequenceDiagramElements(
            objects=objects,
            messages=[],
            self_calls=[self_call],
            notes=[],
            fragments=[],
            activations=[],
            message_labels={},
        )

        # Act
        result = generator.generate(elements)

        # Assert
        lines = [line for line in result.split("\n") if line.strip() and not line.startswith("@")]
        self_call_lines = [line for line in lines if "Process" in line]
        assert len(self_call_lines) == 0, "マッチしない自己呼び出しはスキップされること"
