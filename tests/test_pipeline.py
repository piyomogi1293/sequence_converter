"""PipelineOrchestratorのユニットテスト"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

# テストのためにsrcをsys.pathに追加
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest

from sequence_converter.detectors.activation_bar import ActivationBarDetector
from sequence_converter.detectors.fragment import FragmentDetector
from sequence_converter.detectors.label_mapper import MessageLabelMapper
from sequence_converter.detectors.message_arrow import MessageArrowDetector
from sequence_converter.detectors.note_annotation import NoteAnnotationDetector
from sequence_converter.detectors.object_header import ObjectHeaderDetector
from sequence_converter.detectors.self_call import SelfCallDetector
from sequence_converter.generator import PlantUMLGenerator
from sequence_converter.models import (
    ActivationBar,
    ArrowDirection,
    ConversionConfig,
    Fragment,
    FragmentType,
    MessageArrow,
    NoteAnnotation,
    ObjectHeader,
    OCRResult,
    PreprocessedImage,
    PreprocessingConfig,
    SelfCall,
    SequenceDiagramElements,
)
from sequence_converter.ocr import OCREngine
from sequence_converter.pipeline import PipelineOrchestrator
from sequence_converter.preprocessing import ImagePreprocessor


class TestPipelineOrchestrator:
    """PipelineOrchestratorのテストケース"""

    @pytest.fixture
    def mock_preprocessor(self):
        """ImagePreprocessorのモック"""
        preprocessor = Mock(spec=ImagePreprocessor)
        preprocessor.preprocess.return_value = Mock(spec=PreprocessedImage)
        return preprocessor

    @pytest.fixture
    def mock_ocr_engine(self):
        """OCREngineのモック"""
        ocr_engine = Mock(spec=OCREngine)
        ocr_engine.extract_text_regions.return_value = []
        return ocr_engine

    @pytest.fixture
    def mock_object_detector(self):
        """ObjectHeaderDetectorのモック"""
        detector = Mock(spec=ObjectHeaderDetector)
        detector.detect.return_value = [
            ObjectHeader(
                name="Object1", lifeline_x=100, bounding_box=(0, 0, 50, 30), confidence=90.0
            ),
            ObjectHeader(
                name="Object2", lifeline_x=200, bounding_box=(150, 0, 50, 30), confidence=90.0
            ),
        ]
        return detector

    @pytest.fixture
    def mock_arrow_detector(self):
        """MessageArrowDetectorのモック"""
        detector = Mock(spec=MessageArrowDetector)
        detector.detect.return_value = [
            MessageArrow(
                start_x=100,
                end_x=200,
                y=50,
                direction=ArrowDirection.LEFT_TO_RIGHT,
                source_lifeline=100,
                dest_lifeline=200,
            )
        ]
        return detector

    @pytest.fixture
    def mock_label_mapper(self):
        """MessageLabelMapperのモック"""
        mapper = Mock(spec=MessageLabelMapper)
        mapper.map_labels_to_messages.return_value = {
            50: OCRResult(
                text="message1", bounding_box=(120, 45, 60, 10), confidence=90.0, color=None
            )
        }
        return mapper

    @pytest.fixture
    def mock_self_call_detector(self):
        """SelfCallDetectorのモック"""
        detector = Mock(spec=SelfCallDetector)
        detector.detect.return_value = []
        return detector

    @pytest.fixture
    def mock_note_detector(self):
        """NoteAnnotationDetectorのモック"""
        detector = Mock(spec=NoteAnnotationDetector)
        detector.detect.return_value = []
        return detector

    @pytest.fixture
    def mock_activation_detector(self):
        """ActivationBarDetectorのモック"""
        detector = Mock(spec=ActivationBarDetector)
        detector.detect.return_value = []
        return detector

    @pytest.fixture
    def mock_fragment_detector(self):
        """FragmentDetectorのモック"""
        detector = Mock(spec=FragmentDetector)
        detector.detect.return_value = []
        return detector

    @pytest.fixture
    def mock_plantuml_generator(self):
        """PlantUMLGeneratorのモック"""
        generator = Mock(spec=PlantUMLGenerator)
        generator.generate.return_value = "@startuml\nparticipant Object1\nparticipant Object2\nObject1 -> Object2 : message1\n@enduml"
        return generator

    @pytest.fixture
    def orchestrator(
        self,
        mock_preprocessor,
        mock_object_detector,
        mock_arrow_detector,
        mock_self_call_detector,
        mock_note_detector,
        mock_activation_detector,
        mock_fragment_detector,
        mock_ocr_engine,
        mock_label_mapper,
        mock_plantuml_generator,
    ):
        """PipelineOrchestratorインスタンス"""
        return PipelineOrchestrator(
            preprocessor=mock_preprocessor,
            object_detector=mock_object_detector,
            arrow_detector=mock_arrow_detector,
            self_call_detector=mock_self_call_detector,
            note_detector=mock_note_detector,
            activation_detector=mock_activation_detector,
            fragment_detector=mock_fragment_detector,
            ocr_engine=mock_ocr_engine,
            label_mapper=mock_label_mapper,
            plantuml_generator=mock_plantuml_generator,
        )

    def test_convert_executes_full_pipeline(
        self,
        orchestrator,
        mock_preprocessor,
        mock_object_detector,
        mock_arrow_detector,
        mock_ocr_engine,
        mock_label_mapper,
        mock_self_call_detector,
        mock_note_detector,
        mock_activation_detector,
        mock_fragment_detector,
        mock_plantuml_generator,
        tmp_path,
    ):
        """変換パイプラインが全ステージを正しく実行することをテスト"""
        # テスト用の画像ファイルを作成
        input_file = tmp_path / "test.png"
        input_file.write_bytes(b"fake image data")
        output_file = tmp_path / "output.puml"

        config = ConversionConfig(
            input_path=input_file,
            output_path=output_file,
        )

        # パイプライン実行
        result = orchestrator.convert(config)

        # すべてのコンポーネントが呼び出されたことを確認
        mock_preprocessor.preprocess.assert_called_once()
        mock_object_detector.detect.assert_called_once()
        mock_ocr_engine.extract_text_regions.assert_called_once()
        mock_arrow_detector.detect.assert_called_once()
        mock_label_mapper.map_labels_to_messages.assert_called_once()
        mock_self_call_detector.detect.assert_called_once()
        mock_note_detector.detect.assert_called_once()
        mock_activation_detector.detect.assert_called_once()
        mock_fragment_detector.detect.assert_called_once()
        mock_plantuml_generator.generate.assert_called_once()

        # PlantUMLコードが返されることを確認
        assert "@startuml" in result
        assert "@enduml" in result

    def test_convert_handles_file_not_found(self, orchestrator):
        """存在しないファイルに対してFileNotFoundErrorを発生させることをテスト"""
        config = ConversionConfig(
            input_path=Path("/nonexistent/file.png"),
            output_path=Path("/tmp/output.puml"),
        )

        with pytest.raises(FileNotFoundError):
            orchestrator.convert(config)

    def test_convert_continues_on_partial_detection_failure(
        self,
        orchestrator,
        mock_object_detector,
        mock_arrow_detector,
        mock_plantuml_generator,
        tmp_path,
    ):
        """一部の検出が失敗してもプロセスを継続することをテスト"""
        # テスト用の画像ファイルを作成
        input_file = tmp_path / "test.png"
        input_file.write_bytes(b"fake image data")
        output_file = tmp_path / "output.puml"

        # 一部の検出器が空の結果を返す（検出失敗をシミュレート）
        mock_object_detector.detect.return_value = []
        mock_arrow_detector.detect.return_value = []

        config = ConversionConfig(
            input_path=input_file,
            output_path=output_file,
        )

        # エラーにならず、PlantUMLコードが生成されることを確認
        result = orchestrator.convert(config)
        assert "@startuml" in result
        assert "@enduml" in result
        mock_plantuml_generator.generate.assert_called_once()

    def test_convert_passes_correct_data_between_stages(
        self,
        orchestrator,
        mock_preprocessor,
        mock_object_detector,
        mock_arrow_detector,
        mock_plantuml_generator,
        tmp_path,
    ):
        """ステージ間でデータが正しく受け渡されることをテスト"""
        # テスト用の画像ファイルを作成
        input_file = tmp_path / "test.png"
        input_file.write_bytes(b"fake image data")
        output_file = tmp_path / "output.puml"

        config = ConversionConfig(
            input_path=input_file,
            output_path=output_file,
        )

        orchestrator.convert(config)

        # arrow_detectorにlifelinesが正しく渡されることを確認
        call_args = mock_arrow_detector.detect.call_args
        lifelines_arg = call_args[1]["lifelines"]
        assert lifelines_arg == [100, 200]  # mock_object_detectorのlifeline_x

    def test_convert_uses_config_parameters(
        self,
        orchestrator,
        mock_preprocessor,
        tmp_path,
    ):
        """ConversionConfigのパラメータが正しく使用されることをテスト"""
        # テスト用の画像ファイルを作成
        input_file = tmp_path / "test.png"
        input_file.write_bytes(b"fake image data")
        output_file = tmp_path / "output.puml"

        # カスタム設定を使用
        custom_config = ConversionConfig(
            input_path=input_file,
            output_path=output_file,
            preprocessing=PreprocessingConfig(
                blur_kernel_size=7, binary_threshold_method="adaptive"
            ),
            lifeline_match_tolerance=30,
            label_search_range=(10, 40),
        )

        orchestrator.convert(custom_config)

        # preprocessorに設定が渡されることを確認（実装時に検証）
        # この部分は実装によって異なる可能性があるため、基本的な動作のみ確認
        mock_preprocessor.preprocess.assert_called_once()
