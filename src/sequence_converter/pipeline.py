"""Pipeline orchestrator for sequence diagram conversion."""

import logging
from pathlib import Path

from sequence_converter.detectors.activation_bar import ActivationBarDetector
from sequence_converter.detectors.fragment import FragmentDetector
from sequence_converter.detectors.label_mapper import MessageLabelMapper
from sequence_converter.detectors.message_arrow import MessageArrowDetector
from sequence_converter.detectors.note_annotation import NoteAnnotationDetector
from sequence_converter.detectors.object_header import ObjectHeaderDetector
from sequence_converter.detectors.self_call import SelfCallDetector
from sequence_converter.generator import PlantUMLGenerator
from sequence_converter.logger import setup_logger
from sequence_converter.models import ConversionConfig, PreprocessedImage, SequenceDiagramElements
from sequence_converter.ocr import OCREngine
from sequence_converter.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """パイプライン全体を制御し、エラーハンドリングを含む変換処理を実行するコンポーネント"""

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        object_detector: ObjectHeaderDetector,
        arrow_detector: MessageArrowDetector,
        self_call_detector: SelfCallDetector,
        note_detector: NoteAnnotationDetector,
        activation_detector: ActivationBarDetector,
        fragment_detector: FragmentDetector,
        ocr_engine: OCREngine,
        label_mapper: MessageLabelMapper,
        plantuml_generator: PlantUMLGenerator,
    ):
        """
        全コンポーネントを依存性注入で受け取る

        Args:
            preprocessor: 画像前処理コンポーネント
            object_detector: オブジェクトヘッダー検出コンポーネント
            arrow_detector: メッセージ矢印検出コンポーネント
            self_call_detector: 自己呼び出し検出コンポーネント
            note_detector: ノート注釈検出コンポーネント
            activation_detector: アクティベーションバー検出コンポーネント
            fragment_detector: フラグメント検出コンポーネント
            ocr_engine: OCRエンジンコンポーネント
            label_mapper: メッセージラベルマッパーコンポーネント
            plantuml_generator: PlantUML生成コンポーネント
        """
        self.preprocessor = preprocessor
        self.object_detector = object_detector
        self.arrow_detector = arrow_detector
        self.self_call_detector = self_call_detector
        self.note_detector = note_detector
        self.activation_detector = activation_detector
        self.fragment_detector = fragment_detector
        self.ocr_engine = ocr_engine
        self.label_mapper = label_mapper
        self.plantuml_generator = plantuml_generator

    def convert(self, config: ConversionConfig) -> str:
        """
        画像からPlantUMLへの変換パイプラインを実行

        Args:
            config: 変換設定

        Returns:
            str: 生成されたPlantUMLコード

        Raises:
            FileNotFoundError: 入力ファイルが存在しない
            ValueError: サポートされていない形式

        処理フロー:
            1. 前処理
            2. オブジェクトヘッダー検出
            3. OCRテキスト抽出
            4. メッセージ矢印検出
            5. ラベルマッピング
            6. その他の検出（自己呼び出し、ノート、アクティベーション、フラグメント）
            7. PlantUML生成
        """
        # ファイル存在チェック
        if not config.input_path.exists():
            logger.error(f"Input file not found: {config.input_path}")
            raise FileNotFoundError(f"Input file not found: {config.input_path}")

        logger.info(f"Starting conversion pipeline for {config.input_path}")

        # ステップ1: 画像前処理
        logger.info("Step 1/7: Preprocessing image")
        preprocessed = self.preprocessor.preprocess(str(config.input_path))
        logger.info("Preprocessing completed")

        # ステップ2: オブジェクトヘッダー検出
        logger.info("Step 2/7: Detecting object headers")
        objects = self.object_detector.detect(preprocessed)
        lifelines = [obj.lifeline_x for obj in objects]
        logger.info(f"Detected {len(objects)} objects with lifelines: {lifelines}")

        # ステップ3: OCRテキスト領域抽出
        logger.info("Step 3/7: Extracting text regions with OCR")
        # グレースケール画像を使用してOCRを実行
        ocr_results = self.ocr_engine.extract_text_regions(preprocessed.grayscale)
        logger.info(f"Extracted {len(ocr_results)} text regions")

        # ステップ4: メッセージ矢印検出
        logger.info("Step 4/7: Detecting message arrows")
        messages = self.arrow_detector.detect(preprocessed, lifelines)
        logger.info(f"Detected {len(messages)} message arrows")

        # ステップ5: メッセージラベルマッピング
        logger.info("Step 5/7: Mapping labels to messages")
        message_labels = self.label_mapper.map_labels_to_messages(messages, ocr_results)
        logger.info(f"Mapped {len(message_labels)} message labels")

        # ステップ6: その他の要素検出
        logger.info("Step 6/7: Detecting other elements")

        # 自己呼び出し検出
        self_calls = self.self_call_detector.detect(preprocessed, lifelines)
        logger.info(f"Detected {len(self_calls)} self calls")

        # ノート注釈検出
        notes = self.note_detector.detect(preprocessed, lifelines)
        logger.info(f"Detected {len(notes)} note annotations")

        # アクティベーションバー検出
        activations = self.activation_detector.detect(preprocessed, lifelines)
        logger.info(f"Detected {len(activations)} activation bars")

        # フラグメント検出
        fragments = self.fragment_detector.detect(preprocessed)
        logger.info(f"Detected {len(fragments)} fragments")

        # ステップ7: PlantUML生成
        logger.info("Step 7/7: Generating PlantUML code")
        elements = SequenceDiagramElements(
            objects=objects,
            messages=messages,
            self_calls=self_calls,
            notes=notes,
            activations=activations,
            fragments=fragments,
            message_labels=message_labels,
        )
        plantuml_code = self.plantuml_generator.generate(elements)
        logger.info("PlantUML code generated successfully")

        # オプション: ファイルに保存
        if config.output_path:
            logger.info(f"Saving PlantUML to {config.output_path}")
            self.plantuml_generator.save_to_file(plantuml_code, config.output_path)

        logger.info("Conversion pipeline completed successfully")
        return plantuml_code
