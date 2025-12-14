"""CLI interface for sequence converter."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sequence_converter.detectors.activation_bar import ActivationBarDetector
from sequence_converter.detectors.fragment import FragmentDetector
from sequence_converter.detectors.label_mapper import MessageLabelMapper
from sequence_converter.detectors.message_arrow import MessageArrowDetector
from sequence_converter.detectors.note_annotation import NoteAnnotationDetector
from sequence_converter.detectors.object_header import ObjectHeaderDetector
from sequence_converter.detectors.self_call import SelfCallDetector
from sequence_converter.generator import PlantUMLGenerator
from sequence_converter.logger import setup_logger
from sequence_converter.models import ConversionConfig  # , PreprocessingConfig
from sequence_converter.ocr import OCREngine
from sequence_converter.pipeline import PipelineOrchestrator
from sequence_converter.preprocessing import ImagePreprocessor

# from typing import Optional


app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)


def create_pipeline_orchestrator() -> PipelineOrchestrator:
    """パイプラインオーケストレーターを作成する"""
    # OCRエンジンを作成
    ocr_engine = OCREngine()

    # 各種検出器を作成
    preprocessor = ImagePreprocessor()
    object_detector = ObjectHeaderDetector(ocr_engine)
    arrow_detector = MessageArrowDetector()
    label_mapper = MessageLabelMapper()
    self_call_detector = SelfCallDetector(ocr_engine)
    note_detector = NoteAnnotationDetector(ocr_engine)
    activation_detector = ActivationBarDetector()
    fragment_detector = FragmentDetector(ocr_engine)
    plantuml_generator = PlantUMLGenerator()

    # パイプラインオーケストレーターを作成
    return PipelineOrchestrator(
        preprocessor=preprocessor,
        object_detector=object_detector,
        arrow_detector=arrow_detector,
        self_call_detector=self_call_detector,
        note_detector=note_detector,
        activation_detector=activation_detector,
        fragment_detector=fragment_detector,
        ocr_engine=ocr_engine,
        label_mapper=label_mapper,
        plantuml_generator=plantuml_generator,
    )


@app.command()
def convert(
    input_dir: Path = typer.Option(
        Path("input"),
        "--input",
        "-i",
        help="入力ディレクトリ（PNG/JPG画像）",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output",
        "-o",
        help="出力ディレクトリ（.pumlファイル）",
    ),
) -> None:
    """
    シーケンス図画像をPlantUMLに変換

    Examples:
        $ sequence-converter convert
        $ sequence-converter convert --input ./images --output ./plantuml
    """
    # ロガーをセットアップ
    setup_logger()

    # 入力ディレクトリの存在チェック
    if not input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory {input_dir} does not exist")
        raise typer.Exit(code=1)

    # 出力ディレクトリを作成（存在しない場合）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 対応している画像形式
    supported_formats = {".png", ".jpg", ".jpeg"}

    # 入力ディレクトリ内の画像ファイルを取得
    image_files = [
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_formats
    ]

    if not image_files:
        console.print(f"[yellow]Warning:[/yellow] No image files found in {input_dir}")
        return

    console.print(f"[green]Found {len(image_files)} image file(s) to process[/green]")

    # パイプラインオーケストレーターを作成
    pipeline = create_pipeline_orchestrator()

    # 各画像ファイルを処理
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for image_file in image_files:
            task = progress.add_task(f"Processing {image_file.name}...", total=None)

            try:
                # 出力ファイルパスを生成
                output_file = output_dir / f"{image_file.stem}.puml"

                # 変換設定を作成
                config = ConversionConfig(
                    input_path=image_file,
                    output_path=output_file,
                )

                # 変換を実行
                plantuml_code = pipeline.convert(config)

                # 標準出力に表示
                console.print(f"\n[cyan]Generated PlantUML for {image_file.name}:[/cyan]")
                console.print(plantuml_code)
                console.print(f"\n[green]Saved to {output_file}[/green]\n")

                progress.update(task, completed=True)

            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"\n[red]Error processing {image_file.name}:[/red] {str(e)}\n")
                logger.exception(f"Error processing {image_file}")

    console.print("[green]Conversion completed![/green]")


if __name__ == "__main__":
    app()
