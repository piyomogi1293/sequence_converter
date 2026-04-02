# Design: sequence-converter-core

> **作成日時**: 2025-12-13T02:46:51Z
> **要件トレーサビリティ**: [requirements.md](./requirements.md)
> **調査ログ**: [research.md](./research.md)

---

## 概要

sequence-converter-coreは、シーケンス図画像（PNG/JPG）をPlantUMLテキスト形式に変換するPythonツールです。コンピュータビジョン（OpenCV）とOCR（Tesseract/pytesseract）を組み合わせ、パイプラインアーキテクチャによりモジュール化された検出・抽出・生成プロセスを実現します。各パイプラインステージ（前処理 → オブジェクト検出 → メッセージ検出 → OCR → PlantUML生成）は独立したコンポーネントとして設計され、拡張性・保守性・テスタビリティを確保します。

## アーキテクチャパターンと境界マップ

### 選定パターン

**パイプライン（Pipe-and-Filter）パターン**を採用します（[research.md](./research.md)のアーキテクチャパターン評価を参照）。

**選定理由**:

- 9つの異なる検出機能（オブジェクトヘッダー、メッセージ矢印、自己呼び出し、ノート、アクティベーション、フラグメント等）を独立したフィルターとして実装可能
- 各フィルターの責務が明確で、単体テストが容易
- 将来的な検出アルゴリズムの差し替えや新機能追加が容易（非機能要件：拡張性・保守性）
- 画像処理分野で実績あり（OpenCV + Pythonジェネレータ）

### コンポーネント境界

``` bash
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer (typer)                       │
│  - コマンドライン引数解析                                       │
│  - ファイル入力検証                                             │
│  - エラー表示とログ出力                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────────┐
│                   Pipeline Orchestrator                         │
│  - パイプライン全体の制御                                       │
│  - ステージ間のデータ受け渡し                                   │
│  - エラーハンドリングと継続処理                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        v                v                v
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Preprocessing │ │   Detection   │ │   Generation  │
│    Module     │ │    Module     │ │    Module     │
└───────────────┘ └───────────────┘ └───────────────┘
        │                │                │
        v                v                v
  - Grayscale      - Object Header   - PlantUML
  - Binarization   - Message Arrow      Formatter
  - Noise Removal  - Self Call       - File Writer
                   - Note Annotation
                   - Activation Bar
                   - Fragment
                   - OCR (pytesseract)
                   - Message Label Mapper (新規)
```

**境界の説明**:

- **CLI Layer**: 外部との境界。ユーザー入力のバリデーション（Pydantic使用）。
- **Pipeline Orchestrator**: 制御フロー境界。各モジュールを呼び出し、データを受け渡す。
- **Preprocessing Module**: 画像データ境界。生画像を処理済み画像に変換。
- **Detection Module**: 検出ロジック境界。画像から構造的データ（オブジェクト、メッセージ等）を抽出。
- **Generation Module**: 出力データ境界。構造的データをPlantUMLテキストに変換。

## 技術スタックとアライメント

### コア技術

| 技術 | 用途 | ステアリング整合性 |
|------|------|-------------------|
| **Python 3.x** | 実装言語 | ✅ tech.md で言及 |
| **OpenCV (opencv-python)** | 画像処理・輪郭検出 | ✅ tech.md で言及 |
| **Tesseract + pytesseract** | OCRエンジン | ✅ requirements.md 制約条件、tech.md で言及 |
| **Pydantic v2** | データバリデーション | ✅ tech.md で言及 |
| **typer** | CLI フレームワーク | ✅ tech.md で言及（typer or click） |
| **rich** | ログ・進捗表示 | ✅ tech.md で言及 |
| **numpy** | 数値計算・画像配列 | ✅ tech.md で言及 |
| **Pillow** | 画像読み込み | ✅ tech.md で言及 |

### 外部依存関係

- **Tesseract OCR Engine** (システムレベル):
  - バージョン: 3.05+ （`image_to_data`サポート）
  - インストール: システムパッケージマネージャ（apt, brew等）経由
  - リスク: システム依存性（[research.md](./research.md) リスク1参照）

- **PlantUML** (検証用、オプショナル):
  - バージョン: 1.2025.0互換
  - 用途: 統合テストでの生成コード検証

## コンポーネントとインターフェース契約

### コンポーネント1: ImagePreprocessor

**責務**: 入力画像を処理可能な形式に変換（グレースケール、二値化、ノイズ除去）

**要件トレース**: 要件1（画像入力と前処理）

**パブリックインターフェース**:

``` python
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
from numpy.typing import NDArray

@dataclass
class PreprocessedImage:
    """前処理済み画像データ（内部データ構造）"""
    grayscale: NDArray[np.uint8]
    binary: NDArray[np.uint8]
    original_shape: tuple[int, int]  # (height, width)

class PreprocessingConfig(BaseModel):
    """前処理設定（外部データ）"""
    blur_kernel_size: int = Field(5, ge=3, le=15, description="ブラーカーネルサイズ（奇数のみ）")
    binary_threshold_method: str = Field("otsu", pattern="^(otsu|adaptive)$", description="二値化手法")
    noise_removal_method: str = Field("median", pattern="^(median|gaussian)$", description="ノイズ除去手法")

    def validate_blur_kernel(self) -> "PreprocessingConfig":
        """ブラーカーネルサイズが奇数であることを検証"""
        if self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        return self

class ImagePreprocessor:
    """画像前処理コンポーネント"""

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        """
        Args:
            config: 前処理設定（デフォルト値で初期化）
        """
        self.config = config

    def preprocess(self, image_path: str) -> PreprocessedImage:
        """
        画像を読み込み、グレースケール・二値化・ノイズ除去を実行

        Args:
            image_path: 入力画像ファイルパス（PNG/JPG）

        Returns:
            PreprocessedImage: 前処理済み画像データ

        Raises:
            FileNotFoundError: ファイルが存在しない
            ValueError: サポートされていないフォーマット
        """
        ...
```

**依存関係**:

- OpenCV (`cv2`)
- Pillow (`PIL`)
- numpy
- Pydantic v2

**設計ノート**:

- [research.md](./research.md) トピック1: OpenCVベストプラクティスに基づき、`THRESH_BINARY + THRESH_OTSU` による適応的二値化を実装
- ノイズ除去には `cv2.medianBlur` または `cv2.GaussianBlur` を使用（設定により切り替え可能）
- デフォルトパラメータ: `blur_kernel_size=5`, `binary_threshold_method="otsu"`, `noise_removal_method="median"`
- CLIから前処理パラメータを調整可能（ConversionConfigに含める）

---

### コンポーネント2: ObjectHeaderDetector

**責務**: 画像上部からオブジェクトヘッダー（参加者）矩形を検出し、OCRでオブジェクト名を抽出

**要件トレース**: 要件2（オブジェクトヘッダー検出）

**パブリックインターフェース**:

``` python
from pydantic import BaseModel, Field

class ObjectHeader(BaseModel):
    """オブジェクトヘッダー検出結果（外部データ）"""
    name: str = Field(..., min_length=1, description="オブジェクト名")
    lifeline_x: int = Field(..., ge=0, description="ライフラインX座標")
    bounding_box: tuple[int, int, int, int] = Field(..., description="(x, y, width, height)")
    confidence: float = Field(..., ge=-1.0, le=100.0, description="OCR信頼度")

class ObjectHeaderDetector:
    """オブジェクトヘッダー検出コンポーネント"""

    def __init__(self, ocr_engine: 'OCREngine'):
        """
        Args:
            ocr_engine: OCRエンジンインスタンス（依存性注入）
        """
        self.ocr_engine = ocr_engine

    def detect(self, preprocessed: PreprocessedImage) -> list[ObjectHeader]:
        """
        画像上部から矩形を検出し、OCRでオブジェクト名を取得

        Args:
            preprocessed: 前処理済み画像

        Returns:
            list[ObjectHeader]: 検出されたオブジェクトヘッダー（左から右へソート済み）
        """
        ...
```

**依存関係**:

- OpenCV (`cv2.findContours`, `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`)
- OCREngine（後述）
- Pydantic v2

**設計ノート**:

- [research.md](./research.md) トピック1に基づき、`RETR_EXTERNAL`と`CHAIN_APPROX_SIMPLE`を使用してパフォーマンス最適化
- 検出失敗時はログに記録し、残りのオブジェクトの処理を継続（受け入れ基準2-5に準拠）

---

### コンポーネント3: MessageArrowDetector

**責務**: メッセージ矢印（水平線セグメント）を検出し、ライフラインにマッピング

**要件トレース**: 要件3（メッセージ矢印検出とライフラインマッピング）

**パブリックインターフェース**:

```python
from enum import Enum

class ArrowDirection(str, Enum):
    """矢印方向"""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"

class MessageArrow(BaseModel):
    """メッセージ矢印検出結果"""
    start_x: int = Field(..., ge=0)
    end_x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    direction: ArrowDirection
    source_lifeline: int | None = Field(None, description="送信元ライフラインX座標")
    dest_lifeline: int | None = Field(None, description="宛先ライフラインX座標")

class MessageArrowDetector:
    """メッセージ矢印検出コンポーネント"""

    LIFELINE_MATCH_TOLERANCE: int = 20  # ピクセル（制約条件より）

    def detect(
        self,
        preprocessed: PreprocessedImage,
        lifelines: list[int]
    ) -> list[MessageArrow]:
        """
        水平線セグメントを検出し、矢印方向を判定後、ライフラインにマッピング

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト（ObjectHeaderから取得）

        Returns:
            list[MessageArrow]: 検出されたメッセージ矢印（上から下へソート済み）
        """
        ...
```

**依存関係**:

- OpenCV (`cv2.HoughLinesP` for 水平線検出, 輪郭検出で矢印形状判定)
- Pydantic v2

**設計ノート**:

- Hough変換（`HoughLinesP`）で水平線セグメントを検出
- 端点近傍で三角形または「>」形状を検出して方向判定
- [research.md](./research.md) リスク2に対応し、`LIFELINE_MATCH_TOLERANCE`をパラメータ化

---

### コンポーネント4: OCREngine

**責務**: pytesseractを使用してテキスト領域とバウンディングボックスを抽出

**要件トレース**: 要件4（メッセージラベル抽出）、要件2（オブジェクト名取得）

**パブリックインターフェース**:

``` python
class OCRResult(BaseModel):
    """OCR抽出結果"""
    text: str
    bounding_box: tuple[int, int, int, int]  # (left, top, width, height)
    confidence: float = Field(..., ge=-1.0, le=100.0)
    color: str | None = Field(None, description="テキスト色: red, blue, または None")

class OCREngine:
    """OCRエンジンコンポーネント"""

    def __init__(self, config: str = "--oem 3 --psm 6"):
        """
        Args:
            config: Tesseract設定オプション（[research.md](./research.md) トピック2参照）
        """
        self.config = config

    def extract_text_regions(
        self,
        image: NDArray[np.uint8]
    ) -> list[OCRResult]:
        """
        画像全体からテキスト領域を抽出

        Args:
            image: 入力画像（グレースケールまたはカラー）

        Returns:
            list[OCRResult]: 抽出されたテキスト領域
        """
        ...

    def extract_from_region(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int]
    ) -> str:
        """
        指定領域からテキストを抽出

        Args:
            image: 入力画像
            bbox: バウンディングボックス (x, y, width, height)

        Returns:
            str: 抽出されたテキスト
        """
        ...
```

**依存関係**:

- pytesseract (`image_to_data`, [research.md](./research.md) トピック2参照)
- OpenCV (色判定用)
- Pydantic v2

**設計ノート**:

- `image_to_data(output_type=pytesseract.Output.DICT)`で単語レベルバウンディングボックスを取得
- テキスト色判定：カラー画像の該当領域からRGB値を取得し、赤（R > threshold）、青（B > threshold）を判定
- 信頼度が低い結果はログに記録（[research.md](./research.md) リスク1対策）

---

### コンポーネント4.5: MessageLabelMapper

**責務**: メッセージ矢印とOCRラベルをY座標近傍でマッピング

**要件トレース**: 要件4（メッセージラベル抽出と色情報の付与）

**パブリックインターフェース**:

```python
class MessageLabelMapper:
    """メッセージラベルマッピングコンポーネント"""

    LABEL_SEARCH_RANGE: tuple[int, int] = (5, 30)  # 上下のピクセル範囲（制約条件より）

    def map_labels_to_messages(
        self,
        messages: list[MessageArrow],
        ocr_results: list[OCRResult]
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
        ...
```

**依存関係**:

- Pydantic v2（MessageArrow, OCRResult）
- Python標準ライブラリ（math, logging）

**設計ノート**:

- 要件4（受け入れ基準4-2, 4-3）に基づき、矢印のY座標から上下5〜30ピクセル範囲のテキストを検索
- 複数のメッセージが近接している場合、バウンディングボックスのX座標も考慮してマッチング精度を向上
- `LABEL_SEARCH_RANGE` はパラメータ化され、ConversionConfigから調整可能

---

### コンポーネント5: SelfCallDetector

**責務**: 自己呼び出しを表す円形状を検出

**要件トレース**: 要件5（自己呼び出し検出）

**パブリックインターフェース**:

```python
class SelfCall(BaseModel):
    """自己呼び出し検出結果"""
    lifeline_x: int = Field(..., ge=0)
    y: int = Field(..., ge=0, description="円の中心Y座標")
    label: str | None = Field(None, description="自己呼び出しメッセージラベル")
    confidence: float = Field(..., ge=-1.0, le=100.0)

class SelfCallDetector:
    """自己呼び出し検出コンポーネント"""

    def __init__(self, ocr_engine: OCREngine):
        self.ocr_engine = ocr_engine

    def detect(
        self,
        preprocessed: PreprocessedImage,
        lifelines: list[int]
    ) -> list[SelfCall]:
        """
        Hough変換で円形状を検出し、ライフライン近傍のものを自己呼び出しとして識別

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト

        Returns:
            list[SelfCall]: 検出された自己呼び出し（Y座標ソート済み）
        """
        ...
```

**依存関係**:

- OpenCV (`cv2.HoughCircles`, [research.md](./research.md) 受け入れ基準5-1参照)
- OCREngine
- Pydantic v2

---

### コンポーネント6: NoteAnnotationDetector

**責務**: ノート注釈を表す矩形を検出し、テキストを抽出

**要件トレース**: 要件6（ノート注釈検出）

**パブリックインターフェース**:

``` python
class NoteAnnotation(BaseModel):
    """ノート注釈検出結果"""
    bounding_box: tuple[int, int, int, int]
    text: str
    related_lifeline: int | None = Field(None, description="関連するライフラインX座標")
    confidence: float = Field(..., ge=-1.0, le=100.0)

class NoteAnnotationDetector:
    """ノート注釈検出コンポーネント"""

    def __init__(self, ocr_engine: OCREngine):
        self.ocr_engine = ocr_engine

    def detect(
        self,
        preprocessed: PreprocessedImage,
        lifelines: list[int]
    ) -> list[NoteAnnotation]:
        """
        矩形領域を検出し、重要なテキストを含むものをノートとして識別

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト

        Returns:
            list[NoteAnnotation]: 検出されたノート注釈（Y座標ソート済み）
        """
        ...
```

**依存関係**:

- OpenCV (`findContours`)
- OCREngine
- Pydantic v2

---

### コンポーネント7: ActivationBarDetector

**責務**: アクティベーションバーを表す垂直矩形を検出

**要件トレース**: 要件7（アクティベーションバー検出）

**パブリックインターフェース**:

``` python
class ActivationBar(BaseModel):
    """アクティベーションバー検出結果"""
    lifeline_x: int = Field(..., ge=0)
    y_start: int = Field(..., ge=0)
    y_end: int = Field(..., ge=0)

class ActivationBarDetector:
    """アクティベーションバー検出コンポーネント"""

    def detect(
        self,
        preprocessed: PreprocessedImage,
        lifelines: list[int]
    ) -> list[ActivationBar]:
        """
        ライフライン近傍の垂直矩形を検出

        Args:
            preprocessed: 前処理済み画像
            lifelines: ライフラインX座標リスト

        Returns:
            list[ActivationBar]: 検出されたアクティベーションバー
        """
        ...
```

**依存関係**:

- OpenCV (`findContours`でアスペクト比が高い矩形を検出)
- Pydantic v2

---

### コンポーネント8: FragmentDetector

**責務**: フラグメント（alt/loop等）を表す大きな矩形フレームを検出

**要件トレース**: 要件8（フラグメント検出）

**パブリックインターフェース**:

``` python
class FragmentType(str, Enum):
    """フラグメントタイプ"""
    ALT = "alt"
    LOOP = "loop"
    OPT = "opt"
    PAR = "par"
    GROUP = "group"  # ラベルなしの場合

class Fragment(BaseModel):
    """フラグメント検出結果"""
    type: FragmentType
    title: str | None = Field(None, description="フラグメントタイトル")
    bounding_box: tuple[int, int, int, int]
    confidence: float = Field(..., ge=-1.0, le=100.0)

class FragmentDetector:
    """フラグメント検出コンポーネント"""

    def __init__(self, ocr_engine: OCREngine):
        self.ocr_engine = ocr_engine

    def detect(
        self,
        preprocessed: PreprocessedImage
    ) -> list[Fragment]:
        """
        大きな矩形フレームを検出し、左上ラベルからタイプを識別

        Args:
            preprocessed: 前処理済み画像

        Returns:
            list[Fragment]: 検出されたフラグメント（Y座標ソート済み）
        """
        ...
```

**依存関係**:

- OpenCV (`findContours`)
- OCREngine
- Pydantic v2

---

### コンポーネント9: PlantUMLGenerator

**責務**: 検出された要素からPlantUMLコードを生成

**要件トレース**: 要件9（PlantUMLコード生成）

**パブリックインターフェース**:

``` python
from pathlib import Path

class SequenceDiagramElements(BaseModel):
    """シーケンス図要素の集約（外部データ）"""
    objects: list[ObjectHeader]
    messages: list[MessageArrow]
    self_calls: list[SelfCall]
    notes: list[NoteAnnotation]
    activations: list[ActivationBar]
    fragments: list[Fragment]
    message_labels: dict[int, OCRResult]  # message Y座標 -> ラベル

class PlantUMLGenerator:
    """PlantUML生成コンポーネント"""

    def generate(
        self,
        elements: SequenceDiagramElements
    ) -> str:
        """
        シーケンス図要素からPlantUMLコードを生成

        Args:
            elements: 検出された全要素

        Returns:
            str: PlantUMLコード（@startuml ... @enduml）

        処理順序:
            1. オブジェクト宣言（participant）- 左から右へ
            2. 全イベント要素をY座標でソート（メッセージ、自己呼び出し、ノート）
            3. フラグメントはY座標範囲でグルーピング
            4. ソート済みイベントを順にPlantUML構文に変換
            5. アクティベーションバー範囲内のメッセージに++/--構文を付与
        """
        ...

    def save_to_file(
        self,
        plantuml_code: str,
        output_path: Path
    ) -> None:
        """
        PlantUMLコードをファイルに保存

        Args:
            plantuml_code: PlantUMLコード
            output_path: 出力ファイルパス（.puml）
        """
        ...
```

**依存関係**:

- PlantUML構文仕様 v1.2025.0（[research.md](./research.md) トピック3参照）
- Pydantic v2

**設計ノート**:

- 色付きテキスト：`[#red]`、`[#blue]`構文を使用（制約条件に準拠）
- ノート：`note over [object_name] : [text]`構文（制約条件に準拠）
- アクティベーション：`++`/`--`構文
- フラグメント：`alt`/`loop`等のキーワード
- [research.md](./research.md) リスク3に対応し、生成コードの構文検証ロジックを実装

---

### コンポーネント10: PipelineOrchestrator

**責務**: パイプライン全体の制御とエラーハンドリング

**要件トレース**: 全要件（統合制御）、非機能要件（エラー耐性）

**パブリックインターフェース**:

``` python
from pydantic import FilePath

class ConversionConfig(BaseModel):
    """変換設定（外部データ）"""
    input_path: FilePath
    output_path: Path
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    lifeline_match_tolerance: int = Field(20, ge=1, le=100)
    label_search_range: tuple[int, int] = Field((5, 30), description="(min, max) pixels")

class PipelineOrchestrator:
    """パイプライン制御コンポーネント"""

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
        """依存性注入によりすべてのコンポーネントを受け取る"""
        ...

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
        """
        ...
```

**依存関係**:

- すべての検出・生成コンポーネント
- Pydantic v2
- rich（進捗表示とログ）

**設計ノート**:

- 部分的な検出失敗時もプロセスを継続（非機能要件：エラー耐性）
- 各ステージの失敗をログに記録（非機能要件：ログ出力）
- 依存性注入により、各コンポーネントのモック化とテストが容易（設計原則：テスタビリティ）

---

### コンポーネント11: CLI

**責務**: コマンドライン引数解析と入出力制御

**要件トレース**: 要件1（画像入力）、要件9（ファイル保存）、非機能要件（利用性）

**パブリックインターフェース**:

``` python
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def convert(
    input_dir: Path = typer.Option(
        Path("input"),
        "--input",
        "-i",
        help="入力ディレクトリ（PNG/JPG画像）"
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output",
        "-o",
        help="出力ディレクトリ（.pumlファイル）"
    ),
) -> None:
    """
    シーケンス図画像をPlantUMLに変換

    Examples:
        $ sequence-converter convert
        $ sequence-converter convert --input ./images --output ./plantuml
    """
    ...

if __name__ == "__main__":
    app()
```

**依存関係**:

- typer（[research.md](./research.md) 技術判断3参照）
- PipelineOrchestrator
- rich（進捗バー、エラー表示）

---

## データモデル

### エンティティ1: SequenceDiagramElements

```python
class SequenceDiagramElements(BaseModel):
    """シーケンス図要素の集約"""
    objects: list[ObjectHeader]
    messages: list[MessageArrow]
    self_calls: list[SelfCall]
    notes: list[NoteAnnotation]
    activations: list[ActivationBar]
    fragments: list[Fragment]
    message_labels: dict[int, OCRResult]
```

**説明**: パイプライン全体で検出された全要素を集約するデータモデル。PlantUMLGenerator への入力として使用。

---

### エンティティ2: PreprocessedImage

```python
@dataclass
class PreprocessedImage:
    """前処理済み画像データ"""
    grayscale: NDArray[np.uint8]
    binary: NDArray[np.uint8]
    original_shape: tuple[int, int]
```

**説明**: 内部データ構造として標準dataclassを使用（[research.md](./research.md) 技術判断2参照）。パフォーマンス最適化のため、Pydanticのバリデーションオーバーヘッドを回避。

---

## エラーハンドリング戦略

### エラー分類と処理方針

| エラー種類 | 処理方針 | 要件トレース |
|-----------|---------|--------------|
| **入力ファイルエラー**（存在しない、破損、非対応フォーマット） | エラーメッセージを表示し、処理を中断 | 受け入れ基準1-2, 1-3 |
| **検出部分失敗**（一部のオブジェクト、矢印、ノート等が検出できない） | ログに記録し、処理を継続 | 受け入れ基準2-5, 3-5, 非機能要件（エラー耐性） |
| **OCR低信頼度** | ログに警告を出力し、結果を使用 | [research.md](./research.md) リスク1 |
| **ライフラインマッチング失敗** | ログに記録し、該当メッセージをスキップ | 受け入れ基準3-5 |
| **PlantUML生成エラー** | エラーメッセージを表示し、処理を中断 | 要件9 |

### エラーメッセージ例

```python
# 入力ファイルエラー
raise FileNotFoundError(f"Input file not found: {input_path}")
raise ValueError(f"Unsupported file format: {input_path.suffix}. Supported: .png, .jpg")

# 検出部分失敗（ログのみ）
logger.warning(f"Failed to detect object header at bbox {bbox}. Continuing with remaining objects.")
logger.warning(f"Low OCR confidence ({confidence:.1f}%) for text '{text}' at {bbox}")
```

## ログ戦略

### ログレベルと用途

| レベル | 用途 | 例 |
|--------|------|-----|
| **INFO** | 処理進捗 | "Preprocessing completed", "Detected 5 objects" |
| **WARNING** | 検出失敗、低信頼度 | "Failed to match arrow to lifeline", "Low OCR confidence" |
| **ERROR** | 回復不可能なエラー | "Unsupported file format", "PlantUML generation failed" |
| **DEBUG** | 詳細なデバッグ情報 | "Contour count: 42", "Lifeline positions: [120, 240, 360]" |

### ログ出力

- **ライブラリ**: Python標準 `logging` + `rich.logging.RichHandler`（ステアリング tech.md に準拠）
- **フォーマット**: `[timestamp] [level] [module] message`
- **出力先**: 標準エラー出力（stderr）

**設計ノート**:

- 非機能要件（ログ出力）に準拠
- ユーザーがログレベルを環境変数またはCLIオプションで制御可能にする

---

## テスタビリティ

### テスト可能性を確保するための設計

1. **依存性注入**:
   - すべてのコンポーネント（OCREngine, 各Detector, PlantUMLGenerator）はコンストラクタで依存関係を受け取る
   - モックやスタブに差し替え可能

2. **インターフェース分離**:
   - 各コンポーネントは単一責任を持ち、独立してテスト可能
   - 例：`ObjectHeaderDetector`は`OCREngine`に依存するが、モックOCREngineを注入してテスト

3. **型安全性**:
   - すべてのパブリックインターフェースに型ヒントを付与（設計原則：型安全性）
   - Pydanticモデルによりデータバリデーションを自動化

4. **テストデータ**:
   - 受け入れ基準に記載のテスト画像 `input/3gpp23228_fig0501.png` と期待出力 `output/3gpp23228_fig0501.puml` を使用
   - ユニットテスト用に小規模なサンプル画像を追加作成

### テスト戦略（ステアリング tech.md 準拠）

- **ユニットテスト**: pytest による各コンポーネントの単体テスト
- **統合テスト**: サンプル画像を使用したパイプライン全体のテスト + PlantUMLツールでの検証（[research.md](./research.md) リスク3対策）
- **実行**: `uv run pytest`

---

## 設計上の重要な保証

### Y座標ソート順序の保証

**契約**: すべての検出コンポーネントは、検出結果をY座標順（上から下へ）にソートして返すことを保証します。

**対象コンポーネント**:

- `MessageArrowDetector.detect()` → `list[MessageArrow]` (Y座標ソート済み)
- `SelfCallDetector.detect()` → `list[SelfCall]` (Y座標ソート済み)
- `NoteAnnotationDetector.detect()` → `list[NoteAnnotation]` (Y座標ソート済み)
- `FragmentDetector.detect()` → `list[Fragment]` (Y座標ソート済み)

**理由**: PlantUMLシーケンス図は時系列（上から下）が意味を持つため、検出順序の一貫性が生成コードの正確性に直結します（受け入れ基準9-5「有効なシーケンス図を生成することを保証」）。

**PlantUMLGenerator の処理フロー**:

1. オブジェクト宣言（X座標順）
2. 全イベント要素（メッセージ、自己呼び出し、ノート）をY座標で統合ソート
3. フラグメントをY座標範囲でグルーピング
4. ソート済みイベントを順にPlantUML構文に変換

**テスト要件**: 各検出コンポーネントの単体テストで、複数要素が検出された場合にY座標昇順でソートされていることを検証する。

---

## 設計変更ログ

### 2025-12-13: デザインレビュー懸念事項対応

**変更内容**:

1. **MessageLabelMapper コンポーネントの追加** (コンポーネント4.5):
   - メッセージ矢印とOCRラベルのマッピング責務を明確化
   - 要件4（受け入れ基準4-2, 4-3）の実装ロジックを明示
   - `LABEL_SEARCH_RANGE` パラメータをConversionConfigから調整可能に

2. **PreprocessingConfig の追加** (コンポーネント1):
   - 画像前処理パラメータ（blur_kernel_size, binary_threshold_method, noise_removal_method）を調整可能に
   - デフォルト値を明示し、様々な画像品質への適応性を向上
   - ConversionConfigに統合してCLIから制御可能

3. **Y座標ソート順序の契約明示**:
   - すべての検出コンポーネントで「Y座標ソート済み」を戻り値の契約として明示
   - PlantUMLGenerator の処理順序を疑似コードで文書化
   - テスト要件として単体テストでのソート検証を追加

**影響範囲**:

- PipelineOrchestrator: MessageLabelMapper を依存関係に追加
- ConversionConfig: PreprocessingConfig フィールドを追加
- 全検出コンポーネント: インターフェース契約にY座標ソート保証を追加

---

**次のステップ**: `/kiro:spec-tasks sequence-converter-core` を実行してタスクを生成してください。

**ディスカバリーログ**: 本設計は [research.md](./research.md) の調査結果に基づいており、OpenCV/pytesseract/PlantUMLの最新ベストプラクティス、Pythonモジュラーアーキテクチャパターン、Pydantic v2バリデーション戦略を統合しています。
