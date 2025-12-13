# Implementation Tasks: sequence-converter-core

> **作成日時**: 2025-12-13T05:09:24Z
> **要件トレース**: [requirements.md](./requirements.md)
> **設計トレース**: [design.md](./design.md)

---

## タスク概要

本ドキュメントは、シーケンス図画像からPlantUMLテキストへの変換ツール（sequence-converter-core）の実装タスクを定義します。パイプラインアーキテクチャに基づき、11のコンポーネントを段階的に実装します。

**総タスク数**: 13メジャータスク、48サブタスク

---

## 1. プロジェクトセットアップとデータモデル定義

**要件カバレッジ**: 1, 2, 3, 4, 5, 6, 7, 8, 9

プロジェクトの基盤となるディレクトリ構造、依存関係、共通データモデルを構築する。

### 1.1 プロジェクト構造とpyproject.toml作成

- [ ] `src/sequence_converter/` ディレクトリ構造を作成する
- [ ] `pyproject.toml` でプロジェクトメタデータと依存関係を定義する（Python 3.10+, opencv-python, pytesseract, pydantic v2, typer, rich, numpy, Pillow）
- [ ] `tests/` ディレクトリを作成し、`input/` と `output/` ディレクトリを配置する
- [ ] テスト画像 `input/3gpp23228_fig0501.png` を配置する

### 1.2 Pydanticデータモデルの定義

- [ ] `src/sequence_converter/models.py` を作成する
- [ ] `PreprocessingConfig`, `ObjectHeader`, `MessageArrow`, `ArrowDirection`, `OCRResult`, `SelfCall`, `NoteAnnotation`, `ActivationBar`, `Fragment`, `FragmentType`, `SequenceDiagramElements`, `ConversionConfig` をPydanticモデルとして定義する
- [ ] `PreprocessedImage` をdataclassとして定義する

### 1.3 ログ設定とエラーハンドリングユーティリティ

- [ ] `src/sequence_converter/logger.py` を作成し、rich.logging.RichHandlerを使ったロガーを設定する
- [ ] カスタム例外クラス（`UnsupportedFormatError`, `ImageProcessingError`）を定義する

---

## 2. 画像前処理コンポーネントの実装

**要件カバレッジ**: 1

入力画像をOpenCVで読み込み、グレースケール変換、二値化、ノイズ除去を実行する。

### 2.1 ImagePreprocessorクラスの実装

- [ ] `src/sequence_converter/preprocessing.py` を作成する
- [ ] `ImagePreprocessor` クラスを実装し、`PreprocessingConfig` を受け取るコンストラクタを定義する
- [ ] `preprocess(image_path: str) -> PreprocessedImage` メソッドを実装する（PIL/OpenCVで画像読み込み、グレースケール変換、二値化（THRESH_BINARY + THRESH_OTSU）、ノイズ除去）

### 2.2 画像前処理のユニットテスト

- [ ] `tests/test_preprocessing.py` を作成する
- [ ] サポート形式（PNG/JPG）の画像が正しく読み込まれることをテストする
- [ ] 非サポート形式でValueErrorが発生することをテストする
- [ ] 破損画像でエラーハンドリングが機能することをテストする
- [ ] 前処理後の画像がgrayscale, binary属性を持つことを検証する

---

## 3. OCRエンジンコンポーネントの実装

**要件カバレッジ**: 2, 4

pytesseractを使用してテキスト領域を抽出し、バウンディングボックスとテキスト色情報を取得する。

### 3.1 OCREngineクラスの実装

- [ ] `src/sequence_converter/ocr.py` を作成する
- [ ] `OCREngine` クラスを実装し、Tesseract設定（`--oem 3 --psm 6`）を受け取るコンストラクタを定義する
- [ ] `extract_text_regions(image: NDArray) -> list[OCRResult]` メソッドを実装する（`image_to_data(output_type=pytesseract.Output.DICT)` 使用）
- [ ] `extract_from_region(image: NDArray, bbox: tuple) -> str` メソッドを実装する
- [ ] テキスト色判定ロジック（RGB値から赤・青・デフォルトを判定）を実装する

### 3.2 OCRエンジンのユニットテスト

- [ ] `tests/test_ocr.py` を作成する
- [ ] サンプル画像からテキスト領域が抽出されることをテストする
- [ ] バウンディングボックス座標が正しく取得されることをテストする
- [ ] テキスト色判定（赤・青・デフォルト）が機能することをテストする
- [ ] 低信頼度テキストでログ警告が出力されることをテストする

---

## 4. オブジェクトヘッダー検出コンポーネントの実装

**要件カバレッジ**: 2

画像上部から矩形を検出し、OCRでオブジェクト名を取得してライフライン位置を記録する。

### 4.1 ObjectHeaderDetectorクラスの実装

- [ ] `src/sequence_converter/detectors/object_header.py` を作成する
- [ ] `ObjectHeaderDetector` クラスを実装し、OCREngineを依存性注入で受け取る
- [ ] `detect(preprocessed: PreprocessedImage) -> list[ObjectHeader]` メソッドを実装する（`cv2.findContours` で矩形検出、OCRでテキスト抽出、X座標順にソート）
- [ ] 検出失敗時のログ記録とエラー耐性を実装する

### 4.2 オブジェクトヘッダー検出のユニットテスト

- [ ] `tests/test_detectors/test_object_header.py` を作成する
- [ ] 複数の矩形が検出され、左から右へソートされることをテストする
- [ ] OCRが矩形領域から正しくテキストを抽出することをテストする
- [ ] ライフラインX座標が矩形中心として記録されることをテストする
- [ ] 一部矩形の検出失敗時もログを記録し処理を継続することをテストする

---

## 5. メッセージ矢印検出コンポーネントの実装

**要件カバレッジ**: 3

水平線セグメントを検出し、矢印方向を判定してライフラインにマッピングする。

### 5.1 MessageArrowDetectorクラスの実装

- [ ] `src/sequence_converter/detectors/message_arrow.py` を作成する
- [ ] `MessageArrowDetector` クラスを実装する
- [ ] `detect(preprocessed: PreprocessedImage, lifelines: list[int]) -> list[MessageArrow]` メソッドを実装する（HoughLinesP で水平線検出、端点で矢印形状判定、ライフラインマッチング（20px許容範囲）、Y座標順ソート）
- [ ] ライフラインマッチング失敗時のログ記録を実装する

### 5.2 メッセージ矢印検出のユニットテスト

- [ ] `tests/test_detectors/test_message_arrow.py` を作成する
- [ ] 水平線セグメントが検出されることをテストする
- [ ] 矢印方向（LEFT_TO_RIGHT / RIGHT_TO_LEFT）が正しく判定されることをテストする
- [ ] ライフラインマッチング（20px許容範囲）が機能することをテストする
- [ ] 検出結果がY座標順にソートされることをテストする
- [ ] マッチング失敗時にログ記録され処理が継続することをテストする

---

## 6. メッセージラベルマッパーコンポーネントの実装

**要件カバレッジ**: 4

メッセージ矢印とOCRラベルをY座標近傍（上下5〜30ピクセル）でマッピングする。

### 6.1 MessageLabelMapperクラスの実装

- [ ] `src/sequence_converter/detectors/label_mapper.py` を作成する
- [ ] `MessageLabelMapper` クラスを実装する
- [ ] `map_labels_to_messages(messages: list[MessageArrow], ocr_results: list[OCRResult]) -> dict[int, OCRResult]` メソッドを実装する（Y座標から上下5〜30px範囲のテキスト検索、最も近いテキストを選択、X座標も考慮）
- [ ] マッチング失敗時のログ記録を実装する

### 6.2 メッセージラベルマッピングのユニットテスト

- [ ] `tests/test_detectors/test_label_mapper.py` を作成する
- [ ] 矢印Y座標から上下5〜30px範囲のテキストが検索されることをテストする
- [ ] 複数テキストが範囲内に存在する場合、最も近いものが選択されることをテストする
- [ ] マッチしなかった矢印に対してログが記録されることをテストする

---

## 7. 自己呼び出し検出コンポーネントの実装

**要件カバレッジ**: 5

Hough変換で円形状を検出し、ライフライン近傍のものを自己呼び出しとして識別する。

### 7.1 SelfCallDetectorクラスの実装

- [ ] `src/sequence_converter/detectors/self_call.py` を作成する
- [ ] `SelfCallDetector` クラスを実装し、OCREngineを依存性注入で受け取る
- [ ] `detect(preprocessed: PreprocessedImage, lifelines: list[int]) -> list[SelfCall]` メソッドを実装する（HoughCircles で円検出、ライフライン近傍判定、円の右側テキスト検索、Y座標順ソート）

### 7.2 自己呼び出し検出のユニットテスト

- [ ] `tests/test_detectors/test_self_call.py` を作成する
- [ ] 円形状が検出されることをテストする
- [ ] 円の中心X座標がライフライン近傍であることを判定できることをテストする
- [ ] 円の右側テキストが自己呼び出しラベルとして取得されることをテストする
- [ ] 検出結果がY座標順にソートされることをテストする

---

## 8. ノート注釈検出コンポーネントの実装

**要件カバレッジ**: 6

矩形領域を検出し、重要なテキストを含むものをノート注釈として識別する。

### 8.1 NoteAnnotationDetectorクラスの実装

- [ ] `src/sequence_converter/detectors/note_annotation.py` を作成する
- [ ] `NoteAnnotationDetector` クラスを実装し、OCREngineを依存性注入で受け取る
- [ ] `detect(preprocessed: PreprocessedImage, lifelines: list[int]) -> list[NoteAnnotation]` メソッドを実装する（findContours で矩形検出、テキストコンテンツ確認、最も近いライフラインに関連付け、Y座標順ソート）

### 8.2 ノート注釈検出のユニットテスト

- [ ] `tests/test_detectors/test_note_annotation.py` を作成する
- [ ] ライフラインと重なる／独立する矩形が検出されることをテストする
- [ ] ノート内のテキストがOCRで抽出されることをテストする
- [ ] ノートが最も近いライフラインに関連付けられることをテストする
- [ ] 検出結果がY座標順にソートされることをテストする

---

## 9. アクティベーションバーとフラグメント検出コンポーネントの実装

**要件カバレッジ**: 7, 8

垂直矩形からアクティベーションバーを、大きな矩形フレームからフラグメントを検出する。

### 9.1 ActivationBarDetectorクラスの実装

- [ ] `src/sequence_converter/detectors/activation_bar.py` を作成する
- [ ] `ActivationBarDetector` クラスを実装する
- [ ] `detect(preprocessed: PreprocessedImage, lifelines: list[int]) -> list[ActivationBar]` メソッドを実装する（findContours でアスペクト比が高い垂直矩形を検出、Y座標範囲を記録）

### 9.2 FragmentDetectorクラスの実装

- [ ] `src/sequence_converter/detectors/fragment.py` を作成する
- [ ] `FragmentDetector` クラスを実装し、OCREngineを依存性注入で受け取る
- [ ] `detect(preprocessed: PreprocessedImage) -> list[Fragment]` メソッドを実装する（findContours で大きな矩形フレーム検出、左上ラベルからタイプ識別（alt/loop等）、Y座標順ソート）

### 9.3 アクティベーションバーとフラグメント検出のユニットテスト

- [ ] `tests/test_detectors/test_activation_bar.py` を作成し、垂直矩形検出とY座標範囲記録をテストする
- [ ] `tests/test_detectors/test_fragment.py` を作成し、フラグメントタイプ識別とラベルなし時のグループ処理をテストする
- [ ] フラグメント検出結果がY座標順にソートされることをテストする

---

## 10. PlantUMLジェネレーターコンポーネントの実装

**要件カバレッジ**: 9

検出された全要素からPlantUMLシーケンス図コードを生成する。

### 10.1 PlantUMLGeneratorクラスの実装

- [ ] `src/sequence_converter/generator.py` を作成する
- [ ] `PlantUMLGenerator` クラスを実装する
- [ ] `generate(elements: SequenceDiagramElements) -> str` メソッドを実装する（オブジェクト宣言、イベントY座標ソート、フラグメントグルーピング、PlantUML構文変換、アクティベーション構文付与、色情報付与）
- [ ] `save_to_file(plantuml_code: str, output_path: Path) -> None` メソッドを実装する

### 10.2 PlantUML生成のユニットテスト

- [ ] `tests/test_generator.py` を作成する
- [ ] オブジェクト宣言が左から右へ正しく生成されることをテストする
- [ ] メッセージ、自己呼び出し、ノートがY座標順に変換されることをテストする
- [ ] 色情報（[#red], [#blue]）が正しく付与されることをテストする
- [ ] ノート構文（note over [object_name] : [text]）が正しく生成されることをテストする
- [ ] アクティベーション構文（++/--）が範囲内メッセージに付与されることをテストする
- [ ] フラグメント（alt/loop等）が正しく生成されることをテストする
- [ ] 生成されたPlantUMLが構文的に正しいことを検証する

---

## 11. パイプラインオーケストレーターの実装

**要件カバレッジ**: 1, 2, 3, 4, 5, 6, 7, 8, 9

全コンポーネントを統合し、エラーハンドリングを含む変換パイプラインを制御する。

### 11.1 PipelineOrchestratorクラスの実装

- [ ] `src/sequence_converter/pipeline.py` を作成する
- [ ] `PipelineOrchestrator` クラスを実装し、全コンポーネントを依存性注入で受け取る
- [ ] `convert(config: ConversionConfig) -> str` メソッドを実装する（前処理 → オブジェクト検出 → メッセージ検出 → OCR → ラベルマッピング → その他検出 → PlantUML生成のフローを実行）
- [ ] 部分的な検出失敗時もプロセスを継続するエラーハンドリングを実装する
- [ ] 各ステージでログ出力を実装する

### 11.2 パイプライン統合テスト

- [ ] `tests/test_pipeline.py` を作成する
- [ ] テスト画像 `input/3gpp23228_fig0501.png` を使用してパイプライン全体をテストする
- [ ] 期待出力 `output/3gpp23228_fig0501.puml` と生成結果を比較する
- [ ] 部分的な検出失敗時もプロセスが継続することをテストする
- [ ] エラーログが適切に記録されることをテストする

---

## 12. CLIインターフェースの実装

**要件カバレッジ**: 1, 9

typerを使用してコマンドライン引数を解析し、入出力を制御する。

### 12.1 CLIエントリーポイントの実装

- [ ] `src/sequence_converter/cli.py` を作成する
- [ ] typerアプリケーションを作成し、`convert` コマンドを実装する（`--input`, `--output` オプション、デフォルト: `input/`, `output/`）
- [ ] rich.consoleでエラー表示と進捗バーを実装する
- [ ] PipelineOrchestratorを呼び出し、結果を標準出力とファイルに保存する

### 12.2 CLI実行テスト

- [ ] `tests/test_cli.py` を作成する
- [ ] デフォルト引数（input/, output/）でコマンドが実行されることをテストする
- [ ] カスタム引数（--input, --output）で正しく動作することをテストする
- [ ] 非対応ファイル形式でエラーメッセージが表示されることをテストする

---

## 13. エンドツーエンド統合テストとドキュメント整備

**要件カバレッジ**: 1, 2, 3, 4, 5, 6, 7, 8, 9

実際のシーケンス図画像を使用した統合テストとREADMEの作成。

### 13.1 エンドツーエンド統合テスト

- [ ] `tests/test_e2e.py` を作成する
- [ ] `input/3gpp23228_fig0501.png` を処理し、`output/3gpp23228_fig0501.puml` が生成されることをテストする
- [ ] 生成されたPlantUMLがPlantUMLツールで有効なシーケンス図として表示されることを検証する（オプショナル: PlantUML CLIを使用）
- [ ] 複数の画像ファイルを一括処理できることをテストする

### 13.2 開発環境セットアップとpre-commitフック設定

- [ ] `.pre-commit-config.yaml` を作成し、black, isort, flake8を設定する
- [ ] `uv run pre-commit install` で pre-commit フックを有効化する
- [ ] `uv run black .`, `uv run isort .`, `uv run flake8` が正常に動作することを確認する

### 13.3 README.mdとドキュメントの作成

- [ ] `README.md` を作成し、プロジェクト概要、インストール手順、使用例、依存関係、開発方法を記載する
- [ ] システム要件（Tesseract OCR 3.05+のインストール方法）を明記する
- [ ] サンプル実行例とPlantUML出力例を記載する

---

## タスク進捗トラッキング

- **フェーズ1（基盤構築）**: タスク1-3（完了: 0/3）
- **フェーズ2（検出コンポーネント）**: タスク4-9（完了: 0/6）
- **フェーズ3（生成と統合）**: タスク10-12（完了: 0/3）
- **フェーズ4（テストとドキュメント）**: タスク13（完了: 0/1）

**全体進捗**: 0/13タスク完了

---

## 次のステップ

実装を開始する準備が整いました。

**推奨開始順序**:
1. タスク1（プロジェクトセットアップ）
2. タスク2（画像前処理）
3. タスク3（OCRエンジン）
4. タスク4以降（検出コンポーネントを順次実装）

**実装開始コマンド**:
```bash
# 特定タスク実行（推奨）
/kiro:spec-impl sequence-converter-core 1.1

# 複数タスク実行（慎重に使用）
/kiro:spec-impl sequence-converter-core 1.1,1.2

# 全タスク実行（非推奨）
/kiro:spec-impl sequence-converter-core
```

**重要**: タスク間でコンテキストをクリアすることを推奨します。

---

**作成者ノート**: 本タスクリストは [design.md](./design.md) のコンポーネント構成と [requirements.md](./requirements.md) の受け入れ基準に基づいて生成されました。各タスクは1-3時間で完了可能な粒度に設計されています。
