# Research Log: sequence-converter-core

> **作成日時**: 2025-12-13T02:46:51Z
> **ディスカバリースコープ**: フルディスカバリー（新規機能・複雑な画像処理パイプライン）

---

## サマリー

シーケンス図画像からPlantUMLテキストへの変換という複雑な画像処理パイプラインを設計するため、OpenCV、pytesseract、PlantUMLの最新仕様とPythonモジュラーアーキテクチャパターンを調査しました。主要な発見として、(1) パイプラインパターンによるモジュール分離、(2) Pydantic v2によるデータ境界でのバリデーション、(3) OpenCVの輪郭検出と pytesseractのバウンディングボックス取得の組み合わせが最適なアプローチであることが確認されました。

## リサーチログ

### トピック1: OpenCV 輪郭検出とベストプラクティス

**調査内容**: シーケンス図要素（矩形、円、矢印）の検出に必要なOpenCV輪郭検出の最新手法

**情報源**:

- [Contour Detection using OpenCV (Python/C++)](https://learnopencv.com/contour-detection-using-opencv-python-c/)
- [OpenCV: Contours : Getting Started](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html)
- [Find and Draw Contours using OpenCV - Python - GeeksforGeeks](https://www.geeksforgeeks.org/python/find-and-draw-contours-using-opencv-python/)

**発見事項**:

- 前処理：グレースケール変換 → 二値化（threshold）が標準フロー
- 背景は黒、検出対象は白である必要がある
- `cv.findContours()` の引数：(1) ソース画像、(2) 輪郭取得モード、(3) 輪郭近似手法
- パフォーマンス：`RETR_LIST`（階層なし）と`RETR_EXTERNAL`（親輪郭のみ）が最速、`RETR_TREE`（全階層）が最遅
- メモリ最適化：`cv.CHAIN_APPROX_SIMPLE` により冗長点を削除し輪郭を圧縮

**設計への影響**: 画像前処理モジュールでグレースケール変換と二値化を実装。オブジェクトヘッダー、メッセージ矢印、ノート、フラグメント等の検出に`findContours()`を利用。`RETR_EXTERNAL`と`CHAIN_APPROX_SIMPLE`を採用してパフォーマンスとメモリ効率を最適化。

---

### トピック2: pytesseract OCR とバウンディングボックス

**調査内容**: OCRによるテキスト抽出とバウンディングボックス情報の取得方法

**情報源**:

- [How To Use OCR Bounding Boxes | by Michael Orozco-Fletcher | Medium](https://medium.com/@michael71314/how-to-use-ocr-bounding-boxes-c00303bc11c4)
- [Python OCR Tutorial: Tesseract, Pytesseract, and OpenCV](https://nanonets.com/blog/ocr-with-tesseract/)
- [pytesseract · PyPI](https://pypi.org/project/pytesseract/)

**発見事項**:

- 文字レベル：`pytesseract.image_to_boxes(img)` を使用
- 単語レベル：`pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)` を使用（Tesseract 3.05+）
- バウンディングボックスデータ：`left`, `top`, `width`, `height` が含まれる
- OCR信頼度：0-100の範囲、-1は信頼度なし
- 前処理の重要性：コントラスト改善とノイズ除去が精度向上に不可欠
- 設定オプション：`--psm 6`（行単位処理）、`--oem 3`（OCRエンジンモード3）

**設計への影響**: OCRモジュールで`image_to_data`を使用して単語レベルのバウンディングボックスを取得。メッセージラベル、オブジェクト名、ノートテキスト等の抽出に活用。前処理モジュールでコントラスト調整を実装。

---

### トピック3: PlantUML シーケンス図構文（2025版）

**調査内容**: PlantUML シーケンス図の最新構文仕様（バージョン 1.2025.0）

**情報源**:

- [Sequence Diagram syntax and features](https://plantuml.com/sequence-diagram)
- [UML Sequence Diagram — Ashley's PlantUML Doc](https://plantuml-documentation.readthedocs.io/en/latest/diagrams/sequence.html)
- [Drawing UML with PlantUML Language Reference Guide (Version 1.2025.0)](https://pdf.plantuml.net/PlantUML_Language_Reference_Guide_en.pdf)

**発見事項**:

- 基本構文：`sender -> receiver : message`
- ダイアグラム開始/終了：`@startuml` / `@enduml`
- 参加者宣言：`participant`キーワードで順序制御可能
- 矢印タイプ：`->`（実線同期）、`-->`（点線非同期）
- アクティベーション：`++`（アクティベート）、`--`（デアクティベート）、`**`（インスタンス生成）、`!!`（破棄）
- オブジェクト生成：`create`キーワード
- ノート構文：`note over [object_name] : [text]`
- 色指定：`[#red]`、`[#blue]`等
- フラグメント：`alt`、`loop`等のキーワード

**設計への影響**: PlantUML生成モジュールでバージョン1.2025.0の構文に準拠。検出した要素を正しいPlantUML構文にマッピング。色情報の付与、ノート構文、フラグメント構文を実装。

---

### トピック4: Pythonモジュラーパイプラインアーキテクチャ

**調査内容**: 画像処理におけるモジュラーパイプラインパターンの実装方法

**情報源**:

- [GitHub - jagin/image-processing-pipeline: Modular image processing pipeline using OpenCV and Python generators](https://github.com/jagin/image-processing-pipeline)
- [Modular image processing pipeline using OpenCV and Python generators | Medium](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696)
- [Pipeline Pattern Implementation | Pythonic Implementation of Pipeline patterns | Medium](https://medium.com/@r.das699/pipeline-patterns-implementation-python-cd1e510a59a1)

**発見事項**:

- パイプ＆フィルターアーキテクチャ：データが連続的な処理ユニット（フィルター）を通過し、パイプで接続される
- モジュール性とスケーラビリティ：各処理ステップを独立したモジュールとして実装
- Pythonジェネレータの活用：メモリ効率的なデータフロー
- PynPoint, MCMICRO等の実装例：コア機能とパイプラインモジュールの分離

**設計への影響**: パイプラインパターンを採用。各検出ステップ（前処理 → オブジェクト検出 → メッセージ検出 → OCR → PlantUML生成）を独立したモジュールとして設計。拡張性と保守性を確保。

---

### トピック5: Pydantic v2 バリデーション（2025ベストプラクティス）

**調査内容**: Pydantic v2によるデータ構造バリデーションの最新ベストプラクティス

**情報源**:

- [Dataclasses - Pydantic Validation](https://docs.pydantic.dev/latest/concepts/dataclasses/)
- [Python Dataclass vs Pydantic: How to Choose | by Nikulsinh Rajput | Medium](https://medium.com/@hadiyolworld007/python-dataclass-vs-pydantic-how-to-choose-8649dce72d2b)
- [Pydantic: A Guide With Practical Examples | DataCamp](https://www.datacamp.com/tutorial/pydantic)

**発見事項**:

- 使い分け原則：内部ドメインオブジェクトには`dataclass`、境界（外部データ入力）には`Pydantic`
- Pydanticの利点：自動バリデーション、API/設定/ファイルI/Oに最適
- パフォーマンス：`model_validate_json()`が`model_validate(json.loads())`より効率的
- v2.8+：シーケンス型に`FailFast`アノテーション適用可能
- TypeAdapter：dataclassをラップしてPydanticのバリデーション・ダンプ・JSON Schemaメソッドを利用

**設計への影響**: 入力画像ファイル、OCR結果、検出要素等の外部データに対してPydantic v2のデータクラスを使用してバリデーション。内部の中間データ構造には標準dataclassを使用してパフォーマンス最適化。

---

## アーキテクチャパターン評価

### パターン候補1: パイプライン（Pipe-and-Filter）パターン

**メリット**:

- 各処理ステップが独立しており、テストが容易
- 新しい検出機能の追加が容易（拡張性）
- 各フィルターの責務が明確（単一責任原則）
- 画像処理の分野で実績あり（OpenCV + Pythonジェネレータの組み合わせ）

**デメリット**:

- パイプライン全体の制御ロジックが必要
- ステップ間のデータ受け渡しのオーバーヘッド

**採用判断**: ✅ **採用** - 要件の複雑さ（9つの異なる検出機能）と拡張性要求を考慮し、パイプラインパターンが最適。

---

### パターン候補2: レイヤードアーキテクチャ

**メリット**:

- 責務の明確な分離（プレゼンテーション、ビジネスロジック、データ）
- 一般的なWebアプリケーションパターン

**デメリット**:

- 画像処理フローに対してオーバーキル
- パイプライン的な処理フローに不適

**採用判断**: ❌ **不採用** - 画像処理パイプラインには適さない。

---

## 技術判断の記録

### 判断1: OpenCV vs Pillow for 画像処理

**選択肢**: OpenCV, Pillow, scikit-image

**決定**: OpenCV（opencv-python）

**理由**:

- 輪郭検出、Hough変換等の高度な図形検出機能が豊富
- パフォーマンスが優れている（C++実装）
- ステアリング（tech.md）で既に言及されている

**トレードオフ**: 学習曲線がやや急だが、機能の豊富さがそれを補う。

---

### 判断2: Pydantic v2 vs dataclass for データモデル

**選択肢**: Pydantic v2, 標準dataclass, attrs

**決定**: 境界でPydantic v2、内部でdataclass

**理由**:

- 外部データ（ファイル入力、OCR結果）のバリデーションにPydanticが最適
- 内部の中間データ構造には軽量なdataclassでパフォーマンス最適化
- ステアリング（tech.md）でPydantic v2が言及されている

**トレードオフ**: 2つのデータ構造を使い分ける複雑さ vs バリデーションとパフォーマンスのバランス

---

### 判断3: CLI フレームワーク - typer vs click

**選択肢**: typer, click, argparse

**決定**: typer

**理由**:

- 型ヒントベースでPydanticとの親和性が高い
- 自動ヘルプ生成とバリデーション
- ステアリング（tech.md）でtyperまたはclickが言及されており、typerの方がモダン

**トレードオフ**: typerはclick上に構築されているため、依存関係が増えるが、開発体験が向上。

---

## リスクと軽減策

### リスク1: OCR精度の低さ

**影響度**: 高

**発生確率**: 中

**軽減策**:

- 画像前処理（二値化、ノイズ除去、コントラスト調整）を徹底
- Tesseractの設定オプション（`--psm`, `--oem`）を最適化
- OCR信頼度（confidence）をログに記録し、低信頼度の結果を検出可能にする
- テスト画像`3gpp23228_fig0501.png`で精度を検証

---

### リスク2: 複雑なシーケンス図のレイアウト検出失敗

**影響度**: 中

**発生確率**: 中

**軽減策**:

- ライフラインマッチングの許容範囲（20ピクセル）をパラメータ化し、調整可能にする
- 検出失敗をログに記録し、部分的な失敗でもプロセスを継続（要件の非機能要件に準拠）
- エラーレポート機能を提供し、ユーザーが手動で修正可能にする

---

### リスク3: PlantUML構文生成の正確性

**影響度**: 中

**発生確率**: 低

**軽減策**:

- PlantUML 1.2025.0の公式仕様に厳密に準拠
- 単体テストでPlantUML構文の正確性を検証
- 生成したPlantUMLを実際にPlantUMLツールで検証（統合テスト）

---

## 参考資料

- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [pytesseract Documentation](https://pypi.org/project/pytesseract/)
- [PlantUML Language Reference Guide (Version 1.2025.0)](https://pdf.plantuml.net/PlantUML_Language_Reference_Guide_en.pdf)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Modular image processing pipeline using OpenCV and Python generators](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696)
