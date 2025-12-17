#!/usr/bin/env python3
"""OCRエンジン選択機能の動作確認スクリプト"""

import sys

sys.path.insert(0, "src")

import numpy as np

from sequence_converter.models import OCREngineType
from sequence_converter.ocr import EasyOCRBackend, OCREngine, TesseractBackend


def test_tesseract_backend():
    """Tesseractバックエンドの動作確認"""
    print("=" * 60)
    print("Tesseract Backend Test")
    print("=" * 60)

    # OCREngineの初期化（デフォルト: Tesseract）
    ocr = OCREngine()
    print(f"✓ OCREngine initialized with engine_type: {ocr.engine_type}")
    print(f"✓ Backend type: {type(ocr.backend).__name__}")
    assert ocr.engine_type == OCREngineType.TESSERACT
    assert isinstance(ocr.backend, TesseractBackend)
    print("✓ Tesseract backend initialization: PASSED")
    print()


def test_easyocr_backend():
    """EasyOCRバックエンドの動作確認"""
    print("=" * 60)
    print("EasyOCR Backend Test")
    print("=" * 60)

    # OCREngineの初期化（EasyOCR）
    ocr = OCREngine(engine_type=OCREngineType.EASYOCR)
    print(f"✓ OCREngine initialized with engine_type: {ocr.engine_type}")
    print(f"✓ Backend type: {type(ocr.backend).__name__}")
    assert ocr.engine_type == OCREngineType.EASYOCR
    assert isinstance(ocr.backend, EasyOCRBackend)
    print("✓ EasyOCR backend initialization: PASSED")
    print()


def test_custom_config():
    """カスタム設定の動作確認"""
    print("=" * 60)
    print("Custom Configuration Test")
    print("=" * 60)

    # Tesseractカスタム設定
    custom_config = "--oem 1 --psm 3"
    ocr_tesseract = OCREngine(engine_type=OCREngineType.TESSERACT, tesseract_config=custom_config)
    print(f"✓ Tesseract custom config: {ocr_tesseract.backend.config}")
    assert ocr_tesseract.backend.config == custom_config

    # EasyOCRカスタム言語設定
    custom_languages = ["en", "ja", "ch_sim"]
    ocr_easyocr = OCREngine(engine_type=OCREngineType.EASYOCR, easyocr_languages=custom_languages)
    print(f"✓ EasyOCR custom languages: {ocr_easyocr.backend.languages}")
    assert ocr_easyocr.backend.languages == custom_languages

    print("✓ Custom configuration: PASSED")
    print()


def test_extract_text_regions():
    """extract_text_regionsメソッドの動作確認（簡易）"""
    print("=" * 60)
    print("Extract Text Regions Test (Basic)")
    print("=" * 60)

    # 白い画像を作成（テキストなし）
    image = np.ones((200, 400), dtype=np.uint8) * 255

    # Tesseractで実行
    ocr_tesseract = OCREngine(engine_type=OCREngineType.TESSERACT)
    results = ocr_tesseract.extract_text_regions(image)
    print(f"✓ Tesseract extracted {len(results)} text regions from blank image")
    assert isinstance(results, list)

    print("✓ Extract text regions: PASSED")
    print()


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("OCR Engine Selection Feature - Verification Script")
    print("=" * 60)
    print()

    try:
        test_tesseract_backend()
        test_easyocr_backend()
        test_custom_config()
        test_extract_text_regions()

        print("=" * 60)
        print("✓ All tests PASSED!")
        print("=" * 60)
        print()
        print("実装されたOCRエンジン選択機能:")
        print("  - OCREngineType.TESSERACT (デフォルト)")
        print("  - OCREngineType.EASYOCR")
        print()
        print("使用例:")
        print("  # Tesseract使用")
        print("  ocr = OCREngine(engine_type=OCREngineType.TESSERACT)")
        print()
        print("  # EasyOCR使用")
        print("  ocr = OCREngine(engine_type=OCREngineType.EASYOCR)")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
