"""Data models for sequence converter."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Configuration Models
# ============================================================================


class OCREngineType(str, Enum):
    """OCRエンジンタイプ"""

    TESSERACT = "tesseract"
    EASYOCR = "easyocr"


class PreprocessingConfig(BaseModel):
    """前処理設定（外部データ）"""

    blur_kernel_size: int = Field(5, ge=3, le=15, description="ブラーカーネルサイズ（奇数のみ）")
    binary_threshold_method: str = Field(
        "otsu", pattern="^(otsu|adaptive)$", description="二値化手法"
    )
    noise_removal_method: str = Field(
        "median", pattern="^(median|gaussian)$", description="ノイズ除去手法"
    )

    @field_validator("blur_kernel_size")
    @classmethod
    def validate_blur_kernel(cls, v: int) -> int:
        """ブラーカーネルサイズが奇数であることを検証"""
        if v % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        return v


class ConversionConfig(BaseModel):
    """変換設定（外部データ）"""

    input_path: Path
    output_path: Path
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    lifeline_match_tolerance: int = Field(20, ge=1, le=100)
    label_search_range: tuple[int, int] = Field((5, 30), description="(min, max) pixels")


# ============================================================================
# Internal Data Structures
# ============================================================================


@dataclass
class PreprocessedImage:
    """前処理済み画像データ（内部データ構造）"""

    grayscale: NDArray[np.uint8]
    binary: NDArray[np.uint8]
    original_shape: tuple[int, int]  # (height, width)


# ============================================================================
# Detection Result Models
# ============================================================================


class ObjectHeader(BaseModel):
    """オブジェクトヘッダー検出結果（外部データ）"""

    name: str = Field(..., min_length=1, description="オブジェクト名")
    lifeline_x: int = Field(..., ge=0, description="ライフラインX座標")
    bounding_box: tuple[int, int, int, int] = Field(..., description="(x, y, width, height)")
    confidence: float = Field(..., ge=-1.0, le=100.0, description="OCR信頼度")


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
    source_lifeline: Optional[int] = Field(None, description="送信元ライフラインX座標")
    dest_lifeline: Optional[int] = Field(None, description="宛先ライフラインX座標")


class OCRResult(BaseModel):
    """OCR抽出結果"""

    text: str
    bounding_box: tuple[int, int, int, int]  # (left, top, width, height)
    confidence: float = Field(..., ge=-1.0, le=100.0)
    color: Optional[str] = Field(None, description="テキスト色: red, blue, または None")


class SelfCall(BaseModel):
    """自己呼び出し検出結果"""

    lifeline_x: int = Field(..., ge=0)
    y: int = Field(..., ge=0, description="円の中心Y座標")
    label: Optional[str] = Field(None, description="自己呼び出しメッセージラベル")
    confidence: float = Field(..., ge=-1.0, le=100.0)


class NoteAnnotation(BaseModel):
    """ノート注釈検出結果"""

    bounding_box: tuple[int, int, int, int]
    text: str
    related_lifeline: Optional[int] = Field(None, description="関連するライフラインX座標")
    confidence: float = Field(..., ge=-1.0, le=100.0)


class ActivationBar(BaseModel):
    """アクティベーションバー検出結果"""

    lifeline_x: int = Field(..., ge=0)
    y_start: int = Field(..., ge=0)
    y_end: int = Field(..., ge=0)


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
    title: Optional[str] = Field(None, description="フラグメントタイトル")
    bounding_box: tuple[int, int, int, int]
    confidence: float = Field(..., ge=-1.0, le=100.0)


# ============================================================================
# Aggregated Data Models
# ============================================================================


class SequenceDiagramElements(BaseModel):
    """シーケンス図要素の集約（外部データ）"""

    objects: list[ObjectHeader]
    messages: list[MessageArrow]
    self_calls: list[SelfCall]
    notes: list[NoteAnnotation]
    activations: list[ActivationBar]
    fragments: list[Fragment]
    message_labels: dict[int, OCRResult]  # message Y座標 -> ラベル
