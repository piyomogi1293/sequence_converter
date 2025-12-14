"""PlantUML generator component."""

import logging
from pathlib import Path
from typing import Optional

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

logger = logging.getLogger(__name__)


class PlantUMLGenerator:
    """PlantUML生成コンポーネント"""

    def generate(self, elements: SequenceDiagramElements) -> str:
        """
        シーケンス図要素からPlantUMLコードを生成

        Args:
            elements: 検出された全要素

        Returns:
            str: PlantUMLコード（@startuml ... @enduml）
        """
        lines = ["@startuml"]

        # 1. オブジェクト宣言（participant）- 左から右へソート済み
        lines.extend(self._generate_object_declarations(elements.objects))

        # 2. 全イベント要素をY座標でソート（メッセージ、自己呼び出し、ノート）
        # 3. フラグメントはY座標範囲でグルーピング
        # 4. ソート済みイベントを順にPlantUML構文に変換
        # 5. アクティベーションバー範囲内のメッセージに++/--構文を付与
        lines.extend(
            self._generate_events(
                elements.messages,
                elements.self_calls,
                elements.notes,
                elements.fragments,
                elements.activations,
                elements.message_labels,
                elements.objects,
            )
        )

        lines.append("@enduml")

        plantuml_code = "\n".join(lines)
        logger.info(f"Generated PlantUML code ({len(lines)} lines)")
        return plantuml_code

    def save_to_file(self, plantuml_code: str, output_path: Path) -> None:
        """
        PlantUMLコードをファイルに保存

        Args:
            plantuml_code: PlantUMLコード
            output_path: 出力ファイルパス（.puml）
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(plantuml_code, encoding="utf-8")
        logger.info(f"Saved PlantUML code to {output_path}")

    def _generate_object_declarations(self, objects: list[ObjectHeader]) -> list[str]:
        """
        オブジェクト宣言を生成

        Args:
            objects: オブジェクトヘッダーリスト（左から右へソート済み）

        Returns:
            list[str]: PlantUMLオブジェクト宣言行のリスト
        """
        declarations = []
        for obj in objects:
            declarations.append(f'participant "{obj.name}"')
        return declarations

    def _generate_events(
        self,
        messages: list[MessageArrow],
        self_calls: list[SelfCall],
        notes: list[NoteAnnotation],
        fragments: list[Fragment],
        activations: list[ActivationBar],
        message_labels: dict[int, OCRResult],
        objects: list[ObjectHeader],
    ) -> list[str]:
        """
        イベント（メッセージ、自己呼び出し、ノート、フラグメント）を生成

        Args:
            messages: メッセージ矢印リスト
            self_calls: 自己呼び出しリスト
            notes: ノート注釈リスト
            fragments: フラグメントリスト
            activations: アクティベーションバーリスト
            message_labels: メッセージY座標 -> OCRラベルのマッピング
            objects: オブジェクトヘッダーリスト

        Returns:
            list[str]: PlantUMLイベント行のリスト
        """
        # オブジェクト名をX座標でマッピング
        lifeline_to_name = {obj.lifeline_x: obj.name for obj in objects}

        # 全イベントをY座標でソート可能な形式に変換
        events = []

        # メッセージ
        for msg in messages:
            events.append(("message", msg.y, msg))

        # 自己呼び出し
        for sc in self_calls:
            events.append(("self_call", sc.y, sc))

        # ノート
        for note in notes:
            # bounding_boxの2番目の要素がY座標
            events.append(("note", note.bounding_box[1], note))

        # フラグメント
        for frag in fragments:
            # bounding_boxの2番目の要素がY座標
            events.append(("fragment_start", frag.bounding_box[1], frag))
            # フラグメント終了をY座標 + 高さの位置に配置
            frag_end_y = frag.bounding_box[1] + frag.bounding_box[3]
            events.append(("fragment_end", frag_end_y, frag))

        # Y座標でソート
        events.sort(key=lambda e: e[1])

        # イベントをPlantUML構文に変換
        lines = []
        for event_type, y, data in events:
            if event_type == "message":
                line = self._generate_message(data, message_labels, lifeline_to_name, activations)
                if line:
                    lines.append(line)

            elif event_type == "self_call":
                line = self._generate_self_call(data, lifeline_to_name)
                if line:
                    lines.append(line)

            elif event_type == "note":
                line = self._generate_note(data, lifeline_to_name)
                if line:
                    lines.append(line)

            elif event_type == "fragment_start":
                line = self._generate_fragment_start(data)
                if line:
                    lines.append(line)

            elif event_type == "fragment_end":
                lines.append("end")

        return lines

    def _generate_message(
        self,
        msg: MessageArrow,
        message_labels: dict[int, OCRResult],
        lifeline_to_name: dict[int, str],
        activations: list[ActivationBar],
    ) -> Optional[str]:
        """
        メッセージ行を生成

        Args:
            msg: メッセージ矢印
            message_labels: メッセージY座標 -> OCRラベルのマッピング
            lifeline_to_name: ライフラインX座標 -> オブジェクト名のマッピング
            activations: アクティベーションバーリスト

        Returns:
            Optional[str]: PlantUMLメッセージ行、またはNone（ライフラインマッチングに失敗した場合）
        """
        # 送信元と宛先のオブジェクト名を取得
        source_name = lifeline_to_name.get(msg.source_lifeline)
        dest_name = lifeline_to_name.get(msg.dest_lifeline)

        if not source_name or not dest_name:
            logger.warning(
                f"Message at y={msg.y} has unmatched lifelines: "
                f"source={msg.source_lifeline}, dest={msg.dest_lifeline}"
            )
            return None

        # メッセージラベルを取得
        label_result = message_labels.get(msg.y)
        label = label_result.text if label_result else ""

        # 色情報を付与
        if label_result and label_result.color:
            label = f"[#{label_result.color}]{label}"

        # アクティベーション構文を確認
        activation_suffix = self._get_activation_suffix(msg, activations, lifeline_to_name)

        # 矢印の方向に応じて構文を生成
        arrow = "->" if msg.direction == ArrowDirection.LEFT_TO_RIGHT else "<-"

        return f'"{source_name}" {arrow} "{dest_name}" : {label}{activation_suffix}'

    def _generate_self_call(self, sc: SelfCall, lifeline_to_name: dict[int, str]) -> Optional[str]:
        """
        自己呼び出し行を生成

        Args:
            sc: 自己呼び出し
            lifeline_to_name: ライフラインX座標 -> オブジェクト名のマッピング

        Returns:
            Optional[str]: PlantUML自己呼び出し行、またはNone（ライフラインマッチングに失敗した場合）
        """
        obj_name = lifeline_to_name.get(sc.lifeline_x)

        if not obj_name:
            logger.warning(f"Self-call at y={sc.y} has unmatched lifeline: {sc.lifeline_x}")
            return None

        label = sc.label or ""

        return f'"{obj_name}" -> "{obj_name}" : {label}'

    def _generate_note(
        self, note: NoteAnnotation, lifeline_to_name: dict[int, str]
    ) -> Optional[str]:
        """
        ノート行を生成

        Args:
            note: ノート注釈
            lifeline_to_name: ライフラインX座標 -> オブジェクト名のマッピング

        Returns:
            Optional[str]: PlantUMLノート行
        """
        # related_lifelineがある場合はその名前を使用
        if note.related_lifeline:
            obj_name = lifeline_to_name.get(note.related_lifeline)
            if obj_name:
                return f'note over "{obj_name}" : {note.text}'

        # related_lifelineがない場合は、一般的なノートとして出力
        return f"note left : {note.text}"

    def _generate_fragment_start(self, frag: Fragment) -> str:
        """
        フラグメント開始行を生成

        Args:
            frag: フラグメント

        Returns:
            str: PlantUMLフラグメント開始行
        """
        frag_type = frag.type.value  # Enumの値を取得

        if frag.title:
            return f"{frag_type} {frag.title}"
        else:
            return frag_type

    def _get_activation_suffix(
        self,
        msg: MessageArrow,
        activations: list[ActivationBar],
        lifeline_to_name: dict[int, str],
    ) -> str:
        """
        メッセージがアクティベーションバー範囲内にある場合、++/--構文を返す

        Args:
            msg: メッセージ矢印
            activations: アクティベーションバーリスト
            lifeline_to_name: ライフラインX座標 -> オブジェクト名のマッピング

        Returns:
            str: アクティベーション構文（" ++", " --"）または空文字列
        """
        # アクティベーションバーの範囲内かチェック
        for act in activations:
            if act.y_start <= msg.y <= act.y_end:
                # メッセージの宛先がアクティベーションバーのライフラインと一致するか確認
                if msg.dest_lifeline == act.lifeline_x:
                    # アクティベーション開始位置付近なら++
                    if abs(msg.y - act.y_start) < 20:
                        return " ++"
                    # アクティベーション終了位置付近なら--
                    elif abs(msg.y - act.y_end) < 20:
                        return " --"

        return ""
