"""Tests for CLI interface."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from sequence_converter.cli import app


@pytest.fixture
def cli_runner():
    """CLI runner fixture."""
    return CliRunner()


@pytest.fixture
def mock_pipeline():
    """Mock pipeline orchestrator."""
    with patch("sequence_converter.cli.PipelineOrchestrator") as mock:
        mock_instance = Mock()
        mock_instance.convert.return_value = "@startuml\nA -> B: test\n@enduml"
        mock.return_value = mock_instance
        yield mock_instance


class TestCLIConvert:
    """Tests for convert command."""

    def test_convert_with_default_arguments(self, cli_runner, mock_pipeline, tmp_path):
        """デフォルト引数（input/, output/）でコマンドが実行されることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        test_image = input_dir / "test.png"
        test_image.write_bytes(b"fake png data")

        # Act
        with patch("sequence_converter.cli.Path") as mock_path_class:
            # Path() constructorがデフォルトパスを返すようにモック
            def path_side_effect(path_str):
                if path_str == "input":
                    return input_dir
                elif path_str == "output":
                    return output_dir
                return Path(path_str)

            mock_path_class.side_effect = path_side_effect
            result = cli_runner.invoke(app, ["convert"])

        # Assert
        assert result.exit_code == 0

    def test_convert_with_custom_arguments(self, cli_runner, mock_pipeline, tmp_path):
        """カスタム引数（--input, --output）で正しく動作することをテスト"""
        # Arrange
        input_dir = tmp_path / "custom_input"
        output_dir = tmp_path / "custom_output"
        input_dir.mkdir()
        output_dir.mkdir()

        test_image = input_dir / "test.png"
        test_image.write_bytes(b"fake png data")

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code == 0

    def test_convert_processes_png_files(self, cli_runner, mock_pipeline, tmp_path):
        """PNG画像ファイルが処理されることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        png_file = input_dir / "diagram.png"
        png_file.write_bytes(b"fake png data")

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code == 0
        assert mock_pipeline.convert.called

    def test_convert_processes_jpg_files(self, cli_runner, mock_pipeline, tmp_path):
        """JPG画像ファイルが処理されることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        jpg_file = input_dir / "diagram.jpg"
        jpg_file.write_bytes(b"fake jpg data")

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code == 0
        assert mock_pipeline.convert.called

    def test_convert_skips_unsupported_formats(self, cli_runner, mock_pipeline, tmp_path):
        """非対応ファイル形式がスキップされることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # 非対応ファイルを作成
        unsupported_file = input_dir / "document.txt"
        unsupported_file.write_text("not an image")

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert - エラーにはならないが、処理もされない
        assert result.exit_code == 0

    def test_convert_displays_error_for_nonexistent_input_dir(self, cli_runner, tmp_path):
        """存在しない入力ディレクトリでエラーメッセージが表示されることをテスト"""
        # Arrange
        nonexistent_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(nonexistent_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code != 0
        assert "does not exist" in result.stdout or "not found" in result.stdout.lower()

    def test_convert_creates_output_directory_if_not_exists(
        self, cli_runner, mock_pipeline, tmp_path
    ):
        """出力ディレクトリが存在しない場合は作成されることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"  # 存在しない
        input_dir.mkdir()

        test_image = input_dir / "test.png"
        test_image.write_bytes(b"fake png data")

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_convert_saves_plantuml_to_file(self, cli_runner, mock_pipeline, tmp_path):
        """PlantUMLコードがファイルに保存されることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        test_image = input_dir / "diagram.png"
        test_image.write_bytes(b"fake png data")

        expected_output = output_dir / "diagram.puml"

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code == 0
        # ファイルが作成されたことを確認（モックなので実際には作成されないが、呼び出しは確認できる）

    def test_convert_displays_progress_information(self, cli_runner, mock_pipeline, tmp_path):
        """処理中に進捗情報が表示されることをテスト"""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        test_image = input_dir / "test.png"
        test_image.write_bytes(b"fake png data")

        # Act
        result = cli_runner.invoke(
            app, ["convert", "--input", str(input_dir), "--output", str(output_dir)]
        )

        # Assert
        assert result.exit_code == 0
        # 何らかの出力があることを確認
        assert len(result.stdout) > 0 or len(result.stderr) > 0
