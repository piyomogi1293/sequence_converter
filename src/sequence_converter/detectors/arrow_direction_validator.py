"""Arrow direction validation and testing framework."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sequence_converter.models import ArrowDirection

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthArrow:
    """Ground truth arrow from backup file."""

    source: str
    dest: str
    direction: ArrowDirection
    label: str
    line_number: int


class ArrowDirectionValidator:
    """Validates arrow direction detection against ground truth from .puml.backup files."""

    def __init__(self):
        """Initialize validator."""
        self.ground_truth_arrows: list[GroundTruthArrow] = []
        self.name_aliases: dict[str, list[str]] = {}

    def set_name_aliases(self, aliases: dict[str, list[str]]):
        """Set name aliases for matching.

        Args:
            aliases: Dictionary mapping canonical names to list of aliases
                    e.g., {"UE": ["U"], "P-CSCF": ["P"], ...}
        """
        self.name_aliases = aliases

    def _normalize_name(self, name: str) -> str:
        """Normalize name to canonical form for matching.

        Args:
            name: Original name

        Returns:
            Canonical name
        """
        # Check if this is already a canonical name
        if name in self.name_aliases:
            return name

        # Check if this is an alias
        for canonical, aliases in self.name_aliases.items():
            if name in aliases:
                return canonical

        # No match, return as-is
        return name

    def load_ground_truth(self, backup_file_path: Path) -> list[GroundTruthArrow]:
        """Load ground truth arrows from .puml.backup file.

        Args:
            backup_file_path: Path to .puml.backup file

        Returns:
            List of ground truth arrows
        """
        if not backup_file_path.exists():
            logger.error(f"Backup file not found: {backup_file_path}")
            return []

        arrows = []
        with open(backup_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                # Parse PlantUML arrow syntax: "Source -> Dest : Label" or "Source <- Dest : Label"
                # Left to right: ->
                match_lr = re.match(r'^"?([^"]+)"?\s+->\s+"?([^"]+)"?\s*:\s*(.*)$', line)
                if match_lr:
                    source = match_lr.group(1).strip()
                    dest = match_lr.group(2).strip()
                    label = match_lr.group(3).strip()
                    arrows.append(
                        GroundTruthArrow(
                            source=source,
                            dest=dest,
                            direction=ArrowDirection.LEFT_TO_RIGHT,
                            label=label,
                            line_number=line_num,
                        )
                    )
                    continue

                # Right to left: <-
                match_rl = re.match(r'^"?([^"]+)"?\s+<-\s+"?([^"]+)"?\s*:\s*(.*)$', line)
                if match_rl:
                    # Note: In PlantUML, "A <- B" means B -> A
                    # So the first entity is the destination, second is source
                    dest = match_rl.group(1).strip()
                    source = match_rl.group(2).strip()
                    label = match_rl.group(3).strip()
                    arrows.append(
                        GroundTruthArrow(
                            source=source,
                            dest=dest,
                            direction=ArrowDirection.RIGHT_TO_LEFT,
                            label=label,
                            line_number=line_num,
                        )
                    )
                    continue

        self.ground_truth_arrows = arrows
        logger.info(f"Loaded {len(arrows)} ground truth arrows from {backup_file_path}")
        return arrows

    def calculate_accuracy(self, detected_arrows: list[tuple[str, str, ArrowDirection]]) -> dict:
        """Calculate detection accuracy metrics.

        Args:
            detected_arrows: List of (source, dest, direction) tuples

        Returns:
            Dictionary with accuracy metrics
        """
        if not self.ground_truth_arrows:
            logger.warning("No ground truth loaded")
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        # Normalize detected arrow names and create lookup dict
        detected_dict = {}
        for src, dst, direction in detected_arrows:
            norm_src = self._normalize_name(src)
            norm_dst = self._normalize_name(dst)
            key = (norm_src, norm_dst)
            detected_dict[key] = direction

        correct = 0
        total = len(self.ground_truth_arrows)
        mismatches = []

        for gt_arrow in self.ground_truth_arrows:
            # Normalize ground truth names
            norm_src = self._normalize_name(gt_arrow.source)
            norm_dst = self._normalize_name(gt_arrow.dest)
            key = (norm_src, norm_dst)

            if key in detected_dict:
                detected_dir = detected_dict[key]
                if detected_dir == gt_arrow.direction:
                    correct += 1
                else:
                    mismatches.append(
                        {
                            "source": gt_arrow.source,
                            "dest": gt_arrow.dest,
                            "expected": gt_arrow.direction.value,
                            "detected": detected_dir.value,
                            "label": gt_arrow.label,
                            "line": gt_arrow.line_number,
                        }
                    )
            else:
                mismatches.append(
                    {
                        "source": gt_arrow.source,
                        "dest": gt_arrow.dest,
                        "expected": gt_arrow.direction.value,
                        "detected": "NOT_DETECTED",
                        "label": gt_arrow.label,
                        "line": gt_arrow.line_number,
                    }
                )

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mismatches": mismatches,
        }
