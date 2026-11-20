"""Versioned JSON SkillStore enforcing O(1) capacity caps (Invariants #3 & #4)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from dial_skill.config import find_repo_root
from dial_skill.schemas import ACTIVE_SKILLS, SkillCard, SkillName

MAX_TRIGGER_CONDITIONS = 5
MAX_VETO_CONDITIONS = 5
MAX_INVESTIGATION_STEPS = 4


class SkillStore:
    """In-memory + atomic JSON-persisted store for the 8 Deepnote-derived Dual-Boundary Skill Cards."""

    def __init__(self, skills: dict[str, SkillCard], version: int = 0) -> None:
        self.version = version
        self.skills: dict[str, SkillCard] = {}
        for name, card in skills.items():
            validated = SkillCard.model_validate(card)
            self.skills[name] = validated
        self.validate_o1_caps()

    @classmethod
    def from_json_file(cls, path: str | Path | None = None) -> "SkillStore":
        cfg_path = Path(path) if path else find_repo_root() / "configs" / "seed_skills_v0.json"
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        version = int(raw.get("version", 0))
        skills = {
            name: SkillCard.model_validate(card_data)
            for name, card_data in raw["skills"].items()
        }
        return cls(skills=skills, version=version)

    def validate_o1_caps(self) -> None:
        """Enforce hard O(1) prompt capacity caps on every Skill Card (Invariant #4)."""
        for name, card in self.skills.items():
            if len(card.trigger_conditions) > MAX_TRIGGER_CONDITIONS:
                raise ValueError(
                    f"Skill '{name}' exceeds max trigger_conditions ({len(card.trigger_conditions)} > {MAX_TRIGGER_CONDITIONS})"
                )
            if len(card.veto_conditions) > MAX_VETO_CONDITIONS:
                raise ValueError(
                    f"Skill '{name}' exceeds max veto_conditions ({len(card.veto_conditions)} > {MAX_VETO_CONDITIONS})"
                )
            if len(card.investigation_plan) > MAX_INVESTIGATION_STEPS:
                raise ValueError(
                    f"Skill '{name}' exceeds max investigation_plan ({len(card.investigation_plan)} > {MAX_INVESTIGATION_STEPS})"
                )
            if not (-0.35 <= card.margin_offset <= 0.35):
                raise ValueError(f"Skill '{name}' margin_offset out of bounds: {card.margin_offset}")
            if not (0.0 <= card.deescalation_bias <= 2.0):
                raise ValueError(
                    f"Skill '{name}' deescalation_bias out of bounds: {card.deescalation_bias}"
                )

    def get(self, name: SkillName | str) -> SkillCard | None:
        if name == "none":
            return None
        return self.skills.get(str(name))

    def require(self, name: SkillName | str) -> SkillCard:
        card = self.get(name)
        if card is None:
            raise KeyError(f"Unknown skill '{name}'. Available: {list(self.skills.keys())}")
        return card

    def set_skill(self, card: SkillCard, bump_version: bool = False) -> None:
        validated = SkillCard.model_validate(card)
        self.skills[validated.name] = validated
        self.validate_o1_caps()
        if bump_version:
            self.version += 1

    def clone(self) -> "SkillStore":
        """Create a deep copy for safe counterfactual replay or rollback."""
        return SkillStore(
            skills={k: copy.deepcopy(v) for k, v in self.skills.items()},
            version=self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "skills": {
                k: self.skills[k].model_dump(mode="json")
                for k in sorted(self.skills.keys(), key=lambda x: ACTIVE_SKILLS.index(x) if x in ACTIVE_SKILLS else 99)
            },
        }

    def save(self, path: str | Path) -> Path:
        """Atomically persist the SkillStore snapshot to JSON."""
        self.validate_o1_caps()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
        return out_path

    def total_rule_count(self) -> int:
        return sum(
            len(c.trigger_conditions) + len(c.veto_conditions) + len(c.investigation_plan)
            for c in self.skills.values()
        )

    def estimate_serialized_tokens(self) -> int:
        """Approximate token count of all active skill contracts (for O(1) prompt-growth plots)."""
        text = json.dumps(self.to_dict(), ensure_ascii=False)
        return max(1, len(text) // 4)
