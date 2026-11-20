"""When2Speak benchmark loader & conversation-level stratified sampler (Invariant #8)."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from dial_skill.schemas import ChatMessage, When2SpeakTurn
from dial_skill.utils.drive import get_drive_root

logger = logging.getLogger(__name__)

SPLIT_TURN_BUDGETS: dict[str, tuple[Literal["validation", "test"], int]] = {
    "val_calib": ("validation", 200),
    "pilot": ("test", 400),
    "main": ("test", 2000),
    "full": ("test", -1),
}


def _parse_speak_label(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw) > 0)
    text = str(raw).strip().lower()
    if text in (">", "speak", "intervene", "true", "1", "yes", "<speak>"):
        return True
    if text in ("<", "silent", "silence", "wait", "false", "0", "no", "<silent>"):
        return False
    return False


def _parse_messages_from_row(row: dict[str, Any], conv_id: str) -> list[ChatMessage]:
    raw_msgs = (
        row.get("messages")
        or row.get("history")
        or row.get("dialogue")
        or row.get("context")
        or []
    )
    channel = str(row.get("channel") or row.get("channel_name") or "#group-chat")
    parsed: list[ChatMessage] = []
    if isinstance(raw_msgs, str):
        for i, line in enumerate(raw_msgs.splitlines()):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                spk, txt = line.split(":", 1)
            else:
                spk, txt = f"user_{i % 3}", line
            parsed.append(
                ChatMessage(
                    msg_id=f"{conv_id}_m{i}",
                    speaker=spk.strip(),
                    text=txt.strip(),
                    channel=channel,
                )
            )
    elif isinstance(raw_msgs, list):
        for i, item in enumerate(raw_msgs):
            if isinstance(item, dict):
                spk = str(item.get("speaker") or item.get("role") or item.get("user") or f"user_{i % 3}")
                txt = str(item.get("text") or item.get("content") or item.get("utterance") or "")
                mid = str(item.get("msg_id") or f"{conv_id}_m{i}")
                ts = str(item.get("timestamp") or "")
                parsed.append(
                    ChatMessage(msg_id=mid, speaker=spk, text=txt, timestamp=ts, channel=channel)
                )
            else:
                parsed.append(
                    ChatMessage(
                        msg_id=f"{conv_id}_m{i}",
                        speaker=f"user_{i % 3}",
                        text=str(item),
                        channel=channel,
                    )
                )

    # If the dataset stores the latest turn separately in `current_turn` or `utterance`
    latest = row.get("current_turn") or row.get("target_message")
    if isinstance(latest, dict):
        parsed.append(
            ChatMessage(
                msg_id=str(latest.get("msg_id") or f"{conv_id}_m{len(parsed)}"),
                speaker=str(latest.get("speaker") or latest.get("role") or "speaker"),
                text=str(latest.get("text") or latest.get("content") or ""),
                timestamp=str(latest.get("timestamp") or ""),
                channel=channel,
            )
        )
    elif isinstance(latest, str) and latest.strip():
        parsed.append(
            ChatMessage(
                msg_id=f"{conv_id}_m{len(parsed)}",
                speaker=str(row.get("speaker") or "speaker"),
                text=latest.strip(),
                channel=channel,
            )
        )

    return parsed


def normalize_when2speak_rows(
    rows: list[dict[str, Any]],
    split: Literal["train", "validation", "test"],
) -> list[When2SpeakTurn]:
    """Convert raw HuggingFace or JSONL rows into canonical When2SpeakTurn objects."""
    turns: list[When2SpeakTurn] = []
    for idx, row in enumerate(rows):
        conv_id = str(row.get("conversation_id") or row.get("conv_id") or row.get("dialogue_id") or f"{split}_c{idx}")
        turn_idx = int(row.get("turn_index") if row.get("turn_index") is not None else row.get("turn_id", idx))
        sample_id = str(row.get("sample_id") or f"w2s_{split}_{conv_id}_t{turn_idx}_{idx}")
        msgs = _parse_messages_from_row(row, conv_id)
        if not msgs:
            continue

        raw_label = (
            row.get("label_speak")
            if "label_speak" in row
            else row.get("should_speak", row.get("label", row.get("token", row.get("action", 0))))
        )
        label_speak = _parse_speak_label(raw_label)
        itype = str(row.get("intervention_type") or row.get("category") or row.get("type") or "general")
        ref_resp = row.get("reference_response") or row.get("response") or row.get("assistant_response")

        turns.append(
            When2SpeakTurn(
                sample_id=sample_id,
                conversation_id=conv_id,
                turn_index=turn_idx,
                split=split,
                messages=msgs,
                label_speak=label_speak,
                intervention_type=itype,
                reference_response=str(ref_resp) if ref_resp else None,
                metadata={"source_index": idx},
            )
        )
    return turns


def stratified_sample_conversations(
    turns: list[When2SpeakTurn],
    target_turns: int,
    seed: int = 42,
) -> list[When2SpeakTurn]:
    """Conversation-level stratified sampling preserving SPEAK rate and intervention-type coverage."""
    if target_turns < 0 or len(turns) <= target_turns:
        return list(turns)

    by_conv: dict[str, list[When2SpeakTurn]] = defaultdict(list)
    for t in turns:
        by_conv[t.conversation_id].append(t)
    for conv_turns in by_conv.values():
        conv_turns.sort(key=lambda x: x.turn_index)

    # Group conversations by primary intervention type (or 'silent_only')
    strata: dict[str, list[str]] = defaultdict(list)
    for cid, cturns in by_conv.items():
        pos_types = [t.intervention_type for t in cturns if t.label_speak]
        stratum = pos_types[0] if pos_types else "__silent_only__"
        strata[stratum].append(cid)

    rng = random.Random(seed)
    for cid_list in strata.values():
        cid_list.sort()
        rng.shuffle(cid_list)

    # Round-robin across strata proportional to stratum size so ~13% SPEAK & all 5 types are preserved
    selected_cids: list[str] = []
    selected_turns: list[When2SpeakTurn] = []
    stratum_keys = sorted(strata.keys())
    stratum_ptrs = {k: 0 for k in stratum_keys}

    # Compute desired proportional turn quotas per stratum
    total_available = sum(len(by_conv[c]) for c in by_conv)
    quotas = {
        k: max(1, int(round(target_turns * sum(len(by_conv[c]) for c in strata[k]) / max(1, total_available))))
        for k in stratum_keys
    }
    stratum_counts = {k: 0 for k in stratum_keys}

    # First pass: satisfy per-stratum proportional quotas
    progressed = True
    while len(selected_turns) < target_turns and progressed:
        progressed = False
        for k in stratum_keys:
            if len(selected_turns) >= target_turns:
                break
            if stratum_counts[k] >= quotas[k]:
                continue
            ptr = stratum_ptrs[k]
            if ptr < len(strata[k]):
                cid = strata[k][ptr]
                stratum_ptrs[k] += 1
                selected_cids.append(cid)
                cturns = by_conv[cid]
                selected_turns.extend(cturns)
                stratum_counts[k] += len(cturns)
                progressed = True

    # Second pass: fill any residual turns up to target_turns
    while len(selected_turns) < target_turns:
        added_any = False
        for k in stratum_keys:
            ptr = stratum_ptrs[k]
            if ptr < len(strata[k]):
                cid = strata[k][ptr]
                stratum_ptrs[k] += 1
                selected_cids.append(cid)
                selected_turns.extend(by_conv[cid])
                added_any = True
                if len(selected_turns) >= target_turns:
                    break
        if not added_any:
            break

    # Trim cleanly to target_turns if the last conversation overshot slightly
    return selected_turns[:target_turns]


def assert_no_calibration_leakage(
    calib_turns: list[When2SpeakTurn],
    eval_turns: list[When2SpeakTurn],
    max_calib_turns: int = 200,
) -> None:
    """Enforce Invariant #8: calibration MUST only use validation split (<=200 turns) with zero overlap."""
    if len(calib_turns) > max_calib_turns:
        raise AssertionError(
            f"Invariant #8 violated: calibration split has {len(calib_turns)} > {max_calib_turns} turns."
        )
    bad_splits = {t.split for t in calib_turns if t.split != "validation"}
    if bad_splits:
        raise AssertionError(
            f"Invariant #8 violated: calibration turns contain non-validation splits: {bad_splits}"
        )
    calib_convs = {t.conversation_id for t in calib_turns}
    eval_convs = {t.conversation_id for t in eval_turns}
    overlap = calib_convs & eval_convs
    if overlap:
        raise AssertionError(
            f"Invariant #8 violated: {len(overlap)} conversations overlap between calibration and eval."
        )


def _read_local_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_when2speak_split(
    preset: Literal["val_calib", "pilot", "main", "full"] = "pilot",
    *,
    dataset_id: str = "duke-trust-lab/When2Speak",
    local_dir: str | Path | None = None,
    seed: int = 42,
    subset_name: str | None = None,
) -> list[When2SpeakTurn]:
    """Load When2Speak split from local JSONL cache or Hugging Face and apply conversation-level sampling."""
    if preset not in SPLIT_TURN_BUDGETS:
        raise ValueError(f"Unknown When2Speak preset '{preset}'. Choose from {list(SPLIT_TURN_BUDGETS)}")
    hf_split, target_turns = SPLIT_TURN_BUDGETS[preset]

    cache_dir = Path(local_dir) if local_dir else get_drive_root() / "datasets" / "when2speak"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_preset_path = cache_dir / f"when2speak_{preset}_seed{seed}.jsonl"

    if cached_preset_path.is_file():
        return [When2SpeakTurn.model_validate(r) for r in _read_local_jsonl(cached_preset_path)]

    # Check raw local split JSONL before hitting Hugging Face
    raw_local_candidates = [
        cache_dir / f"{hf_split}.jsonl",
        cache_dir / f"{hf_split}_token.jsonl",
        cache_dir / f"when2speak_{hf_split}.jsonl",
    ]
    raw_rows: list[dict[str, Any]] | None = None
    for cand in raw_local_candidates:
        if cand.is_file():
            raw_rows = _read_local_jsonl(cand)
            break

    if raw_rows is None:
        from datasets import load_dataset  # lazy import

        ds = load_dataset(dataset_id, name=subset_name, split=hf_split)
        raw_rows = [dict(row) for row in ds]

    all_turns = normalize_when2speak_rows(raw_rows, split=hf_split)
    sampled = stratified_sample_conversations(all_turns, target_turns=target_turns, seed=seed)

    # Save deterministic preset snapshot to Drive cache
    with cached_preset_path.open("w", encoding="utf-8") as f:
        for turn in sampled:
            f.write(turn.model_dump_json() + "\n")

    return sampled
