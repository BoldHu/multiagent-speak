"""When2Speak benchmark loader & conversation-level stratified sampler (Invariant #8)."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
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

DEFAULT_WHEN2SPEAK_SUBSET: Literal["dialogue", "token"] = "dialogue"

_SPEAKER_PREFIX_RE = re.compile(r"^([A-Za-z0-9_\-\[\]]{1,32})\s*:\s*(.+)$")
_EXPLICIT_LABEL_KEYS = ("label_speak", "should_speak", "label", "token", "action")


def _parse_speak_label(raw: Any) -> bool:
    """Parse binary SPEAK/SILENT label matching duke-trust-lab/When2Speak (`<` = SPEAK, `>` = SILENT)."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw) > 0)
    text = str(raw).strip()
    lower = text.lower()
    # Official When2Speak token convention: `<` = SPEAK (~13%), `>` = SILENT (~87%)
    if lower in ("<", "speak", "intervene", "true", "1", "yes", "<speak>"):
        return True
    if lower in (">", "", "silent", "silence", "wait", "false", "0", "no", "<silent>"):
        return False
    # In When2Speak `dialogue` config, any non-`>` natural-language assistant reply is SPEAK
    return True


def _split_speaker_text(raw_text: str, default_speaker: str) -> tuple[str, str]:
    match = _SPEAKER_PREFIX_RE.match(raw_text.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return default_speaker, raw_text.strip()


def _parse_messages_and_target(
    row: dict[str, Any],
    fallback_conv_id: str,
) -> tuple[list[ChatMessage], bool, str | None]:
    """Parse dialogue turns and target label from HF OpenAI-chat format or explicit-label JSONL."""
    raw_msgs = (
        row.get("messages")
        or row.get("history")
        or row.get("dialogue")
        or row.get("context")
        or []
    )
    channel = str(row.get("channel") or row.get("channel_name") or "#group-chat")
    has_explicit_label = any(k in row and row[k] is not None for k in _EXPLICIT_LABEL_KEYS)

    # Detect official duke-trust-lab/When2Speak OpenAI chat format:
    # messages = [..., {"role": "user", "content": "Speaker_k: ..."}, {"role": "assistant", "content": "<" | ">" | "<reply>"}]
    target_assistant_content: str | None = None
    history_items: Any = raw_msgs
    if (
        not has_explicit_label
        and isinstance(raw_msgs, list)
        and raw_msgs
        and isinstance(raw_msgs[-1], dict)
        and str(raw_msgs[-1].get("role", "")).lower() == "assistant"
    ):
        target_assistant_content = str(raw_msgs[-1].get("content") or raw_msgs[-1].get("text") or "")
        history_items = raw_msgs[:-1]

    parsed: list[ChatMessage] = []
    if isinstance(history_items, str):
        for i, line in enumerate(history_items.splitlines()):
            line = line.strip()
            if not line:
                continue
            spk, txt = _split_speaker_text(line, f"Speaker_{i % 3}")
            parsed.append(
                ChatMessage(msg_id=f"{fallback_conv_id}_m{len(parsed)}", speaker=spk, text=txt, channel=channel)
            )
    elif isinstance(history_items, list):
        for i, item in enumerate(history_items):
            if isinstance(item, dict):
                role = str(item.get("role", "")).lower()
                if role == "system":
                    continue
                raw_txt = str(item.get("text") or item.get("content") or item.get("utterance") or "").strip()
                if not raw_txt:
                    continue
                if role == "assistant":
                    # Prior silent markers `>` or token `<` in multi-turn SFT history are not spoken utterances
                    if raw_txt in (">", "<"):
                        continue
                    parsed.append(
                        ChatMessage(
                            msg_id=str(item.get("msg_id") or f"{fallback_conv_id}_m{len(parsed)}"),
                            speaker="[AGENT]",
                            text=raw_txt,
                            timestamp=str(item.get("timestamp") or ""),
                            channel=channel,
                        )
                    )
                    continue

                explicit_spk = item.get("speaker") or item.get("user")
                if explicit_spk:
                    parsed.append(
                        ChatMessage(
                            msg_id=str(item.get("msg_id") or f"{fallback_conv_id}_m{len(parsed)}"),
                            speaker=str(explicit_spk),
                            text=raw_txt,
                            timestamp=str(item.get("timestamp") or ""),
                            channel=channel,
                        )
                    )
                else:
                    # Handle either single-turn "Speaker_0: ..." or multi-line "Speaker_0: ...\nSpeaker_1: ..."
                    lines = [ln.strip() for ln in raw_txt.splitlines() if ln.strip()]
                    if len(lines) > 1 and all(_SPEAKER_PREFIX_RE.match(ln) for ln in lines):
                        for ln in lines:
                            spk, txt = _split_speaker_text(ln, f"Speaker_{len(parsed) % 3}")
                            parsed.append(
                                ChatMessage(
                                    msg_id=f"{fallback_conv_id}_m{len(parsed)}",
                                    speaker=spk,
                                    text=txt,
                                    timestamp=str(item.get("timestamp") or ""),
                                    channel=channel,
                                )
                            )
                    else:
                        spk, txt = _split_speaker_text(raw_txt, f"Speaker_{len(parsed) % 3}")
                        parsed.append(
                            ChatMessage(
                                msg_id=str(item.get("msg_id") or f"{fallback_conv_id}_m{len(parsed)}"),
                                speaker=spk,
                                text=txt,
                                timestamp=str(item.get("timestamp") or ""),
                                channel=channel,
                            )
                        )
            else:
                spk, txt = _split_speaker_text(str(item), f"Speaker_{len(parsed) % 3}")
                parsed.append(
                    ChatMessage(msg_id=f"{fallback_conv_id}_m{len(parsed)}", speaker=spk, text=txt, channel=channel)
                )

    # Support optional separate `current_turn` / `target_message` field
    latest = row.get("current_turn") or row.get("target_message")
    if isinstance(latest, dict):
        parsed.append(
            ChatMessage(
                msg_id=str(latest.get("msg_id") or f"{fallback_conv_id}_m{len(parsed)}"),
                speaker=str(latest.get("speaker") or latest.get("role") or "speaker"),
                text=str(latest.get("text") or latest.get("content") or ""),
                timestamp=str(latest.get("timestamp") or ""),
                channel=channel,
            )
        )
    elif isinstance(latest, str) and latest.strip():
        spk, txt = _split_speaker_text(latest, str(row.get("speaker") or "speaker"))
        parsed.append(
            ChatMessage(msg_id=f"{fallback_conv_id}_m{len(parsed)}", speaker=spk, text=txt, channel=channel)
        )

    # Determine binary label and optional reference response
    ref_resp = row.get("reference_response") or row.get("response") or row.get("assistant_response")
    if target_assistant_content is not None:
        stripped = target_assistant_content.strip()
        label_speak = _parse_speak_label(stripped)
        if label_speak and stripped not in ("<", "speak", "<speak>") and not ref_resp:
            ref_resp = stripped
    else:
        raw_label = next((row[k] for k in _EXPLICIT_LABEL_KEYS if k in row and row[k] is not None), 0)
        label_speak = _parse_speak_label(raw_label)

    return parsed, label_speak, (str(ref_resp) if ref_resp else None)


def _derive_conversation_id(
    row: dict[str, Any],
    msgs: list[ChatMessage],
    split: str,
    idx: int,
) -> str:
    explicit = row.get("conversation_id") or row.get("conv_id") or row.get("dialogue_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    # Deterministic conversation grouping by opening turn for conversation-level stratified sampling
    if msgs:
        anchor = f"{msgs[0].speaker}:{msgs[0].text[:240]}".encode("utf-8")
        digest = hashlib.sha256(anchor).hexdigest()[:12]
        return f"{split}_conv_{digest}"
    return f"{split}_c{idx}"


def normalize_when2speak_rows(
    rows: list[dict[str, Any]],
    split: Literal["train", "validation", "test"],
) -> list[When2SpeakTurn]:
    """Convert raw HuggingFace or JSONL rows into canonical When2SpeakTurn objects."""
    turns: list[When2SpeakTurn] = []
    for idx, row in enumerate(rows):
        tmp_cid = str(row.get("conversation_id") or row.get("conv_id") or row.get("dialogue_id") or f"{split}_c{idx}")
        msgs, label_speak, ref_resp = _parse_messages_and_target(row, tmp_cid)
        if not msgs:
            continue

        conv_id = _derive_conversation_id(row, msgs, split, idx)
        # Re-normalize msg_ids cleanly if conv_id was inferred from opening turn
        if conv_id != tmp_cid:
            msgs = [
                m.model_copy(update={"msg_id": f"{conv_id}_m{j}"})
                if m.msg_id.startswith(f"{tmp_cid}_m")
                else m
                for j, m in enumerate(msgs)
            ]

        turn_idx = int(
            row["turn_index"]
            if row.get("turn_index") is not None
            else (row["turn_id"] if row.get("turn_id") is not None else max(0, len(msgs) - 1))
        )
        sample_id = str(row.get("sample_id") or f"w2s_{split}_{conv_id}_t{turn_idx}_{idx}")
        itype = str(row.get("intervention_type") or row.get("category") or row.get("type") or "general")

        turns.append(
            When2SpeakTurn(
                sample_id=sample_id,
                conversation_id=conv_id,
                turn_index=turn_idx,
                split=split,
                messages=msgs,
                label_speak=label_speak,
                intervention_type=itype,
                reference_response=ref_resp,
                metadata={"source_index": idx},
            )
        )

    if turns:
        pos = sum(1 for t in turns if t.label_speak)
        logger.info(
            "Normalized When2Speak [%s]: %d turns across %d conversations (%d SPEAK = %.1f%%)",
            split,
            len(turns),
            len({t.conversation_id for t in turns}),
            pos,
            100.0 * pos / len(turns),
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

    selected_turns: list[When2SpeakTurn] = []
    stratum_keys = sorted(strata.keys())
    stratum_ptrs = {k: 0 for k in stratum_keys}

    total_available = sum(len(by_conv[c]) for c in by_conv)
    quotas = {
        k: max(1, int(round(target_turns * sum(len(by_conv[c]) for c in strata[k]) / max(1, total_available))))
        for k in stratum_keys
    }
    stratum_counts = {k: 0 for k in stratum_keys}

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
                cturns = by_conv[cid]
                selected_turns.extend(cturns)
                stratum_counts[k] += len(cturns)
                progressed = True

    while len(selected_turns) < target_turns:
        added_any = False
        for k in stratum_keys:
            ptr = stratum_ptrs[k]
            if ptr < len(strata[k]):
                cid = strata[k][ptr]
                stratum_ptrs[k] += 1
                selected_turns.extend(by_conv[cid])
                added_any = True
                if len(selected_turns) >= target_turns:
                    break
        if not added_any:
            break

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
    subset_name: Literal["dialogue", "token"] = DEFAULT_WHEN2SPEAK_SUBSET,
) -> list[When2SpeakTurn]:
    """Load When2Speak split from local JSONL cache or Hugging Face and apply conversation-level sampling.

    Defaults to `subset_name="dialogue"` on `duke-trust-lab/When2Speak`, which supplies both exact
    binary timing labels (`>` = SILENT, non-`>` text = SPEAK) and natural-language reference replies.
    """
    if preset not in SPLIT_TURN_BUDGETS:
        raise ValueError(f"Unknown When2Speak preset '{preset}'. Choose from {list(SPLIT_TURN_BUDGETS)}")
    hf_split, target_turns = SPLIT_TURN_BUDGETS[preset]

    cache_dir = Path(local_dir) if local_dir else get_drive_root() / "datasets" / "when2speak"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_preset_path = cache_dir / f"when2speak_{subset_name}_{preset}_seed{seed}.jsonl"

    if cached_preset_path.is_file():
        return [When2SpeakTurn.model_validate(r) for r in _read_local_jsonl(cached_preset_path)]

    # Check raw local split JSONL before hitting Hugging Face
    short_split = "val" if hf_split == "validation" else hf_split
    raw_local_candidates = [
        cache_dir / f"finetune_{short_split}_{subset_name}.jsonl",
        cache_dir / f"{hf_split}_{subset_name}.jsonl",
        cache_dir / f"{hf_split}.jsonl",
        cache_dir / f"when2speak_{hf_split}.jsonl",
    ]
    raw_rows: list[dict[str, Any]] | None = None
    for cand in raw_local_candidates:
        if cand.is_file():
            raw_rows = _read_local_jsonl(cand)
            break

    if raw_rows is None:
        from datasets import load_dataset  # lazy import

        try:
            ds = load_dataset(dataset_id, name=subset_name, split=hf_split)
        except ValueError:
            # Handle datasets where validation split is named "val"
            alt_split = "val" if hf_split == "validation" else hf_split
            ds = load_dataset(dataset_id, name=subset_name, split=alt_split)
        raw_rows = [dict(row) for row in ds]

    all_turns = normalize_when2speak_rows(raw_rows, split=hf_split)
    sampled = stratified_sample_conversations(all_turns, target_turns=target_turns, seed=seed)

    # Save deterministic preset snapshot to Drive cache
    with cached_preset_path.open("w", encoding="utf-8") as f:
        for turn in sampled:
            f.write(turn.model_dump_json() + "\n")

    return sampled
