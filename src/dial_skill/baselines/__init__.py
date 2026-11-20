"""Fair baseline implementations: B0 Direct Chat, B1 Single-Model Confidence Gate, B2 ProACT-Reimpl."""

from dial_skill.baselines.direct_chat import DirectChatBaseline
from dial_skill.baselines.proact import ProACTBaseline
from dial_skill.baselines.single_gate import SingleGateBaseline

__all__ = [
    "DirectChatBaseline",
    "ProACTBaseline",
    "SingleGateBaseline",
]
