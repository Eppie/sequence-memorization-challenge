"""Reproduction of Linsefors & Bushnaq, "Challenge: Hand coding weights for
efficient sequence memorisation" (LessWrong, 2026-07-23)."""

from .capacity import CONDITIONS, SweepGrid, can_store, find_max_facts, fit_scaling
from .data import generate_facts
from .handcoded import HandCodedParams, hand_coded_weights
from .model import ModelShape, accuracy, forward, random_init, train

__all__ = [
    "CONDITIONS",
    "HandCodedParams",
    "ModelShape",
    "SweepGrid",
    "accuracy",
    "can_store",
    "find_max_facts",
    "fit_scaling",
    "forward",
    "generate_facts",
    "hand_coded_weights",
    "random_init",
    "train",
]
