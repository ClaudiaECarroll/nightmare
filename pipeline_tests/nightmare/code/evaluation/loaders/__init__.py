"""Prediction loaders indexed by pipeline name."""

from . import booknlp

REGISTRY = {"booknlp": booknlp}
