"""Compatibility exports for :mod:`molgap.models.egnn`."""

from .models.egnn import (
    scatter_mean,
    _EGNNLayer,
    EGNNWrapper,
)

__all__ = ['scatter_mean', '_EGNNLayer', 'EGNNWrapper']
