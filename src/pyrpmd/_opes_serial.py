"""Deprecated import location; OPES is implemented in pyrpmd.opes."""
from warnings import warn
from .opes import OPESBias, OPESExploreBias, normalized_weights

warn("pyrpmd._opes_serial is deprecated; use pyrpmd.opes",
     DeprecationWarning, stacklevel=2)
