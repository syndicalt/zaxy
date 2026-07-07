"""Rule-extractor registration facade.

The rule extractors live in the ``rules_*`` event-family submodules; each
``@register("event.type")`` decorator self-registers its extractor into
``zaxy.extract.core._RULES`` at import time.  This module's ONLY job is to
import every submodule so that all registrations fire.

Import-reachability contract: any submodule missing from the imports below is
silently skipped — its event types fall through to the generic identity
extractor with no error.  ``tests/test_extract.py`` pins the exact registered
event-type set to make that failure mode loud; update both when adding a
submodule or event type.
"""

from __future__ import annotations

from zaxy.extract import (
    rules_cognition as _rules_cognition,  # noqa: F401  (registration side effect)
)
from zaxy.extract import (
    rules_coordination as _rules_coordination,  # noqa: F401  (registration side effect)
)
from zaxy.extract import rules_indexing as _rules_indexing  # noqa: F401  (registration side effect)
from zaxy.extract import rules_memory as _rules_memory  # noqa: F401  (registration side effect)
from zaxy.extract import rules_workflow as _rules_workflow  # noqa: F401  (registration side effect)
