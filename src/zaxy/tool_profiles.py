"""MCP tool surface profiles.

Profiles change which tools the MCP server lists, never which tools it can
run: dispatch stays unfiltered, so any tool remains callable by name under
any profile. The ``core`` profile lists the front-door verb set with
``memory_checkout`` as the entry point and ``memory_capabilities`` as the
discovery surface; the ``full`` profile lists everything.
"""

from __future__ import annotations

CORE_TOOLS: frozenset[str] = frozenset(
    {
        "memory_checkout",
        "memory_append",
        "memory_query",
        "context_assemble",
        "memory_feedback",
        "memory_invalidate",
        "memory_capabilities",
        # Reserved for the 2.2 metamemory pre-check. Profiles only affect
        # listing, so naming a not-yet-shipped tool here is harmless: it is
        # filtered against the defined Tool table before being listed.
        "memory_feeling_of_knowing",
    }
)

TOOL_PROFILE_NAMES: tuple[str, ...] = ("core", "full")


def resolve_profile(profile_name: str) -> frozenset[str] | None:
    """Return the listed tool names for a profile, or None for the full surface.

    Args:
        profile_name: Profile identifier, ``core`` or ``full``.

    Returns:
        The frozenset of listed tool names for ``core``, or None for ``full``
        (no listing filter).

    Raises:
        ValueError: If the profile name is not a known profile.
    """
    normalized = profile_name.strip().casefold()
    if normalized == "full":
        return None
    if normalized == "core":
        return CORE_TOOLS
    valid = ", ".join(TOOL_PROFILE_NAMES)
    raise ValueError(f"Unknown MCP tool profile: {profile_name!r}. Valid profiles: {valid}.")
