"""A plugin fixture requesting a projection backend, which cannot cross processes."""

from __future__ import annotations

from typing import Any


class BackendPlugin:
    """Fixture plugin registering a projection backend."""

    name = "remote-backend"
    version = "1.0"

    def register(self, api: Any) -> None:
        """Request a projection backend registration."""
        api.register_projection_backend("remote-null", lambda config: None)


PLUGIN = BackendPlugin()
