class WorldMap:
    """Static topology of environment (location) ids.

    Locations are connected either implicitly (when `fully_connected=True`,
    every pair of known env ids is connected at `default_distance`) or
    explicitly via `edges`, a list of `(a, b, distance)` tuples. Explicit
    edges always override the default distance, even when `fully_connected`
    is True.
    """

    def __init__(
        self,
        env_ids: list[str],
        edges: list[tuple[str, str, int]] | None = None,
        default_distance: int = 20,
        fully_connected: bool = True,
    ):
        self.env_ids = list(env_ids)
        self.default_distance = default_distance
        self.fully_connected = fully_connected
        self._edges: dict[tuple[str, str], int] = {}
        if edges:
            for a, b, dist in edges:
                self._edges[(a, b)] = dist
                self._edges[(b, a)] = dist

    def connected(self, a, b) -> bool:
        """Return True if a and b are directly connected (or identical)."""
        if a == b:
            return True
        if (a, b) in self._edges:
            return True
        if self.fully_connected and a in self.env_ids and b in self.env_ids:
            return True
        return False

    def distance(self, a, b) -> int | None:
        """Return the distance between a and b, or None if not connected.

        Explicit edges take precedence over the fully-connected default.
        """
        if a == b:
            return 0
        if (a, b) in self._edges:
            return self._edges[(a, b)]
        if self.fully_connected and a in self.env_ids and b in self.env_ids:
            return self.default_distance
        return None
