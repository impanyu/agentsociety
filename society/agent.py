from society.stm import STM


class Agent:
    """A kernel-scheduled entity: a character, environment, or info_carrier.

    Holds identity/config (id, kind, brain, stm, profile, portable, holder)
    plus kernel-managed runtime state (waiting_until, transit) that the
    Kernel mutates directly as part of the tick loop.
    """

    def __init__(
        self,
        agent_id: str,
        kind: str,
        brain,
        stm: STM,
        *,
        portable: bool = False,
        holder: str | None = None,
        profile: str = "",
    ):
        self.id = agent_id
        self.kind = kind
        self.brain = brain
        self.stm = stm
        self.profile = profile
        self.portable = portable
        self.holder = holder

        # Kernel-managed runtime state.
        # waiting_until: None = not waiting; -1 = waiting forever (only a
        # message wakes it); int = tick at which it wakes from a timeout.
        self.waiting_until: int | None = None
        # transit: None = not moving; else {"dest": str, "arrive_at": int}.
        self.transit: dict | None = None

    def location(self) -> str | None:
        """Current location id, from the status register (or None)."""
        return self.stm.status.get("location")

    def build_view(self, tick: int) -> dict:
        """Build the serialized view passed to brain.decide()."""
        inbox_size = self.stm.inbox.qsize()
        inbox_head = None
        queue = self.stm.inbox_items()
        if queue:
            head = queue[0]
            inbox_head = {"sender": head.sender, "kind": head.kind}

        return {
            "tick": tick,
            "agent_id": self.id,
            "kind": self.kind,
            "goals": self.stm.goals.items(),
            "status": self.stm.status.all(),
            "fifo": [
                {"action": action, "result": result}
                for action, result in self.stm.fifo.items()
            ],
            "inbox_size": inbox_size,
            "inbox_head": inbox_head,
        }
