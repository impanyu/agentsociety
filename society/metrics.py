import json
import os
from collections import defaultdict


class Metrics:
    """Track communication patterns and consensus metrics for the agent society."""

    def __init__(
        self,
        agents: dict,
        shared_memory,
        out_dir: str | None,
        interval: int = 10,
    ):
        """
        Initialize Metrics.

        Args:
            agents: Dict of agent_id -> agent (reserved for future use).
            shared_memory: SharedMemory instance or None.
            out_dir: Directory to write stats snapshots to, or None.
            interval: Interval for periodic snapshots (tick % interval == 0).
        """
        self.agents = agents
        self.shared_memory = shared_memory
        self.out_dir = out_dir
        self.interval = interval
        # Track message counts: (sender, recipient) -> count
        self._directed_edges = defaultdict(int)

    def on_message(self, sender: str, recipient: str, kind: str) -> None:
        """
        Record a message between agents.

        Only counts messages with kind in {"say", "gesture"}.

        Args:
            sender: ID of the sending agent.
            recipient: ID of the receiving agent.
            kind: Kind of message (say, gesture, etc.).
        """
        if kind not in ("say", "gesture"):
            return
        edge = f"{sender}->{recipient}"
        self._directed_edges[edge] += 1

    def snapshot(self, tick: int) -> dict:
        """
        Build a snapshot of consensus and communication metrics.

        Always builds the dict; writes to {out_dir}/stats/tick_%06d.json if out_dir is set.

        Args:
            tick: Current tick number.

        Returns:
            Dict with keys: tick, consensus_ratio, comm_graph, consensus_owners.
        """
        # Build consensus_ratio from shared_memory
        if self.shared_memory is None:
            consensus_ratio = {"total": 0, "shared": 0, "ratio": 0.0}
            consensus_owners = []
        else:
            stats = self.shared_memory.stats()
            consensus_ratio = {
                "total": stats["total"],
                "shared": stats["shared"],
                "ratio": stats["ratio"],
            }
            # Build consensus_owners: entries with len(owners) >= 2
            consensus_owners = []
            for entry in self.shared_memory.all_entries():
                if len(entry["owners"]) >= 2:
                    consensus_owners.append(
                        {
                            "id": entry["id"],
                            "text": entry["text"],
                            "owners": entry["owners"],
                        }
                    )

        # Build communication graph
        directed = {}
        for edge, count in self._directed_edges.items():
            directed[edge] = count

        # Build undirected graph by summing both directions
        undirected = {}
        for edge in self._directed_edges.keys():
            sender, recipient = edge.split("->")
            # Create normalized key (sorted pair joined by "|")
            pair = sorted([sender, recipient])
            key = f"{pair[0]}|{pair[1]}"
            if key not in undirected:
                undirected[key] = 0
            undirected[key] += self._directed_edges[edge]

        comm_graph = {"directed": directed, "undirected": undirected}

        snapshot_dict = {
            "tick": tick,
            "consensus_ratio": consensus_ratio,
            "comm_graph": comm_graph,
            "consensus_owners": consensus_owners,
        }

        # Write to disk if out_dir is set
        if self.out_dir is not None:
            stats_dir = os.path.join(self.out_dir, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            filename = os.path.join(stats_dir, f"tick_{tick:06d}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(snapshot_dict, f, ensure_ascii=False)

        return snapshot_dict

    def maybe_snapshot(self, tick: int) -> dict | None:
        """
        Conditionally snapshot based on the interval.

        Returns the snapshot dict only when tick > 0 and tick % interval == 0.

        Args:
            tick: Current tick number.

        Returns:
            Snapshot dict or None.
        """
        if tick > 0 and tick % self.interval == 0:
            return self.snapshot(tick)
        return None
