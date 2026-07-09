import asyncio
import time
import uuid

from society.actions import Action, ActionResult, Message, validate_action
from society.llm import BudgetExceeded


class Kernel:
    """Deterministic tick-barrier scheduler for AgentSociety.

    Deterministic given deterministic brains: each tick, every *eligible*
    agent's `brain.decide(view)` runs concurrently via asyncio.gather (views
    are built from pre-decide state, so brain latency cannot affect what
    any brain observes). Once all decisions are in, validate/execute/
    fifo-append/event-log effects are applied *sequentially*, in the fixed
    order of the awake list (sorted by agent id), so event order and
    message-send order within a tick do not depend on brain latency -- only
    on agent id. (The `think` action performs an LLM call during execute,
    so it runs sequentially with the other agents' effects; this is
    acceptable.) A brain exception is caught and recorded as a failed
    action for that agent rather than aborting the tick.

    Messages sent during a tick are only delivered into recipient inboxes
    after all steps of that tick have completed, so they become visible
    starting the next tick.
    """

    def __init__(
        self,
        agents: dict[str, "Agent"],
        worldmap,
        event_log,
        shared_memory=None,
        llm=None,
        metrics=None,
        config: dict | None = None,
    ):
        self.agents = agents
        self.worldmap = worldmap
        self.event_log = event_log
        self.shared_memory = shared_memory
        self.llm = llm
        self.metrics = metrics
        self.config = config or {}

        self.tick = 0
        self._pending: list[Message] = []
        self._budget_hit = False

        self.presence: dict[str, set] = {}
        self._build_presence()

    # ------------------------------------------------------------------
    # Presence index
    # ------------------------------------------------------------------
    def _build_presence(self) -> None:
        self.presence = {}
        for agent in self.agents.values():
            if agent.kind == "environment":
                continue
            loc = agent.location()
            if loc is not None:
                self.presence.setdefault(loc, set()).add(agent.id)

    def _presence_move(self, agent_id: str, origin, dest) -> None:
        if origin is not None:
            occupants = self.presence.get(origin)
            if occupants is not None:
                occupants.discard(agent_id)
                if not occupants:
                    del self.presence[origin]
        if dest is not None:
            self.presence.setdefault(dest, set()).add(agent_id)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def send(self, msg: Message) -> None:
        """Queue a message for delivery after the current tick's steps."""
        self._pending.append(msg)

    def deliver_pending(self) -> bool:
        """Deliver all messages queued via send() this tick into inboxes.

        Delivery clears the recipient's waiting state (a message always
        wakes a sleeping agent). Returns True if anything was delivered.
        """
        pending, self._pending = self._pending, []
        delivered_any = False
        for msg in pending:
            for rid in msg.recipients:
                recipient = self.agents.get(rid)
                if recipient is None:
                    self.event_log.append(
                        self.tick,
                        "system",
                        "kernel",
                        {"note": "undeliverable", "recipient": rid, "message_id": msg.id},
                    )
                    continue
                recipient.stm.inbox.put_nowait(msg)
                recipient.waiting_until = None
                delivered_any = True
                self.event_log.append(
                    self.tick,
                    "message",
                    msg.sender,
                    {"message": msg.to_dict(), "recipient": rid},
                )
                if self.metrics is not None and msg.kind in ("say", "gesture"):
                    on_message = getattr(self.metrics, "on_message", None)
                    if on_message is not None:
                        on_message(msg.sender, rid, msg.kind)
        return delivered_any

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    def _timeout_elapsed(self, a) -> bool:
        """Whether `a` is waiting on a real (non-forever) timeout that has
        elapsed as of the current tick. Shared by is_eligible() and the
        waiting-clear block in run() so the two never drift apart."""
        return (
            a.waiting_until is not None
            and a.waiting_until != -1
            and a.waiting_until <= self.tick
        )

    def is_eligible(self, a) -> bool:
        """Whether agent `a` should get a decide/execute cycle this tick."""
        if a.transit is not None:
            return False
        if a.stm.inbox.qsize() > 0:
            return True
        if a.waiting_until is None:
            return not a.stm.goals.empty()
        # a is waiting: only a real (non-forever) timeout that has elapsed
        # makes it eligible.
        return self._timeout_elapsed(a)

    # ------------------------------------------------------------------
    # Arrivals / transit
    # ------------------------------------------------------------------
    def _process_arrivals(self) -> None:
        for agent in self.agents.values():
            transit = agent.transit
            if transit is None or transit["arrive_at"] > self.tick:
                continue

            origin = agent.location()
            dest = transit["dest"]

            agent.stm.status.set("location", dest)
            self._presence_move(agent.id, origin, dest)

            arrival_msg = Message(
                id=str(uuid.uuid4()),
                sender="kernel",
                recipients=[agent.id],
                kind="arrival",
                content=dest,
                tick_sent=self.tick,
            )
            agent.stm.inbox.put_nowait(arrival_msg)
            agent.waiting_until = None

            if dest in self.agents:
                self.send(
                    Message(
                        id=str(uuid.uuid4()),
                        sender="kernel",
                        recipients=[dest],
                        kind="system",
                        content=f"{agent.id} arrived",
                        tick_sent=self.tick,
                    )
                )
            if origin is not None and origin in self.agents:
                self.send(
                    Message(
                        id=str(uuid.uuid4()),
                        sender="kernel",
                        recipients=[origin],
                        kind="system",
                        content=f"{agent.id} departed",
                        tick_sent=self.tick,
                    )
                )

            agent.transit = None

            self.event_log.append(
                self.tick,
                "system",
                agent.id,
                {"event": "arrival", "origin": origin, "dest": dest},
            )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    async def execute(self, agent, action: Action) -> ActionResult:
        name = action.name
        params = action.params

        if name == "noop":
            return ActionResult(True, data="noop")

        if name == "wait":
            timeout = params.get("timeout_ticks")
            if timeout is not None:
                agent.waiting_until = self.tick + int(timeout)
            else:
                agent.waiting_until = -1
            return ActionResult(True, data="waiting")

        if name == "pop_message":
            if agent.stm.inbox.empty():
                return ActionResult(False, error="inbox empty")
            msg = agent.stm.inbox.get_nowait()
            return ActionResult(True, data=msg.to_dict())

        if name == "peek_inbox":
            data = [
                {"sender": m.sender, "kind": m.kind} for m in agent.stm.inbox_items()
            ]
            return ActionResult(True, data=data)

        if name == "conclude":
            return ActionResult(True, data=params.get("text"))

        if name == "push_goal":
            agent.stm.goals.push(params["text"])
            return ActionResult(True, data="pushed")

        if name == "pop_goal":
            if agent.stm.goals.empty():
                return ActionResult(False, error="goal stack empty")
            popped = agent.stm.goals.pop()
            return ActionResult(True, data=popped)

        if name == "replace_goal":
            agent.stm.goals.replace(params["text"])
            return ActionResult(True, data="replaced")

        if name == "update_status":
            agent.stm.status.set(params["key"], params["value"])
            return ActionResult(True, data="updated")

        if name == "remove_status":
            agent.stm.status.remove(params["key"])
            return ActionResult(True, data="removed")

        if name in ("say", "gesture"):
            return self._execute_say_or_gesture(agent, action)

        if name == "act_on":
            return self._execute_act_on(agent, action)

        if name == "observe":
            return self._execute_observe(agent, action)

        if name == "read":
            return self._execute_read(agent, action)

        if name == "move":
            return self._execute_move(agent, action)

        if name in ("remember", "recall", "forget", "revise_memory"):
            return await self._execute_memory_action(agent, action)

        if name == "think":
            return await self._execute_think(agent, action)

        return ActionResult(False, error=f"not implemented: {name}")

    def _execute_say_or_gesture(self, agent, action: Action) -> ActionResult:
        """say/gesture: every target must exist and share sender's location,
        else no message is sent and the offenders are named in the error."""
        params = action.params
        targets = params["targets"]
        content = params["content"] if action.name == "say" else params["description"]

        sender_loc = agent.location()
        offenders = []
        for tid in targets:
            target = self.agents.get(tid)
            if target is None or target.location() != sender_loc:
                offenders.append(tid)

        if offenders:
            return ActionResult(
                False, error=f"targets not present at {sender_loc}: {', '.join(offenders)}"
            )

        msg = Message(
            id=str(uuid.uuid4()),
            sender=agent.id,
            recipients=targets,
            kind=action.name,
            content=content,
            tick_sent=self.tick,
        )
        self.send(msg)
        return ActionResult(True, data="sent")

    def _execute_act_on(self, agent, action: Action) -> ActionResult:
        """act_on(target, description): target must be an environment agent
        the actor is currently at. RuleBrain envs answer synchronously
        (wrapped as an env_result Message queued to the actor); LLM-brain
        envs instead receive an act_on Message in their own inbox."""
        params = action.params
        target_id = params["target"]
        description = params["description"]

        target = self.agents.get(target_id)
        if target is None or target.kind != "environment":
            return ActionResult(False, error=f"not an environment: {target_id}")
        if agent.location() != target_id:
            return ActionResult(False, error=f"not at {target_id}")

        handle_act_on = getattr(target.brain, "handle_act_on", None)
        if handle_act_on is not None:
            view = target.build_view(self.tick)
            reply = handle_act_on(agent.id, description, view)
            msg = Message(
                id=str(uuid.uuid4()),
                sender=target_id,
                recipients=[agent.id],
                kind="env_result",
                content=reply,
                tick_sent=self.tick,
            )
            self.send(msg)
        else:
            msg = Message(
                id=str(uuid.uuid4()),
                sender=agent.id,
                recipients=[target_id],
                kind="act_on",
                content=description,
                tick_sent=self.tick,
            )
            self.send(msg)
        return ActionResult(True, data="acted")

    def _is_readable(self, agent, target) -> bool:
        """An info_carrier is readable if it shares the reader's location,
        or is portable and currently held by the reader."""
        if target.location() is not None and target.location() == agent.location():
            return True
        if target.portable and target.holder == agent.id:
            return True
        return False

    def _execute_observe(self, agent, action: Action) -> ActionResult:
        target_id = action.params["target"]
        target = self.agents.get(target_id)
        if target is None:
            return ActionResult(False, error=f"no such target: {target_id}")

        if target.kind == "environment":
            occupants = []
            for oid in sorted(self.presence.get(target_id, set())):
                if oid == agent.id:
                    continue
                occ = self.agents.get(oid)
                if occ is None:
                    continue
                occupants.append(
                    {"id": occ.id, "kind": occ.kind, "status": occ.stm.status.public_view()}
                )
            return ActionResult(
                True,
                data={"status": target.stm.status.public_view(), "occupants": occupants},
            )

        if target.kind == "character":
            if target.location() != agent.location():
                return ActionResult(False, error=f"{target_id} not co-located")
            return ActionResult(True, data=target.stm.status.public_view())

        if target.kind == "info_carrier":
            if not self._is_readable(agent, target):
                return ActionResult(False, error=f"{target_id} not observable here")
            return ActionResult(
                True,
                data={
                    "meta": {"kind": target.kind, "portable": target.portable},
                    "status": target.stm.status.public_view(),
                },
            )

        return ActionResult(False, error=f"cannot observe kind {target.kind}")

    def _execute_read(self, agent, action: Action) -> ActionResult:
        params = action.params
        target_id = params["target"]
        query = params["query"]

        target = self.agents.get(target_id)
        if target is None or target.kind != "info_carrier":
            return ActionResult(False, error=f"not an info_carrier: {target_id}")
        if not self._is_readable(agent, target):
            return ActionResult(False, error=f"{target_id} not readable here")

        retrieve = getattr(target.brain, "retrieve", None)
        if retrieve is None:
            return ActionResult(False, error=f"{target_id} brain cannot retrieve")
        data = retrieve(query)
        return ActionResult(True, data=data)

    def _execute_move(self, agent, action: Action) -> ActionResult:
        destination = action.params["destination"]
        current = agent.location()

        dest_agent = self.agents.get(destination)
        if dest_agent is None or dest_agent.kind != "environment":
            return ActionResult(False, error=f"not an environment: {destination}")
        if destination == current:
            return ActionResult(False, error="already there")
        if not self.worldmap.connected(current, destination):
            return ActionResult(False, error=f"{destination} not connected from {current}")

        d = self.worldmap.distance(current, destination)

        self._presence_move(agent.id, current, None)
        if current is not None and current in self.agents:
            self.send(
                Message(
                    id=str(uuid.uuid4()),
                    sender="kernel",
                    recipients=[current],
                    kind="system",
                    content=f"{agent.id} departing to {destination}",
                    tick_sent=self.tick,
                )
            )

        agent.transit = {"dest": destination, "arrive_at": self.tick + d}
        return ActionResult(True, data={"eta": self.tick + d})

    async def _execute_memory_action(self, agent, action: Action) -> ActionResult:
        if self.shared_memory is None:
            return ActionResult(False, error="no shared memory")

        name = action.name
        params = action.params

        if name == "remember":
            data = await self.shared_memory.remember(agent.id, params["text"], self.tick)
            return ActionResult(True, data=data)

        if name == "recall":
            top_k = params.get("top_k", 5)
            data = await self.shared_memory.recall(agent.id, params["query"], top_k)
            return ActionResult(True, data=data)

        if name == "forget":
            data = self.shared_memory.forget(agent.id, params["memory_id"])
            return ActionResult(True, data=data)

        # revise_memory
        data = await self.shared_memory.revise(
            agent.id, params["memory_id"], params["new_text"], tick=self.tick
        )
        return ActionResult(True, data=data)

    async def _execute_think(self, agent, action: Action) -> ActionResult:
        if self.llm is None:
            return ActionResult(False, error="no llm configured")

        question = action.params["question"]
        view = agent.build_view(self.tick)
        prompt = f"Current view: {view}\n\nQuestion: {question}"
        reply = await self.llm.chat(prompt, bucket="think")
        return ActionResult(True, data=reply)

    # ------------------------------------------------------------------
    # Per-agent step (decide concurrently, apply effects sequentially)
    # ------------------------------------------------------------------
    async def _decide(self, agent) -> tuple:
        """Build the view and call brain.decide() for one agent.

        Returns (action, brain_error): brain_error is None on success, or a
        string description of the exception the brain raised. A brain
        exception is caught here (not propagated) so one misbehaving brain
        can neither abort the tick for its siblings nor leave a dangling
        background mutation after run() has moved on.
        """
        view = agent.build_view(self.tick)
        try:
            action = await agent.brain.decide(view)
        except Exception as exc:  # noqa: BLE001 - isolate brain failures per agent
            if isinstance(exc, BudgetExceeded):
                self._budget_hit = True
            return None, str(exc)
        return action, None

    async def _apply(self, agent, action, brain_error) -> None:
        """Validate + execute + fifo-append + event-log for one agent.

        Always called sequentially (never concurrently) across the awake
        set, in the fixed order the caller iterates, so event order and
        message-send order within a tick are deterministic.
        """
        if brain_error is not None:
            action = Action("<decide-error>", {})
            result = ActionResult(False, error=f"brain error: {brain_error}")
        else:
            error = validate_action(action)
            if error:
                result = ActionResult(False, error=error)
            else:
                try:
                    result = await self.execute(agent, action)
                except BudgetExceeded:
                    self._budget_hit = True
                    result = ActionResult(False, error="budget exceeded")

        agent.stm.fifo.append(
            {"name": action.name, "params": action.params}, result.to_dict()
        )
        self.event_log.append(
            self.tick,
            "action",
            agent.id,
            {
                "action": {"name": action.name, "params": action.params},
                "result": result.to_dict(),
                "location": agent.location(),
            },
        )

    # ------------------------------------------------------------------
    # Budget circuit-breaker
    # ------------------------------------------------------------------
    def _budget_exceeded(self) -> bool:
        """Whether the run should stop with stop_reason="budget".

        The real signal is `self._budget_hit`, set whenever a
        `society.llm.BudgetExceeded` exception surfaces from a brain's
        `decide()` (Phase 1) or from an action handler during `_apply()`
        (Phase 2, e.g. think/remember/recall/revise_memory). A duck-typed
        `metrics.budget_exceeded()` is also honored if present, so a
        Metrics subclass can opt into its own budget signal, but it is not
        required (Metrics does not implement it).
        """
        if self._budget_hit:
            return True
        if self.metrics is None:
            return False
        check = getattr(self.metrics, "budget_exceeded", None)
        if check is None:
            return False
        return bool(check())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(
        self, max_ticks: int | None = None, max_wall_seconds: float | None = None
    ) -> dict:
        start = time.monotonic()
        stop_reason = None

        while True:
            if max_ticks is not None and self.tick >= max_ticks:
                stop_reason = "max_ticks"
                break
            if (
                max_wall_seconds is not None
                and (time.monotonic() - start) >= max_wall_seconds
            ):
                stop_reason = "wall_time"
                break
            if self._budget_exceeded():
                stop_reason = "budget"
                break

            self._process_arrivals()

            awake = []
            for agent in self.agents.values():
                eligible = self.is_eligible(agent)
                if eligible and self._timeout_elapsed(agent):
                    # Waking from a timeout clears the waiting state.
                    agent.waiting_until = None
                if eligible:
                    awake.append(agent)

            if awake:
                # Phase 1: decide concurrently. Views were built from
                # pre-decide state, so brain latency cannot change what any
                # brain observes this tick.
                results = await asyncio.gather(
                    *(self._decide(a) for a in awake), return_exceptions=True
                )
                decisions = {}
                for agent, res in zip(awake, results):
                    if isinstance(res, Exception):
                        if isinstance(res, BudgetExceeded):
                            self._budget_hit = True
                        decisions[agent.id] = (None, str(res))
                    else:
                        decisions[agent.id] = res

                # Phase 2: apply effects sequentially, in a fixed order
                # (agent id) so event/message ordering within a tick is
                # deterministic regardless of decide() completion order.
                for agent in sorted(awake, key=lambda a: a.id):
                    action, brain_error = decisions[agent.id]
                    await self._apply(agent, action, brain_error)

            delivered = self.deliver_pending()

            if self.metrics is not None:
                maybe_snapshot = getattr(self.metrics, "maybe_snapshot", None)
                if maybe_snapshot is not None:
                    maybe_snapshot(self.tick)

            transit_pending = any(a.transit is not None for a in self.agents.values())
            waiting_timers = [
                a.waiting_until
                for a in self.agents.values()
                if a.waiting_until is not None and a.waiting_until != -1
            ]

            if not awake and not delivered and not transit_pending and not waiting_timers:
                stop_reason = "quiescent"
                break

            # Fast-forward only when nothing happened this tick: no agent
            # was awake AND nothing was delivered (an external kernel.send()
            # to a sleeping agent still counts as "something happened", so
            # we must not fast-forward past it -- and there may be no
            # timers/transit to compute a min() over in that case).
            if not awake and not delivered:
                candidates = [
                    a.transit["arrive_at"]
                    for a in self.agents.values()
                    if a.transit is not None
                ]
                candidates.extend(waiting_timers)
                if candidates:
                    self.tick = min(candidates)
                else:
                    self.tick += 1
            else:
                self.tick += 1

        return {"ticks_run": self.tick, "stop_reason": stop_reason}
