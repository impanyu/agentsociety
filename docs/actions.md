# Action 参考手册

面向人类读者的 action 手册。内容以代码为唯一权威来源:动作分类与必填参数取自
`society/actions.py`(`SYNC_ACTIONS` / `ASYNC_ACTIONS` / `REQUIRED_PARAMS` /
`validate_action`),校验与副作用语义取自 `society/kernel.py`(`Kernel.execute`
及其 `_execute_*` 辅助方法)。教智能体使用这些 action 的浓缩版说明书在
`society/skills/actions_skill_zh.md`(中文)与 `actions_skill_en.md`(英文),
会被 `LLMBrain` 注入到 system prompt 里;本文档是给人看的、更完整的参考。

一个 action 用 `{"action": "<name>", "params": {...}}` 表示,由
`society.actions.parse_action` 解析并调用 `validate_action` 校验:
- action 名必须在 `SYNC_ACTIONS | ASYNC_ACTIONS` 中,否则报 `Unknown action: ...`;
- `REQUIRED_PARAMS[name]` 列出的每个参数必须出现在 `params` 里,否则报
  `Missing required parameter '...' for action '...'`;
- 若 `params` 中出现 `targets` 键,必须是 list,否则报
  `Parameter 'targets' must be a list for action '...'`;
- `update_status` 的 `key` 不能是 `"location"`(保留键,只能通过 `move`
  修改),否则报 `Cannot use 'location' as a key in update_status (reserved)`。

校验失败的 action 不会执行:错误信息作为 `ActionResult(ok=False, error=...)`
写回该 agent 的 FIFO(见 `Kernel._apply`),brain 在下一次决策时能看到这条错误
并自行修正,不会中断整个 tick。

---

## 1. 同步 action(sync,当场返回结果)

共 18 个,定义于 `SYNC_ACTIONS`。执行结果当 tick 内就写回调用方 FIFO。

### `pop_message`
- 必填参数:无(`REQUIRED_PARAMS["pop_message"] = []`)。
- 语义:取出并**移除**收件队列(`agent.stm.inbox`)队首的一条 `Message`,
  结果 `data` 为该消息的完整字典(`Message.to_dict()`:id/sender/recipients/
  kind/content/tick_sent/correlation_id)。
- 校验/失败:若队列为空,返回 `ActionResult(False, error="inbox empty")`。

### `peek_inbox`
- 必填参数:无。
- 语义:只读取队列内容(`agent.stm.inbox_items()`),**不出队**,结果 `data`
  是一个列表,每项 `{"sender": ..., "kind": ...}`(不含消息正文)。
- 校验/失败:无特殊校验,队列为空时返回 `data=[]`。

### `think`
- 必填参数:`question`。
- 语义:用当前完整 view(`agent.build_view(tick)`)拼上 `question` 组成
  prompt,调用 `llm.chat(prompt, bucket="think")`,结果 `data` 为 LLM 回复
  文本。相对昂贵,skill 文档提醒智能体节制使用。
- 校验/失败:若内核未配置 `llm`,返回 `ActionResult(False, error="no llm configured")`。

### `conclude`
- 必填参数:`text`。
- 语义:不调用任何外部服务,直接把 `text` 作为结果 `data` 写回 FIFO(即"阶段性
  结论"),**不写入共享长期记忆**。是 `remember` 之前的沉淀步骤。
- 校验/失败:无特殊校验。

### `push_goal`
- 必填参数:`text`。
- 语义:在 `agent.stm.goals`(`GoalStack`,栈底=index 0=最 fundamental)
  **栈顶**压入 `text`,结果 `data="pushed"`。
- 校验/失败:无特殊校验。

### `pop_goal`
- 必填参数:无。
- 语义:弹出并返回栈顶目标,结果 `data` 为被弹出的目标文本。
- 校验/失败:栈为空时返回 `ActionResult(False, error="goal stack empty")`。

### `replace_goal`
- 必填参数:`text`。
- 语义:用 `text` 替换栈顶目标(栈深不变;栈为空时等价于 push),结果
  `data="replaced"`。
- 校验/失败:无特殊校验。

### `update_status`
- 必填参数:`key`, `value`。
- 语义:在 `agent.stm.status`(`StatusRegister`)中设置 `key -> value`,结果
  `data="updated"`。
- 校验/失败:`key == "location"` 时,`validate_action` 直接拒绝(见上文保留键
  规则),不会进入 `Kernel.execute`。

### `remove_status`
- 必填参数:`key`。
- 语义:从状态寄存器中删除该键(不存在则静默忽略),结果 `data="removed"`。
- 校验/失败:无特殊校验。

### `remember`
- 必填参数:`text`。
- 语义:调用共享长期记忆 `shared_memory.remember(agent_id, text, tick)`——先过
  规范化门(超长/多义 → LLM 拆解或压缩为 ≤`memory_max_chars` 的原子短句),
  再走共识插入(embedding top-k 检索相似候选,相似度 ≥ `sim_threshold` 时用
  LLM 批量判定等价;等价则合并 owners 并保留较短文本,不等价则新增)。结果
  `data` 为写入/合并后的条目信息。
- 校验/失败:内核未配置 `shared_memory` 时返回
  `ActionResult(False, error="no shared memory")`(此规则对 remember/recall/
  forget/revise_memory 通用)。

### `recall`
- 必填参数:`query`。可选 `top_k`(默认 5)。
- 语义:调用 `shared_memory.recall(agent_id, query, top_k)`,按 owner 过滤
  (只能检索到 owners 包含自己的条目)+ embedding 相似度排序,结果 `data`
  为候选文本/条目列表。既用于 `remember` 前查重,也用于一般检索。
- 校验/失败:同 `remember` 的"无共享记忆"规则。

### `forget`
- 必填参数:`memory_id`。
- 语义:调用 `shared_memory.forget(agent_id, memory_id)`——把自己从该条目的
  owners 中移除;owners 变空才真正物理删除(仍有其他 owner 则保留)。
- 校验/失败:同上"无共享记忆"规则。

### `revise_memory`
- 必填参数:`memory_id`, `new_text`。
- 语义:调用 `shared_memory.revise(agent_id, memory_id, new_text, tick=tick)`,
  语义上等价于对旧条目 `forget` 之后让 `new_text` 走一遍规范化+共识插入,一步
  完成"修订"。
- 校验/失败:同上"无共享记忆"规则。

### `observe`
- 必填参数:`target`。
- 语义(按目标 `kind` 分派,`Kernel._execute_observe`):
  - `target` 不存在:`ActionResult(False, error="no such target: ...")`。
  - `environment`:返回 `{"status": <环境公开状态>, "occupants": [{"id","kind","status"}...]}`,
    在场者取自 `presence` 索引(在场索引由 `move` 与初始配置维护),
    按 id 排序,不含 observe 发起者自己。**无同地限制**——可以远程 observe
    任意 environment。
  - `character`:必须与观察者**同地**(`target.location() == agent.location()`),
    否则 `ActionResult(False, error="{target} not co-located")`;通过则返回其
    公开状态(`StatusRegister.public_view()`,默认排除 `mood` 等私有键)。
  - `info_carrier`:必须"可读"(见下方 `_is_readable` 规则),否则
    `ActionResult(False, error="{target} not observable here")`;通过则返回
    `{"meta": {"kind", "portable"}, "status": <公开状态>}`。
  - 其他 kind:`ActionResult(False, error="cannot observe kind {kind}")`。

### `read`
- 必填参数:`target`, `query`。
- 语义:`target` 必须是 `info_carrier` 且对读者"可读"(`_is_readable`:
  target 与 agent 同地,或 target 是 `portable` 且 `target.holder == agent.id`),
  再由内核直调该载体 brain 的被动接口 `retrieve(query)`(不占用载体自己的
  tick),结果 `data` 为检索到的相关片段。
- 校验/失败:
  - 非 info_carrier 或不存在:`not an info_carrier: {target}`;
  - 不可读:`{target} not readable here`;
  - 载体 brain 不支持 `retrieve`(例如误配成 rule/llm brain 的载体):
    `{target} brain cannot retrieve`。

### `move`
- 必填参数:`destination`。
- 语义:发起离场(见下文"移动"pipeline 与 tick 语义节)。
- 校验/失败(`Kernel._execute_move`,按顺序检查):
  1. `destination` 必须是已定义的 `environment` agent,否则
     `not an environment: {destination}`;
  2. `destination == current` 时 `already there`;
  3. `worldmap.connected(current, destination)` 必须为真,否则
     `{destination} not connected from {current}`(见 §"地图与移动")。
  - 通过后:立即离开当前位置(在场索引移除),若原位置也是一个 agent
    (即 `current` 是 environment id),会收到一条 `system` 消息
    `"{agent} departing to {destination}"`;随后进入 `transit`
    (`{"dest": destination, "arrive_at": tick + distance}`),结果
    `data={"eta": tick + distance}`。

### `wait`
- 必填参数:无。可选 `timeout_ticks`。
- 语义:带 `timeout_ticks` 时,`agent.waiting_until = tick + timeout_ticks`
  (到期自动唤醒);不带时 `agent.waiting_until = -1`(只有收到消息才会被
  唤醒,即"等到天荒地老")。结果 `data="waiting"`。等待期间该 agent
  不会被调度(不消耗 LLM)。

### `noop`
- 必填参数:无。
- 语义:什么也不做,结果 `data="noop"`。框架在 brain 输出解析失败/异常时
  自动兜底为 `noop` + 错误 result;智能体也可以在确实无事可做时主动选择它。

---

## 2. 异步 action(async,可能跨 tick 才真正送达)

共 3 个,定义于 `ASYNC_ACTIONS`。在**发起的当 tick 就返回 `ok`**(表示"已发出"),
但实际投递到接收方要等到下一个 tick(见 tick 语义一节)。

### `say`
- 必填参数:`targets`(list),`content`。
- 语义(`Kernel._execute_say_or_gesture`):校验 `targets` 中每一个 id 都存在
  且与发起者**同地**(`target.location() == sender.location()`);任何一个不
  满足都算"违规者"(offender)。全部通过后,构造一条
  `Message(kind="say", content=content, recipients=targets, tick_sent=当前tick)`,
  经 `kernel.send()` 排入待投递队列(见"消息投递"),结果 `data="sent"`。
- 校验/失败:只要有违规者,**整条消息都不发送**,返回
  `ActionResult(False, error="targets not present at {sender_loc}: {违规者id列表,逗号分隔}")`。

### `gesture`
- 必填参数:`targets`(list),`description`。
- 语义:与 `say` 完全相同的校验与投递机制,只是 `content` 取自
  `description`,`Message.kind="gesture"`;语义上表达非语言的动作/姿态。
- 校验/失败:同 `say`。

### `act_on`
- 必填参数:`target`,`description`。
- 语义(`Kernel._execute_act_on`):`target` 必须是 `environment` 且发起者当前
  就在该环境(`agent.location() == target_id`)。
  - 若目标环境的 brain 提供 `handle_act_on(agent_id, description, view)`
    (`RuleBrain` 默认提供,回复形如 `"{agent_id} 对环境做了: {description}"`,
    场景可自定义规则),内核**当场同步调用**它,把回复包装成一条
    `kind="env_result"` 消息发回发起者的收件队列(仍要等到下一 tick 才在
    `agent.stm.inbox` 中出现,因为走的是同一个 `send()`/`deliver_pending()`
    机制)。
  - 若目标环境是 `LLMBrain`(没有 `handle_act_on`),则改为把一条
    `kind="act_on"` 的消息直接送进目标环境自己的收件队列,由它在自己的下一次
    `decide()` 循环里处理并自行决定如何回应(不保证同步收到回复)。
  - 无论哪种情况,发起者当 tick 就拿到 `ActionResult(True, data="acted")`。
- 校验/失败:`target` 不是已定义 environment,或不是自己当前所在环境,均返回
  `False`(错误分别为 `not an environment: {target}` / `not at {target}`)。

---

## 3. 五种典型 pipeline

与 `society/skills/actions_skill_zh.md` 保持一致,LLMBrain 会把浓缩版注入
system prompt;这里给出对应到内核校验规则的说明。

### 3.1 消息处理
`peek_inbox` → `pop_message` → 依据消息内容 `push_goal` 压入对应小目标 →
围绕该目标反复执行合适的 action(`observe` / `think` / `say` / `recall` 等)
直到目标达成 → `pop_goal` 弹出、回到更上层目标或 `wait`。
`peek_inbox` 让智能体在真正花成本处理消息前先判断"值不值得现在处理"。

### 3.2 社交
`observe`(当前所在 environment,取得 `occupants` 在场者集合与其公开状态)→
从在场者中选出 `targets`(必须是真实存在且与自己同地的 agent,否则
`say`/`gesture` 会因"违规者"而整条失败)→ `say` 或 `gesture` 表达 → `wait`
等待对方回应(对方的回复消息下一 tick 送达时会自动唤醒等待中的自己)。

### 3.3 移动
`observe` 当前环境确认位置与相邻环境 → `move(destination)` 发起(内核校验
destination 是 environment 且经 `worldmap.connected()` 判定与当前位置连通)
→ 校验通过后立即离开原环境,进入"在途"状态(时长 = `worldmap.distance()`,
默认全联通 + `defaults.distance` 个 tick,或地图 `edges` 中显式指定的距离),
在途期间不被调度、也不能执行任何 action → 到达时内核自动把
`status.location` 更新为新环境、在场索引更新、并投递一条 `arrival` 消息把
智能体唤醒 → 智能体 `observe` 新环境,了解在场者与环境状态。

### 3.4 记忆卫生
`remember` 之前先用 `recall(query)` 查一遍长期记忆,避免录入重复事实;若只是
阶段性判断、尚未确定为稳定事实,先用 `conclude` 把结论沉淀进 FIFO 而不是急着
`remember`;结论被反复验证或明确认可后,再 `remember` 写入共享长期记忆
(会经过规范化门与共识合并,见 `remember` 一节);发现旧记忆有误/过时时,用
一步到位的 `revise_memory`,而不是自己手动拆成 `forget` + `remember`。

### 3.5 目标管理
根本目标(fundamental)由场景在初始化时注入到目标栈**栈底**(`goals` 列表
从底到顶的顺序压入),一般不应被自己 `pop_goal` 掉;当前要执行的具体目标在
**栈顶**,用 `push_goal` 添加更细粒度的子目标;子目标一旦达成立即
`pop_goal`,避免目标栈无限增长导致决策失焦;若目标的表述需要调整但层级不变,
用 `replace_goal` 而不是先 `pop_goal` 再 `push_goal`。

---

## 4. tick 语义简述

- **全局时钟** `t = 0, 1, 2, …`;每个 tick,当前"醒着"的每个智能体恰好完成一次
  `decide → validate → execute → 写回 FIFO/事件日志` 循环。同一 tick 内所有
  醒着智能体的 `decide()` 并发执行(view 取自 decide 前的状态,因此 brain
  延迟不会改变任何人这一 tick 看到的东西),随后按 agent id 排序**依次**
  应用副作用(校验/执行/写 FIFO/写事件),保证事件顺序与消息发送顺序在同一
  tick 内是确定性的、与 brain 完成先后无关。
- **消息投递(t+1 语义)**:tick 内通过 `say`/`gesture`/`act_on`/`move` 产生的
  消息只是暂存(`kernel.send()`),要等这一整个 tick 的所有步骤都跑完之后才
  被投递进各接收方的收件队列(`deliver_pending()`)——也就是说,t 时发出的
  消息最早在 t+1 才对接收方可见。
- **休眠(sleep)**:收件队列为空**且**目标栈为空的智能体,这一 tick 不会被
  调度(`is_eligible` 返回 False),不消耗任何 LLM 调用;一旦收件队列收到
  新消息(包括 `arrival`/`system` 系统消息),会在下一次判定时立刻被唤醒变
  为"醒着"。
- **wait**:`wait()` 会把自己标记为"等待中"(`waiting_until = -1` 表示只有
  消息能唤醒;带 `timeout_ticks` 则到期自动唤醒),等待期间同样不被调度。
- **在途(transit)**:执行 `move` 之后到到达目的地之前的这段时间,智能体
  处于"在途"状态,连续多个 tick 完全不被调度(`is_eligible` 对处于 transit
  的 agent 恒返回 False),到达时由内核推送一条 `arrival` 消息唤醒它。
- **静止态(quiescence)**:当某一 tick 结束时,没有智能体醒着、没有消息被
  投递、没有智能体在途、也没有任何计时器(`waiting_until` 的非 `-1` 定时器)
  在等待触发——四者同时满足,视为整个社会"静止",`kernel.run()` 以
  `stop_reason="quiescent"` 停止。若没有醒着的智能体也没有消息投递,但存在
  在途/定时器,时钟会直接快进到最近的到达/超时时刻(空转推进),而不是逐 tick
  空转。其余停止条件:达到 `max_ticks`、达到 `max_wall_seconds`、或触发预算
  熔断——`society/llm.py` 的 `LLMClient` 在调用前发现将超出 `max_calls`/
  `max_tokens` 时抛出 `BudgetExceeded`;`Kernel` 在 decide 阶段
  (`_decide`,brain 抛出的 `BudgetExceeded`)与 apply/execute 阶段
  (`_apply`,`think`/`remember`/`recall`/`revise_memory` 等异步 handler 抛出
  的 `BudgetExceeded`)分别捕获它,置位内部标志 `self._budget_hit = True`
  (对应的 action 仍按失败记录,不会让其它智能体这一 tick 的效果落空);
  下一次 tick 循环开始时 `_budget_exceeded()` 读到该标志即以
  `stop_reason="budget"` 停止。无论哪种停止条件触发,都会先跑完当前
  tick 再落盘全部输出。
