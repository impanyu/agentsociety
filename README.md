# AgentSociety

一个从零实现的多智能体互动框架:每个智能体拥有结构化短期记忆(STM)与共享
长期记忆(LTM),在各自的 observation → action 循环中异步运行,通过消息队列
交互;框架提供从小说/文本自动初始化场景(含地图)的抽取器 (`society.extract`)、
按全局时钟输出的剧本生成器 (`society.screenplay`),以及跨智能体长期记忆的
共识压缩机制。完整设计见 `docs/specs/2026-07-08-agent-society-design.md`,
action 参考手册见 `docs/actions.md`。

## 快速开始(Quickstart)

### 1. 环境准备

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp config.json.example config.json
```

编辑 `config.json`,填入真实的 `api_key`(OpenAI 兼容接口),按需调整
`base_url` / `chat_model` / `embed_model` / `max_concurrency` / `max_calls` /
`max_tokens`:

```json
{"api_key": "sk-...", "base_url": "https://api.openai.com/v1",
 "chat_model": "gpt-4o-mini", "embed_model": "text-embedding-3-small",
 "max_concurrency": 16, "max_calls": null, "max_tokens": null}
```

`api_key` 也可以用环境变量 `OPENAI_API_KEY` 代替;`config.json` 未找到时会
回退读该环境变量。`max_concurrency`(默认 16)传给 `LLMClient` 限制并发在
飞请求数;`max_calls` / `max_tokens`(默认均为 `null`,即不设上限)是跨所有
bucket 累计的调用次数 / token 数上限,一旦下一次调用会超出就抛出
`BudgetExceeded`,`kernel.run()` 据此在跑完当前 tick 后以
`stop_reason="budget"` 停止并落盘全部输出(细节见 `docs/actions.md`)。

设计文档中提到的可选 `--checkpoint`(运行中断点续跑)不在本阶段
(phase 1)范围内,当前 `society.run` 未实现该参数。

### 2. 运行内置的红楼梦 demo 场景

```bash
venv/bin/python -m society.run \
  --scenario scenarios/demo_red_chamber.yaml \
  --ticks 50 \
  --out runs/demo \
  --screenplay
```

`scenarios/demo_red_chamber.yaml` 是一个手写的《红楼梦》迷你场景:林黛玉、
贾宝玉、薛宝钗 3 个 `character`(brain: llm),潇湘馆、怡红院 2 个
`environment`(brain: rule),《石头记》1 个 `info_carrier`(brain: retrieval,
语料见 `scenarios/corpora/shitou_ji.txt`)。`--ticks 50` 限制最多跑 50 个
tick(也可能提前因静止态停止);`--out runs/demo` 指定输出目录;`--screenplay`
在跑完之后额外调用 `society.screenplay` 生成 `screenplay.md`。

其他 CLI 参数:

```bash
venv/bin/python -m society.run --help
```

```
--scenario   场景 yaml 路径(必填)
--ticks      最大 tick 数(必填)
--out        输出目录(必填)
--screenplay 跑完后额外生成 screenplay.md
--config     config.json 路径(默认 config.json)
```

### 3. 从小说文本抽取场景

`society.extract` 把一段自由文本(小说/故事)通过五阶段 LLM 流水线
(角色 → 地点+地图 → 信息载体 → 种子记忆 → kickoff)转成标准场景 YAML:

```bash
venv/bin/python -m society.extract \
  --input novel.txt \
  --output scenarios/my.yaml \
  --max-agents 15 \
  --language zh
```

```bash
venv/bin/python -m society.extract --help
```

```
--input       输入文本文件路径(必填)
--output      输出场景 yaml 路径(必填);info_carrier 语料会写到
              同目录下的 corpora/<id>.txt
--max-agents  agent 数量上限,默认 15(角色优先,再地点,再信息载体)
--language    zh 或 en,默认 zh
--hints       可选的抽取提示(聚焦哪些角色/地点等)
--config      config.json 路径(默认 config.json)
```

抽取产物是一次性离线缓存的 YAML,人可以先审改再用 `society.run --scenario`
加载;抽取器内部也会调用 `load_scenario` 自检产物的可加载性。

## 输出目录结构

`society.run --out <dir>` 跑完之后,`<dir>/` 下会有:

```
<dir>/
├── events.jsonl           # 全局事件日志,每条 action/message/system 事件一行
│                          # (JSON,含 tick + kind + agent/sender + 具体字段)
├── transcripts/
│   └── <agent_id>.md      # 每个 agent 的逐 tick action→result 流水账
│                          # (含收到的消息),人类可读
├── stats/
│   └── tick_NNNNNN.json   # 每 stats_interval(默认 10)个 tick 一份快照
│                          # (跑完还会补一份最终快照):
│                          #   consensus_ratio  — 共识条目(owners≥2)/ 总条目
│                          #   comm_graph       — 交流拓扑(say/gesture 计边,
│                          #                      有向 directed + 无向聚合 undirected)
│                          #   consensus_owners — 每条共识条目 {id, text, owners}
├── screenplay.md          # 仅当 --screenplay 时生成:剧本(离线读 events.jsonl,
│                          # 按 tick/地点/参与者切幕,LLM 两阶段筛选+渲染)
├── llm_usage.json         # LLM 调用按用途分桶的次数/token 统计
│                          # (decide/think/consensus/normalize/extract/screenplay)
└── config_snapshot.yaml   # 本次运行使用的完整场景配置快照 + run_summary
                           # (ticks_run、stop_reason),便于复现
```

## 测试

```bash
venv/bin/python -m pytest -q
```

## 架构概览

- **tick 屏障调度(`society/kernel.py`)**:全局时钟 `t = 0, 1, 2, …`。每个
  tick,当前"醒着"的每个智能体的 `brain.decide(view)` 并发执行(view 取自
  该 tick 开始前的状态,brain 延迟不影响任何人这一 tick 能看到什么);全部
  决策就绪后,按 agent id 排序**依次**校验/执行/写回 FIFO/写事件,保证同一
  tick 内事件与消息发送顺序确定、可复现。**消息投递**遵循 t+1 语义:t 时
  发出的消息要等到 t+1 才进入接收方收件队列。收件队列与目标栈都空的智能体
  进入休眠(零 LLM 成本),来消息即被唤醒;`move` 之后进入"在途"、多个 tick
  不被调度,到达时由内核推送 `arrival` 消息唤醒。全员休眠且无在途无在飞
  消息、无到期定时器 = 静止态,是三种停止条件(`max_ticks` / `max_wall_time`
  / 静止态,外加预算熔断)之一。详见 `docs/actions.md` 的"tick 语义"一节。

- **STM(`society/stm.py`、`society/agent.py`)**:每个智能体的短期记忆是
  "四件套":FIFO 缓存(`deque(maxlen=fifo_size)`,存最近若干条
  `(action, result)`)、目标栈(栈底最 fundamental,由场景 `goals` 自底向上
  注入,支持 push/pop/replace)、状态寄存器(公开/私有键值,`location` 永远
  公开且只能由 `move` 修改)、收件队列(`asyncio.Queue`)。`Agent.build_view`
  把这四者序列化成传给 `brain.decide()` 的 view。

- **Brains(`society/brains/`)**:统一接口 `async def decide(view) -> Action`。
  `LLMBrain` 用于 `character`(以及需要更复杂反应的 `environment`),把角色
  profile + actions skill(`society/skills/actions_skill_{zh,en}.md` 浓缩版)
  注入 system prompt,输出单个 action 的 JSON;`RuleBrain` 用于简单
  `environment`(默认 `act_on` 回复由 python 规则生成);`RetrievalBrain` 用于
  `info_carrier`,被 `read` 时基于语料检索作答,零 LLM 调用。

- **共享 LTM 与共识(`society/ltm.py`)**:单一 Chroma collection,每条记忆
  `{id, text, owners: set[agent_id], created_at, source, scenario, tick}`。
  `remember` 前先过规范化门(超长或多义征兆 → LLM 拆解/压缩为
  ≤`memory_max_chars`(默认 80)的原子短句),再走共识插入:embedding 检索
  top-k(默认 5)、相似度 ≥ `sim_threshold`(默认 0.86)的候选交给 LLM 批量
  判定语义等价——等价则合并 owners 并保留较短文本,不新增;不等价则新增。
  `forget` 只把自己移出 owners,owners 变空才物理删除;`revise_memory` 是
  "旧条目 forget + 新文本走一遍规范化+共识插入"的一步封装。`recall` 按
  owner 过滤 + embedding 相似度排序检索。

- **地图与移动(`society/worldmap.py`)**:节点是全部 `environment` agent,
  边是场景 `map.edges` 里显式的 `(a, b, distance)`;未显式给出的配对默认
  全联通、距离取 `map.default_distance`(场景内 `defaults.distance` 或全局
  默认 20)。

- **场景与抽取(`society/scenario.py`、`society/extract.py`)**:场景 YAML 是
  手写与自动抽取共用的唯一权威格式(`load_scenario` 做结构校验:agent
  id/kind 必填且唯一、brain 取值合法、初始 location 必须引用已定义的
  environment、地图边端点必须是已定义的 environment)。`society.extract`
  用五阶段 LLM 流水线(角色/地点+地图/信息载体/种子记忆/kickoff)把自由文本
  转成同样格式的场景。
