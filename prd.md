下面是这个脚本的**完整逻辑步骤**（从读入到产出），按执行顺序说明关键环节、核心算法与可配参数。

# 整体流程

1. **读取输入 & 预处理**
* 创建一个class `FileProcessor`，处理文件
* 接受一个参数 input (默认: `assets/support_messages_andy.csv`)
* 文件读入到dataframe：使用 `pd.read_csv()`
* Dataframe预处理：
  * 新增一列：role，如果**Sender ID**是`psops`开头就是`customer_service`，否则就是`user`，如果已经有role，就不动。
  * 将 **Created Time** 解析为**时区感知**时间并统一为 **UTC**（使用 `pd.to_datetime()` 和 `pytz.UTC`）。
  * 按**Channel URL**分组，组内先按 **Created Time** 升序，再按 **Message ID** 升序；
  * 新增一列：msg_ch_idx: 对于**Channel URL**生成每一个group msg_ch_idx = 0..N-1（使用 `groupby().cumcount()`）。
  * 生成一个新的df，只保留必须的column：
    - 'Created Time'
    - 'Sender ID'  
    - 'Message'
    - 'Channel URL'
    - 'role'
    - 'msg_ch_idx'
    - 'Message ID'
* 输出到 `out/[source_filename]_out.csv`

  

2. **分块与重叠**

* 创建一个class `ChannelSegmenter`，接受 FileProcessor 的 clean DataFrame
* 使用 `--chunk-size/-c`（默认 80 行）将整段对话切成多个块；块与块之间保留 `--overlap/-l`（默认 20 行）重叠。
* **参数约束**: `overlap < chunk_size/3` 以确保消息不会出现在超过2个相邻块中
* **频道分离**: 按 Channel URL 分组，每个频道独立处理分块，不混合不同频道的消息
* **半开区间**: 使用 `[start, end)` 格式避免边界重复：
  * 第1块：`[0, chunk_size)`
  * 第2块：`[chunk_size - overlap, 2 * chunk_size)`  
  * 第i块（i≥2）：`[(i-1) * chunk_size - overlap, i * chunk_size)`
* **数据结构**: 每个 Chunk 包含：
  * chunk_id, channel_url, start_idx, end_idx
  * messages (DataFrame slice), total_messages 
  * has_overlap_with_previous, overlap_size
  * `format_for_prompt()`: 格式化为 `msg_ch_idx | sender_id | role | timestamp | text`

3. **构造“上文摘要”**

* 对每个块，取**上一个块的尾部**（最多 `overlap` 行）生成简短的“Previous context summary”：

  * 列出最近几条消息（含 `message_index`、`sender id`、`role`、`timestamp` 和 `text`）。
  * 明确提示模型：应基于这些消息推断**仍未解决的 case**，以保持跨块连续性。
  * 列出必要的信息（建议包含：**订单/订单号**、**买家/用户标识**、**当前 case 解决状态**（open/ongoing/resolved/blocked）、**最近一次动作及时间**（例如“已改为自提/已创建退货单/已回复证明”+ 时间）、**待处理方**（seller/agent）、**关键实体**（如快递单号、SKU/产品名）、**关键关键词**（退款/换货/发货/自提/取消等）、**是否返回到旧话题**、**相关 message\_index 列表**）。
  * 若是**第一个块**，则写明“无上文”。

4. **格式化块内消息**

* 将当前块内每条消息格式化为：
  `message_index | sender id | role | timestamp | text`
* 汇总为纯文本区域，作为 Prompt 的 `{chunk_block}`。

5. **组装 Prompt**

* 将“任务说明 + 决策标准 + 模糊处理策略 + 强制 JSON 输出结构 + Previous context summary + 当前块消息”拼成**单一 Prompt**：

  * 要求输出 **严格 JSON**：

    ```json
    {
      "complete_cases": [
        {
          "msg_list": [0,1,2,5],
          "summary": "Brief description …",
          "confidence": 0.9
        }
      ],
      "total_messages_analyzed": <total_number_of_messages>
    }
    ```
  * 明确跨块要**延续未完结的 case**，新问题才新建 case。

6. **调用 LLM（两种方式其一）**

* **Responses API**（推荐，`--use_responses_api` 开启）：强制 `response_format={"type":"json_object"}` 以获得稳定 JSON。
* **Chat Completions**：同样强制 JSON 输出。
* 模型名由 `--model` 指定（默认 `"gpt-4.1-mini"`）。

7. **解析 JSON**

* 正常路径：直接 `json.loads`。
* 容错路径：若返回混入非 JSON 文本，使用正则抓取首个 `{...}` 再 `json.loads`；解析失败则抛错。

8. **块内 case→全局 case 映射（跨块合并的核心）**

* 维护一张**全局映射表**：`message_index -> global_case_id` 和 `next_case_id` 计数器。
* 对当前块返回的每个 case：

  * 查看其 `msg_list` 是否**与已有消息**（上一块及更早块）有重叠：

    * **有重叠**：采用**先占先得**原则，已分配的 `message_index` 不可覆盖。若新 case 包含已分配消息：
      - 将 `msg_list` 拆分为**已分配部分**和**未分配部分**
      - 已分配部分保持原归属，未分配部分可新建 case 或合并到其他 case
      - 若未分配部分太少（< 2条消息），则丢弃该 case
    * **无重叠**：分配新 `global_case_id = next_case_id`，并自增计数器。
  * **仅将未分配的** `message_index` 写入映射表（建立**全局归属**），避免覆盖冲突。
  * 同时把该 case（附 `global_case_id/summary/confidence`）保存到中间结果里。

9. **块间推进**

* 处理完一个块后，移动到下一个块；"上文摘要"由**当前块开始位置之前的消息**生成（避免包含当前块内容）；由于映射表已累积，重叠消息会自然把跨块的相同话题**串起来**。

10. **生成逐行 case 标注**

* 所有块处理结束后，根据 `message_index -> global_case_id` 映射，为原始 CSV 增加一列 `case_id`，输出为：

  * `out/segmented.csv`

11. **聚合全局 case 列表**

* 将各块的 case（含 `global_case_id`）按全局 ID 聚合：

  * 合并 `msg_list`（去重、排序）。
  * `summary` 选择**满足“必含字段”要求**且信息量更高者（若多条均满足，取最近时间覆盖的那条；若均不满足，取最长并在后处理阶段补齐缺项）。
  * `confidence` 取**均值**（四舍五入到三位小数）。
* **后处理补齐（可选）**：如最终 `summary` 未覆盖 `order/user/status` 等必项，可基于 `msg_list` 对应的消息做轻量提取，将缺失项以 `order: N/A` / `status: open`（启发式）等形式补齐再落盘。
* 生成最终 JSON：

  * `out/cases.json`（包含 `"complete_cases": [...]`, `"total_messages_analyzed": N`）

12. **参数与默认值**

* `--chunk-size/-c` = 80（按行数分块；若需按 token 可替换为 token 估算）
* `--overlap/-l` = 20（跨块上下文粘连，必须 < chunk_size/3）
* `--input/-i` = `assets/support_messages_andy.csv`（输入 CSV 文件）
* `--output-dir/-o` = `out`（输出目录）
* `--model` = `"gpt-4.1-mini"`
* `--use_responses_api`：启用 Responses API 强制 JSON 输出

13. **健壮性与可扩展点**

* **JSON 容错**：提供了抓 `{...}` 的兜底解析。
* **列名容错**：自动猜测并允许显式覆盖。
* **跨块冲突处理**：如同一块 case 与多个既有全局 ID 重叠，取**最小 ID** 统一并避免分裂。
* **关键词/结束判定**：脚本包含 `CLOSURE_KEYWORDS` 与 `detect_closed_case`（当前未强制使用），可扩展用于：

  * 影响“上文摘要”的“活跃 case”判断；
  * 在 merge 阶段辅助判定是否应延续或关闭。
* **可替换的分块策略**：目前按行数分块；可改为按 token（集成 `tiktoken`）更稳。

14. **复杂度与性能**

* 线性遍历消息（`O(N)`），合并依赖哈希映射；LLM 调用成本取决于**块数 × Prompt 长度**。
* 通过重叠与上文摘要，尽量减少误切与跨块断裂，兼顾成本与准确度。

---

如果你希望，我可以：

* 根据你实际 CSV 的列名/长度，给出一条**一键运行命令**；
* 或者把“结束/新话题”**领域关键词**（如退款/发货/换货/售后等）扩充进 Prompt 的“Factors/Keywords”部分，进一步提高稳定性。

---

## 🔄 更新：分块与标注工作流（依据新要求）

### 1) 分块策略（半开区间）

* 使用**半开区间** `[start, end)`，避免边界重复。
* 块区间计算公式：
  * 第1块：`[0, chunk_size)`
  * 第2块：`[chunk_size - overlap, 2 * chunk_size)`
  * 第3块：`[2 * chunk_size - overlap, 3 * chunk_size)`
  * 第i块（i≥2）：`[(i-1) * chunk_size - overlap, i * chunk_size)`
* 具体示例（chunk_size=100, overlap=30）：
  * 第1块：`[0, 100)`
  * 第2块：`[70, 200)` 
  * 第3块：`[170, 300)`
  * 第4块：`[270, 400)` …
* 末块安全处理：`end = min(N, i * chunk_size)`，不越界。
* **重叠约束**：严格要求 `overlap < chunk_size/2`，确保任何消息最多只会出现在相邻的2个块中。
* **重叠实现**：从第2块开始，每个块的前 `overlap` 条消息与上一个块重叠，保证上下文连续性。

### 2) 排序与 ID 约束

* **message\_index**：由排序后顺序生成（`0..N-1`），仅用于分析与输出引用；**不替代** CSV 中原有的 **Message ID**。
* **排序规则**：先按 **Created Time** 升序，再按 **Message ID** 升序；随后生成 `message_index`。
* **时间标准化**：`Created Time` 解析为 UTC（或指定统一时区），避免跨时区导致的时序误判。

### 3) 行格式（供 Prompt/摘要使用）

* 统一为：

  `message_index | sender id | role | timestamp | text`

* 如果存在多个频道（Channel URL），请按频道拆分，逐频道独立处理（分块、上文摘要、LLM 调用与合并），不要在同一块混合多个频道。

### 4) 上文摘要（Previous context summary）

**目标**：在每个块开始前，为模型提供**紧凑且判别性强**的前情提示，帮助其延续未完结的 case、避免误切。

#### 4.1 选取范围

* 从**当前块开始位置之前**的最近 `overlap` 条消息作为上文；若 `overlap` 很小（<5），下限取 5 条（`min_prev = 5`）。
* 具体实现：对于第i块（i≥2），上文消息范围为 `[max(0, (i-1)*chunk_size - min_prev), (i-1)*chunk_size)`
* 超长文本行在摘要中**截断**（例如保留前 280–320 字符，末尾加 `…`）。
* 若块为第一个（无上文），写明：`No previous context (this is the first chunk).`

#### 4.2 信息结构

* **Recent messages**：逐行列出
  `message_index | sender id | role | timestamp | text=<截断内容>`
* **Active case hints（可选但推荐）**：

  * 从 `Recent messages` 中基于启发式提取**未关闭**的话题线索：

    * 结束/关闭类关键词缺失（如 `resolved / issue resolved / anything else I can help / 谢谢 / 已解决 / 没问题了`）。
    * 存在**悬而未决的请求**（如“能否…？”、“请帮我…”、“请更新/退货/改地址”等）。
    * **关键实体**（订单号、买家名、商品/SKU）在尾部多次反复出现。
  * 每条 hint 包含：`case_hint`（一句话主题）、`evidence_message_index`（若干近邻行的 index）、`entities`（如 `order_ids`、`buyers`）。
  * 数量建议 ≤3 条，防止噪声。

> 注：实体可由正则与词典初筛，例如：
>
> * 订单号：包含 `-` 的长数字串（如 `9759-767261-6051`）或快递号模式；
> * 买家名：首字母大写的连续 token 或已知买家列表；
> * 物流/退款关键词：`refund, return, exchange, shipment, pickup, label, cancel, out-of-delivery` 等 + 你的中文关键词。

#### 4.3 呈现顺序与体量

* 先给 `Active case hints`（如有），再给 `Recent messages`；或合并为一个紧凑段落，**先结论后证据**。
* 上文摘要总长度建议 ≤ 1–1.5k 字符，避免挤占主块 token。

#### 4.4 上文摘要 Prompt 块（English，用于直接嵌入主 Prompt）

```
Previous context summary:

ACTIVE_CASE_HINTS:
- (optional, up to 3) Each hint summarizes an unresolved issue from the previous chunk, with entities and evidence message_index.
- If none: write "None".

[Example format]
- hint: "Buyer JenWM requested changing order 9759-767261-6051 to local pickup; resolution not confirmed."
  entities: {"order_ids": ["9759-767261-6051"], "buyers": ["JenWM"]}
  evidence_message_index: [188, 189, 193]

RECENT_MESSAGES:
- 187 | sender id=... | role=agent | 2025-03-12T08:31:02Z | text=...
- 188 | sender id=... | role=seller | 2025-03-12T08:32:10Z | text=...
- 189 | sender id=... | role=agent | 2025-03-12T08:33:41Z | text=...
- 190 | sender id=... | role=seller | 2025-03-12T08:34:55Z | text=...
- 191 | sender id=... | role=agent | 2025-03-12T08:35:09Z | text=...

(If this is the first chunk):
No previous context (this is the first chunk).
```

> 以上块直接放在主 Prompt 的“Previous context summary:”位置，无需额外解释文字。

### 5) Prompt 关键要求（含 Summary 内容约束）

* 新 case 触发：出现**新问题/新订单/新买家/不同主题**；回到**未解决的旧问题**应延续原 case。
* 模糊处理：不确定时**优先延续旧 case**，直至出现明确证据（新实体/明确新请求）。
* 输出严格 JSON（单块）：

  ```json
  {
    "complete_cases": [
      {
        "msg_list": [0,1,2,5],
        "summary": "Brief description of the issue, actions taken, and resolution status",
        "confidence": 0.9
      }
    ],
    "total_messages_analyzed": <total_number_of_messages>
  }
  ```

#### ✅ Summary 内容必须包含的关键信息（英文自然语句，1–3 句）

* **Order / Order IDs**（如有多个请列出主要的）
* **User / Buyer 标识**（如买家名或用户ID，能唯一指代即可）
* **Issue topic**（问题主题）与 **key actions taken**（已执行的关键动作）
* **Resolution status**：`open` / `ongoing` / `resolved` / `blocked`
* **Last action + timestamp**（可用块内最近的时间）
* **Pending party**（`seller` / `agent` / `buyer`）如适用

> 若缺少其中某项（例如没有订单号），在 summary 中明确写 `order: N/A` 或跳过并给出原因（如“no order referenced”）。

**Summary 英文模板示例（可内嵌在 Prompt 中）**

```
Format the `summary` as a compact English paragraph (1–3 sentences) that includes:
- order(s): <list or N/A>
- user/buyer: <identifier>
- topic: <main issue>
- actions: <key actions taken>
- status: <open|ongoing|resolved|blocked>
- last_update: <ISO timestamp or relative time>
- pending_party: <seller|agent|buyer|N/A>
```

### 6) 跨块合并（global\_case\_id 映射）

* 维护 `message_index -> global_case_id` 映射与 `next_case_id` 计数器。
* 对当前块返回的每个 case，基于 `msg_list` 与**历史已归档消息**的重叠来确定全局归属：

  1. **多数重叠原则**：统计与各全局 case 的重叠消息数，选择重叠最多者。
  2. **平局规则**：若并列，优先选择**最近活跃**（最新时间）的 case；若仍并列，取**更高平均置信度**者。
  3. **最小重叠阈值**：若最大重叠数 < 阈值（建议 2），且关键实体/订单不一致，则**新建 case**而非强行合并。
  4. **冲突解决机制**：若同一 case 与多个全局 case 都有显著重叠（>= 阈值），按以下优先级解决：
     - 首先选择**时间戳最近**的全局 case
     - 若时间接近（差距 < 10分钟），选择**置信度最高**者
     - 若仍并列，选择**全局 case_id 最小**者（先创建优先）
* **映射表更新规则**：
  - 采用**先占先得**原则：已分配给某全局 case 的 `message_index` 不可被后续处理覆盖
  - 若新 case 包含已分配的消息，则**拆分处理**：已分配消息归原 case，未分配消息可新建或合并到其他 case
* 合并后：为该 case 的**未分配** `message_index` 写入全局映射；聚合时**去重并排序** `msg_list`。
* 摘要与置信度聚合：`summary` 选**信息量最大的一条**（例如最长）；`confidence` 取**加权均值**（按包含消息数量加权，保留 3 位小数）。

### 7) 统计与输出

* `total_messages_analyzed`：**原始数据的总消息数**，即输入 CSV 文件的行数（不含表头）。注意：这不是各块处理的消息数之和，因为存在重叠；也不是去重后的数量，而是实际分析的原始消息总量。
* 产物：

  * `segmented.csv`：在原始数据基础上新增 **message\_index** 与 **case\_id**（全局 case）。
  * `cases.json`：`complete_cases`（含 `global_case_id`、`msg_list`（基于 `message_index`）、`summary`、`confidence`）+ `total_messages_analyzed`。

