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

3. **每个Chunk的Case Segmentation流程**

### 3.1 对每个Chunk执行Case Segmentation

#### 3.1.1 Case Segmentation原则

* **决策标准**：
  * **Continue vs. New Case**：继续现有case如果主题和锚点（tracking/order/buyer/topic）匹配PREVIOUS CONTEXT中的未解决case；如果是新订单/跟踪/买家/主题且与活跃case无关联，则启动新case。
  * **锚点优先级**：`tracking_id > order_id > buyer_handle > topic`
  * **多订单包裹**：共享同一tracking_id的多个订单→一个case，除非明确独立。
  * **模糊处理**：不确定时，倾向于延续已有case，直至有新case的强证据。
  * **唯一性**：每个`msg_ch_idx`属于且仅属于一个case。

#### 3.1.2 Prompt策略

* **第一个chunk**：
  * `previous_tail_summary = None`
  * context_block = "No previous context"
  * 直接对chunk消息进行case segmentation

* **第二个及后续chunk**：
  * `previous_tail_summary` = 前一个chunk的tail summary
  * context_block = 完整的ACTIVE_CASE_HINTS + RECENT_MESSAGES + META结构
  * 基于上文继续或新建cases

* **消息格式化**：`msg_ch_idx | sender_id | role | timestamp | text`

#### 3.1.3 Validation和Auto-Fix Policy

##### Coverage检查（100%要求）
* 验证所有`msg_ch_idx`（0到total_messages-1）都被分配到cases中
* 计算覆盖率：`len(assigned_messages) / total_messages * 100`
* 要求达到100%覆盖率，否则触发auto-fix

##### Missing Message处理（基于邻近度的分配）
* **算法**：`_find_closest_case(missing_msg, complete_cases)`
* **策略**：计算missing_msg与每个case中消息的最小距离
* **优先级**：距离最近 > 案例规模较小（平局时）
* **操作**：将missing_msg添加到最佳case的msg_list中并排序

##### Multi-assignment处理（基于置信度的解析）
* **算法**：`_select_best_case_for_message(case_list, complete_cases)`
* **策略**：选择confidence最高的case保留该消息
* **优先级**：置信度最高 > 第一个case（平局时）
* **操作**：从其他cases中移除该消息，仅保留在最佳case中

##### 实时Action Logging
* 记录每个修复动作：添加missing message、移除duplicate assignment
* 显示case summary预览和置信度信息
* 提供修复前后的统计对比

### 3.2 生成Tail Summary

#### 3.2.1 Tail Summary的目的和结构

* **目的**：为下一个chunk提供结构化的上文信息，确保case连续性
* **输入**：当前chunk的case segmentation结果 + 当前消息 + overlap参数

* **结构**：
  ```
  ACTIVE_CASE_HINTS:
  - topic: "简短标题"
    status: "open|ongoing|blocked"
    evidence_msg_ch_idx: [消息索引列表]

  RECENT_MESSAGES:
  - msg_ch_idx | sender id=sender_id | role=role | timestamp | text=截断文本

  META (optional):
  - overlap: 数值
  - channel: 完整channel_url
  - time_window: ["开始时间", "结束时间"]
  ```

* **Active Case提取**：从complete_cases中提取未解决的cases（status为open/ongoing/blocked）
* **Recent Messages**：取当前chunk尾部最多overlap条消息，文本截断150字符
* **时间窗口**：当前chunk的开始和结束时间戳

4. **LLM集成与调用**

### 4.1 LLM API配置

* **API提供商**：Anthropic Claude API
* **默认模型**：`claude-3-5-sonnet-20241022`（可通过`--model/-m`参数指定）
* **API密钥**：通过环境变量`ANTHROPIC_API_KEY`或`--api-key`参数提供
* **最大tokens**：默认4000（case segmentation和tail summary调用）

### 4.2 调用方式

* **Case Segmentation**：`llm_client.generate(final_prompt, call_label="case_segmentation")`
* **Tail Summary**：`llm_client.generate(final_prompt, call_label="tail_summary")`
* **调用标签**：用于debug日志文件命名和追踪

### 4.3 Debug日志系统

* **日志目录**：`debug_output/`（自动创建）
* **文件命名**：`{call_label}_{timestamp}.log`
* **日志内容**：
  * 请求元数据（时间戳、call_label、模型、max_tokens、prompt长度）
  * 完整prompt内容
  * 完整response内容或错误信息
  * 成功/失败状态

### 4.4 JSON解析

* **正常路径**：直接 `json.loads(response)`
* **容错路径**：使用正则表达式 `re.search(r'\{.*\}', response, re.DOTALL)` 提取JSON
* **失败处理**：抛出RuntimeError并记录到debug日志

5. **JSON输出格式与Hard Constraints**

### 5.1 完整JSON输出结构

每个chunk的case segmentation返回严格的JSON格式：

```json
{
  "complete_cases": [
    {
      "msg_list": [0,1,2,5],
      "summary": "Brief description of the issue, actions taken, and resolution status. Include: orders, buyer, topic, key actions, status, last_update (ISO), pending_party.",
      "status": "open | ongoing | resolved | blocked",
      "pending_party": "seller | platform | N/A", 
      "last_update": "YYYY-MM-DDTHH:MM:SSZ or N/A",
      "is_active_case": true,
      "confidence": 0.9
    }
  ],
  "total_messages_analyzed": <int>
}
```

### 5.2 字段说明

* **msg_list**：该case包含的消息索引列表（基于msg_ch_idx，升序排列，无重复）
* **summary**：1-3句英文描述，必须包含：orders、buyer、topic、key actions、status、last_update、pending_party
* **status**：案例状态（open=新开启、ongoing=进行中、resolved=已解决、blocked=阻塞）
* **pending_party**：待处理方（seller=卖家、platform=平台、N/A=无需等待）
* **last_update**：最后更新时间（ISO格式或N/A）
* **is_active_case**：是否为活跃案例（status为open/ongoing/blocked时为true）
* **confidence**：置信度（0-1之间的浮点数）
* **total_messages_analyzed**：当前chunk分析的消息总数

### 5.3 Hard Constraints系统

#### Coverage & Uniqueness Check硬约束
* 每个`msg_ch_idx`必须被分配到至多一个case中
* 如果消息涉及多个实体，使用锚点优先级选择一个case
* 最终JSON必须达到0重复、0遗漏的msg_ch_idx分配

#### Report & Fix Loop机制  
* 在推理过程中检测到重复或未分配消息时，必须在输出JSON前修复
* 最终JSON必须通过100%覆盖验证
* 失败时触发pipeline中断和错误处理

9. **当前实现状态**

### 9.1 已实现功能（Stage 1-2, 4-5）

* **Stage 1-2**：完整的文件处理和分块系统
* **Stage 4**：第一个chunk的case segmentation，包含validation和auto-fix
* **Stage 5**：第一个chunk的tail summary generation

### 9.2 输出文件

* `out/[source_filename]_out.csv`：预处理后的消息数据
* `out/first_chunk_cases.json`：第一个chunk的case segmentation结果  
* `out/first_chunk_tail_summary.txt`：第一个chunk的tail summary
* `debug_output/{call_label}_{timestamp}.log`：LLM调用的debug日志

### 9.3 待实现功能（多chunk处理）

* **跨chunk的case mapping**：`message_index -> global_case_id`映射表
* **全局case聚合**：合并跨chunk的相同cases
* **完整pipeline**：处理所有chunks并生成最终输出
  * `out/segmented.csv`：带case_id标注的完整数据
  * `out/cases.json`：全局聚合的cases列表

10. **参数与默认值**

* `--chunk-size/-c` = 80（按行数分块；若需按 token 可替换为 token 估算）
* `--overlap/-l` = 20（跨块上下文粘连，必须 < chunk_size/3）
* `--input/-i` = `assets/support_messages_andy.csv`（输入 CSV 文件）
* `--output-dir/-o` = `out`（输出目录）
* `--model/-m` = `"claude-3-5-sonnet-20241022"`（Claude模型名称）
* `--api-key`：Anthropic API密钥（可选，默认使用环境变量ANTHROPIC_API_KEY）

11. **健壮性与可扩展点**

* **JSON 容错**：提供了正则表达式兜底解析 `re.search(r'\{.*\}', response, re.DOTALL)`
* **Validation与Auto-Fix**：100%覆盖率保障，自动修复missing和duplicate assignments
* **Debug日志系统**：完整的LLM调用日志，包含请求/响应内容和错误信息  
* **Policy-based修复**：
  * Missing messages：基于邻近度的智能分配
  * Multi-assignments：基于置信度的优先级选择
  * 实时Action logging显示所有修复动作
* **Hard Constraints执行**：
  * Prompt内置Coverage & Uniqueness Check要求
  * Report & Fix Loop机制确保JSON输出质量
  * 失败时pipeline中断，避免传播错误结果
* **可扩展的分块策略**：当前按行数分块，可扩展为按token数分块

12. **复杂度与性能**

* **时间复杂度**：线性遍历消息（`O(N)`），validation和auto-fix为`O(Cases × Messages)`
* **空间复杂度**：每个chunk的DataFrame slice + case结果存储
* **LLM调用成本**：每chunk需要2次调用（case segmentation + tail summary）
* **Debug开销**：每次LLM调用生成完整日志文件，文件大小取决于prompt和response长度
* **优化策略**：
  * 重叠机制减少误切断裂
  * Hard constraints减少重复修正成本  
  * Policy-based auto-fix避免人工干预

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

