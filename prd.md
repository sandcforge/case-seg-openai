# 客户支持消息案例分割系统 PRD

## 系统概览

本系统是一个基于LLM的客户支持消息案例分割和分析工具，能够将原始的支持对话消息自动分类为不同的案例(cases)，并为每个案例提供详细的跟踪和状态管理。

## 核心架构

### 数据流概览
```
CSV输入 → FileProcessor → 频道列表 → ChannelSegmenter → Chunks → 
LLM处理 → Case Segmentation → Merge Pipeline → 全局Cases → 独立文件输出
```

## 详细功能模块

### 1. 文件处理模块 (FileProcessor)

#### 1.1 输入处理
* 接收CSV格式的客户支持对话数据
* 默认输入文件：`assets/support_messages_andy.csv`
* 支持多频道(Channel URL)数据

#### 1.2 数据预处理
* **角色识别**：根据Sender ID自动分配role
  - `psops`开头 → `customer_service` 
  - 其他 → `user`
* **时间标准化**：将Created Time解析为UTC时区感知格式
* **数据排序**：按Channel URL分组，组内按Created Time和Message ID升序
* **索引生成**：为每个频道生成`msg_ch_idx` (0...N-1)

#### 1.3 频道分组输出
```python
[
    {
        "channel_url": "频道URL",
        "dataframe": pd.DataFrame  # 该频道的消息数据
    },
    ...
]
```

### 2. 频道分割模块 (ChannelSegmenter)

#### 2.1 设计原则
* **频道独立性**：每个频道完全独立处理，避免跨频道数据混合
* **单频道假设**：当前架构假设单频道输入，优化处理逻辑

#### 2.2 分块策略
* **半开区间**：使用`[start, end)`避免边界重复
* **重叠机制**：相邻块间保留overlap条消息维持上下文连续性
* **参数约束**：`overlap < chunk_size/3`确保消息最多出现在2个相邻块中

#### 2.3 分块计算公式
```
第1块：[0, chunk_size)
第2块：[chunk_size - overlap, 2 * chunk_size)
第i块：[(i-1) * chunk_size - overlap, i * chunk_size)
```

### 3. LLM集成模块

#### 3.1 多供应商支持
* **OpenAI**：支持GPT系列模型 (gpt-*)
* **Anthropic**：支持Claude系列模型 (claude-*)
* **API密钥管理**：基于模型前缀自动选择对应的API密钥

#### 3.2 结构化输出
* **Pydantic模型验证**：确保LLM输出符合预期格式
* **JSON Schema约束**：使用OpenAI的structured outputs确保格式正确性
* **容错机制**：正则表达式兜底解析异常格式

#### 3.3 调用类型
* **Case Segmentation**：分析当前chunk的案例分割
* **Tail Summary**：生成上下文摘要供下个chunk使用

### 4. 案例处理流程

#### 4.1 单Chunk处理
对于只有一个chunk的频道：
1. 直接调用`generate_case_segments()`
2. 无需merge pipeline
3. 返回完整的案例分割结果

#### 4.2 多Chunk处理 - 两阶段架构

##### 阶段1：LLM调用阶段
```python
for each chunk:
    case_results = chunk.generate_case_segments(llm_client)
    tail_summary = chunk.generate_tail_summary(llm_client)  # 除最后一个chunk
```

##### 阶段2：数据处理阶段
```python
result = execute_merge_pipeline(chunk_cases, tail_summaries, chunks)
```

### 5. Merge Pipeline (execute_merge_pipeline)

#### 5.1 设计特点
* **关注点分离**：纯数据处理，不依赖LLM
* **独立可测试**：可以用mock数据独立测试合并逻辑

#### 5.2 五阶段处理流程

##### Stage 2: Pairwise Merge
* 遍历相邻chunk对计算重叠区域
* 使用多维评分系统解决冲突：承接性 + 锚点强度 + 置信度 + 邻近度
* Union-Find数据结构追踪案例等价关系

##### Stage 3: Repair
* 基于proximity和confidence修复案例分配
* 处理未分配消息和多重分配冲突

##### Stage 4: Global Aggregation
* 使用Union-Find构建全局映射
* 将本地案例聚合为全局案例，去重并排序msg_list

##### Stage 5: Validation & Repair  
* **完整性验证**：检查消息覆盖率，要求达到100%
* **自动修复**：处理遗漏消息和重复分配
* **质量保障**：确保每条消息恰好属于一个案例

#### 5.3 冲突解决机制
* **锚点优先级**：tracking_id > order_id > buyer_handle > topic
* **评分系统**：综合考虑锚点强度、承接性、置信度和消息邻近度
* **三级修复策略**：邻近度分配 → 置信度选择 → 创建修复案例

### 6. 输出系统

#### 6.1 频道独立输出
每个频道产生独立的输出文件，避免跨频道数据冲突：

```
cases_channel_1.json      # 频道1的案例数据
segmented_channel_1.csv   # 频道1的带标注消息
cases_channel_2.json      # 频道2的案例数据  
segmented_channel_2.csv   # 频道2的带标注消息
...
```

#### 6.2 案例JSON格式
```json
{
    "channel_url": "频道URL",
    "global_cases": [
        {
            "global_case_id": 0,
            "msg_list": [0, 1, 2, 5],
            "summary": "Order 12345 shipping address change request...",
            "status": "ongoing",
            "pending_party": "seller", 
            "last_update": "2025-01-15T10:30:00Z",
            "confidence": 0.9,
            "anchors": {
                "tracking": ["ABC123"],
                "order": ["12345"],
                "buyer": ["john_doe"]
            }
        }
    ],
    "total_messages": 213,
    "chunks_processed": 11
}
```

#### 6.3 分割CSV格式
在原始数据基础上增加案例标注：
* **case_id**：该消息所属的案例ID
* **msg_ch_idx**：消息在频道内的索引 
* 保留所有原始字段

### 7. 测试模式

#### 7.1 案例分割测试 (--test-case-segment)
* 仅处理每个频道的第一个chunk
* 生成`test_case_segments.json`
* 用于验证LLM的案例分割能力

#### 7.2 摘要生成测试 (--test-tail-summary)  
* 仅处理每个频道的第一个chunk
* 生成`test_tail_summary.txt`
* 用于验证上下文摘要生成能力

### 8. 质量保障机制

#### 8.1 数据完整性
* **覆盖率验证**：确保每条消息被分配到且仅分配到一个案例
* **索引一致性**：频道内msg_ch_idx从0开始连续
* **边界处理**：半开区间避免消息重复计算

#### 8.2 错误恢复
* **自动修复**：智能处理遗漏和重复分配
* **降级策略**：LLM调用失败时的兜底处理
* **详细日志**：完整的debug输出追踪处理过程

#### 8.3 性能监控
* **LLM调用追踪**：记录每次API调用的耗时和token使用
* **处理统计**：显示每个阶段的处理结果和质量指标
* **内存管理**：及时清理不需要的数据结构

## 参数配置

### 命令行参数
```bash
python main.py [OPTIONS]

--input/-i              输入CSV文件路径 (默认: assets/support_messages_andy.csv)
--output-dir/-o         输出目录 (默认: out)  
--chunk-size/-c         分块大小 (默认: 80)
--overlap/-l            重叠大小 (默认: 20, 必须 < chunk_size/3)
--model/-m              LLM模型名称 (默认: gpt-5)
--test-case-segment     案例分割测试模式
--test-tail-summary     摘要生成测试模式
```

### 环境配置
* **ANTHROPIC_API_KEY**：Claude模型的API密钥
* **OPENAI_API_KEY**：GPT模型的API密钥
* **conda环境**：使用'dev'环境运行Python命令

## 技术特性

### 架构优势
* **模块化设计**：文件处理、分割、合并、输出各模块职责清晰
* **关注点分离**：LLM调用与数据处理逻辑分离
* **频道独立**：支持多频道数据的独立并行处理
* **可扩展性**：易于添加新的LLM供应商或处理策略

### 算法创新
* **多维评分**：综合多个因子的冲突解决机制
* **Union-Find优化**：高效的案例等价关系管理
* **智能修复**：基于邻近度和置信度的自动修复策略
* **结构化输出**：确保LLM输出格式的一致性和正确性

### 健壮性保障
* **100%覆盖率要求**：确保数据完整性
* **多层验证机制**：从LLM输出到最终结果的全链路验证
* **优雅降级**：各个环节的错误处理和恢复机制
* **详细诊断**：完整的日志和调试信息

## 当前状态

### 已实现功能
* ✅ 完整的文件处理和数据预处理流程
* ✅ 频道独立的分块和案例分割
* ✅ 多供应商LLM集成和结构化输出  
* ✅ 完整的merge pipeline和质量保障
* ✅ 频道独立的输出系统
* ✅ 测试模式和诊断工具

### 架构演进历程
1. **基础实现**：单频道处理，基本的LLM集成
2. **多频道支持**：频道分组处理，避免跨频道污染
3. **架构重构**：关注点分离，merge pipeline独立
4. **方法合并**：简化调用层级，统一处理入口
5. **质量优化**：完善验证机制，提升结果质量

### 性能特性
* **时间复杂度**：O(N)消息遍历 + O(Cases × Messages)验证修复
* **空间复杂度**：每个chunk的DataFrame slice + 案例结果存储
* **LLM调用**：每chunk需要1-2次调用（案例分割 + 可选摘要）
* **输出规模**：每个频道产生2个文件，总大小取决于案例数量和复杂度

系统现已达到生产就绪状态，能够处理复杂的多频道客户支持数据，生成高质量的案例分割结果。