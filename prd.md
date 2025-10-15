# 客户支持消息案例分割系统 PRD v3.2

## 系统概览

本系统是一个基于LLM的客户支持消息案例分割和分析工具，能够将原始的支持对话消息自动分类为不同的案例(cases)，并为每个案例提供详细的跟踪、状态管理和全面的统计分析。

## 核心架构

### 数据流概览 (v3.2 更新)
```
CSV输入 → Session协调器 → 文件预处理 → 频道分组 → Channel处理器 → 
案例分割 → 分类&性能计算 → 统计引擎 → 层次化报告输出 → 会话文件管理
```

### v3.2 架构优化重点
- **代码精简**：移除了大量未使用的逻辑和冗余方法
- **逻辑整合**：将分散的功能合并到更合适的类中
- **抽象层优化**：减少不必要的抽象，提高代码可读性
- **质量验证**：新增统计验证脚本确保数据准确性

### Session-based 架构设计
- **Session Class**: 统一的流程协调器，管理完整的处理生命周期
- **Channel Independence**: 频道级别的独立处理，支持并行操作
- **Statistics Engine**: 综合统计分析模块，支持跨频道数据聚合
- **Output Management**: 基于会话的文件组织和版本管理

## 详细功能模块

### 1. Session 协调器模块 (v3.1 新增)

#### 1.1 Session Class 架构
Session类作为系统的核心协调器，管理完整的处理流程：

```python
class Session:
    def run(self):
        # Stage 1: 跨频道文件数据处理
        self.process_file_data()
        # Stage 2: 会话文件夹结构创建
        self.create_session_folder()
        # Stage 3: 各频道独立处理
        self.process_channels()
        # Stage 4: 跨频道统计生成
        self.generate_statistics()
```

#### 1.2 文件数据处理 (process_file_data)
**统一的跨频道数据预处理逻辑**：

* **输入验证**：接收CSV格式的客户支持对话数据
  - 默认输入：`assets/support_messages_andy.csv`
  - 支持多频道(Channel URL)数据

* **数据清洗**：
  - 过滤删除标记的记录 (`Deleted != True`)
  - **角色识别**：根据Sender ID模式自动分配role
    - `psops`开头 → `customer_service`
    - 其他 → `user`

* **时间处理**：
  - **时区标准化**：Created Time解析为UTC时区感知格式
  - **排序规则**：按Channel URL分组，组内按Created Time + Message ID升序

* **索引生成**：
  - 为每个频道生成连续的`msg_ch_idx` (0...N-1)
  - 保证频道内消息索引的一致性和连续性

#### 1.3 频道数据分组
预处理后生成频道数据列表：
```python
channel_data_list = [
    {
        "channel_url": "频道URL",
        "dataframe": pd.DataFrame  # 该频道的清洗后消息数据
    },
    ...
]
```

#### 1.4 会话管理
* **会话命名**：自动生成时间戳会话名或用户指定
* **输出组织**：`out/session_{session_name}/` 目录结构
* **版本控制**：基于会话的输出版本管理

### 2. Channel 处理器模块 (v3.2 架构精简)

#### 2.1 架构设计原则
* **频道独立性**：每个频道完全独立处理，避免跨频道数据混合
* **并行支持**：支持多频道并行处理（当前为顺序执行）
* **状态一致性**：每个Channel实例维护独立的处理状态

#### 2.2 v3.2 重构优化
**已移除的功能模块**：
- `segment_all_chunks_with_review()` - 复杂审查流程已简化
- `execute_case_review()` - 案例审查功能已整合
- `execute_merge_pipeline()` - 合并管道已优化
- `classify_all_cases()` - 分类逻辑已内联到主流程

**保留的核心处理模式**：
**统一分割模式 (segment_all_chunks)**：
- 简化的单阶段处理，无需复杂的merge pipeline
- 直接对所有chunks进行LLM分割
- 集成的消息格式化和案例修复逻辑

#### 2.3 分块策略
* **半开区间**：使用`[start, end)`避免边界重复
* **重叠机制**：相邻块间保留overlap条消息维持上下文连续性
* **参数约束**：`overlap < chunk_size/4`（已更新约束条件）

#### 2.4 分块计算逻辑
```python
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_messages)
    chunk_messages = df_clean.iloc[start_idx:end_idx]
```

#### 2.5 案例分割与修复 (v3.2 简化)
* **直接LLM分割**：Channel类直接处理案例分割，无需Chunk抽象层
* **内置消息格式化**：`_format_messages_for_prompt()` 统一处理消息格式
* **统一修复**：`repair_case_segment_output()` 处理重复和遗漏
* **质量保证**：确保每条消息恰好属于一个案例

### 3. LLM集成模块

#### 3.1 多供应商支持
* **OpenAI**：支持GPT系列模型 (gpt-*, default: gpt-5)
* **Anthropic**：支持Claude系列模型 (claude-*)
* **API密钥管理**：基于模型前缀自动选择对应的环境变量
  - `gpt-*` → `OPENAI_API_KEY`
  - `claude-*` → `ANTHROPIC_API_KEY`

#### 3.2 结构化输出与调试
* **Pydantic模型验证**：确保LLM输出符合预期格式
* **JSON Schema约束**：使用OpenAI的structured outputs确保格式正确性
* **容错机制**：正则表达式兜底解析异常格式
* **调试日志**：每次LLM调用自动生成详细的debug日志文件
  - 请求内容、响应结果、耗时统计
  - 存储于 `debug_output/` 目录

#### 3.3 调用类型
* **Case Segmentation**：分析当前chunk的案例分割
* **Case Classification**：为每个案例生成主分类和子分类
* **Tail Summary**：生成上下文摘要供下个chunk使用(传统模式)

### 4. 案例管理与分类模块

#### 4.1 Case 对象模型
Case类包含完整的案例信息：

```python
@dataclass
class Case:
    # 基本信息
    case_id: str
    message_id_list: List[int]
    messages: pd.DataFrame
    summary: str
    status: str  # ongoing|resolved|blocked
    
    # 分类信息
    main_category: str
    sub_category: str
    classification_reasoning: str
    classification_confidence: float
    
    # 性能指标
    handle_time: int  # 处理时长(分钟)
    first_res_time: int  # 首次响应时长(分钟)
    usr_msg_num: int  # 用户消息数量
    first_contact_resolution: int  # 首次联系解决(1/0/-1)
```

#### 4.2 分类系统
* **自动分类**：LLM驱动的智能分类引擎
* **主分类**：Payment, Shipment, Order, Product, Technical, Account等
* **子分类**：每个主分类下的具体分类，如Refund_Request, Tracking等
* **置信度评估**：为每个分类决策提供置信度分数

#### 4.3 性能指标计算 (calculate_metrics)
**handle_time计算**：
```python
first_time = df_sorted.iloc[0]['Created Time']
last_time = df_sorted.iloc[-1]['Created Time']
handle_time = int((last_time - first_time).total_seconds() / 60)
```

**first_res_time计算逻辑**：
- 如果支持方发起对话：保持-1
- 如果用户发起：计算到首次支持响应的时间
- 无支持响应：使用handle_time

**first_contact_resolution计算**：
- `status == "resolved"` 且 `handle_time <= 480分钟` → 1
- 已处理但未在8小时内解决 → 0
- 未处理 → -1

### 5. 统计分析引擎 (v3.1 核心功能)

#### 5.1 统计架构设计
Session.generate_statistics() 协调跨频道统计分析：

```python
def generate_statistics(self):
    # 收集所有频道的案例
    all_cases = []
    for channel in self.channels:
        all_cases.extend(channel.cases)
    
    # 计算综合统计
    stats_result = self._calculate_comprehensive_stats(all_cases)
    
    # 生成报告和文件
    self._print_summary_report(stats_result)
    self._save_stats_to_file(stats_result)
```

#### 5.2 统计模块组成

**5.2.1 分类统计 (_calculate_category_stats)**
* **主分类分布**：统计各主分类的案例数量和百分比
* **子分类分布**：统计各子分类的案例数量和百分比  
* **层次化映射**：构建主分类→子分类的层次关系图
  ```python
  main_to_sub_mapping = {
      "Payment": {"Refund_Request": 580, "Payment_Failed": 152, ...},
      "Shipment": {"Tracking": 240, "Delivery_Issue": 305, ...},
      ...
  }
  ```

**5.2.2 性能指标统计 (_calculate_metrics_stats)**
针对三个核心指标进行全面分析：

* **handle_time**: 案例处理时长（分钟）
* **first_res_time**: 首次响应时长（分钟）
* **usr_msg_num**: 用户消息数量

**每个指标计算内容**：
```python
{
    "valid_cases": len(valid_values),  # 有效案例数
    "validity_rate": percentage,       # 有效性比率
    "percentiles": {
        "P5": percentile_5,
        "P50": percentile_50,          # 中位数
        "P95": percentile_95
    },
    "basic_stats": {
        "min": minimum_value,
        "max": maximum_value,
        "mean": average_value,
        "std": standard_deviation
    }
}
```

**5.2.3 首次联系解决统计 (_calculate_fcr_stats)**
* **解决率计算**：resolved案例在有效案例中的比例
* **有效性分析**：排除-1值的数据质量分析
* **解决/未解决案例分布**

#### 5.3 层次化报告输出

**5.3.1 控制台报告格式**
```
📊 CASE STATISTICS SUMMARY
==================================================
Total Cases Analyzed: 5303

📈 CATEGORY DISTRIBUTION
------------------------------
Main Categories:
  Technical: 1235 cases (23.3%)
  Payment: 1147 cases (21.6%)
  ...

Hierarchical Category Breakdown:
  Technical: 1235 cases (23.3%)
    ├─ System_Error: 334 cases (6.3%)
    ├─ App_Issue: 340 cases (6.4%)
    └─ Website_Issue: 184 cases (3.5%)
  Payment: 1147 cases (21.6%)
    ├─ Refund_Request: 580 cases (10.9%)
    └─ Payment_Failed: 152 cases (2.9%)
  ...

⏱️ PERFORMANCE METRICS
------------------------------
Metric Definitions:
  • handle_time: Time between first and last message in minutes
  • first_res_time: Support response time in minutes
  • usr_msg_num: Count of user messages
  (Value of -1 indicates not processed/invalid data)

Handle Time:
  Valid Cases: 4523/5303 (85.3%)
  P5: 1.2, P50: 45.7, P95: 287.5
  Min: 0.0, Max: 1440.0, Mean: 73.2, Std: 89.4
```

**5.3.2 JSON统计文件**
完整的统计结果保存为 `statistics_{session_name}.json`，包含：
* 汇总信息（总案例数、分析时间戳）
* 分类分布（包含层次化映射）
* 性能指标详情
* FCR分析结果

### 5. 代码结构优化 (v3.2 重构总结)

#### 5.1 已移除的模块和文件
**完全移除的文件**：
- `file_processor.py` - 功能已整合到Session.process_file_data()
- `chunk.py` - 抽象层已简化，核心逻辑迁移到Channel类

**已清理的功能**：
- Utils类的冗余方法(`format_dataframe_for_prompt`, `format_one_msg_for_prompt`部分重构)
- Channel类的未使用方法(多个复杂的pipeline方法)
- 多余的命令行参数(`--test-case-segment`)

#### 5.2 功能整合优化
**消息格式化整合**：
- 将分散的格式化逻辑统一到Channel._format_messages_for_prompt()
- 恢复Utils.format_one_msg_for_prompt()用于案例分类
- 消除重复的格式化代码

**处理流程简化**：
- 直接的案例分割流程，无需复杂的中间抽象层
- 减少方法调用链长度，提高代码可读性
- 保持核心功能不变的前提下精简代码结构

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

### 7. 质量验证系统 (v3.2 新增)

#### 7.1 统计验证脚本
**位置**: `unit_test/test_session_statistics.py`

**功能特性**：
* 读取会话输出目录中的所有案例JSON文件
* 重新计算所有性能指标(handle_time, first_res_time, usr_msg_num)
* 与官方statistics_.json文件进行精确比较
* 提供详细的角落案例分析

**使用方法**：
```bash
python3 unit_test/test_session_statistics.py <session_directory>
```

**验证内容**：
- 百分位数计算 (P5, P50, P95)
- 基础统计值 (min, max, mean, std)
- 首次联系解决率
- 数据有效性比率

#### 7.2 角落案例检测
**Handle Time = 0 案例**：
- 单消息案例，无用户交互
- 系统通知或平台公告类消息

**First Response Time = 0 案例**：
- 平台即时响应的案例
- 通常为简单确认或状态更新

**First Response Time = -1 案例**：
- 无有效首次响应时间的案例
- 可能由于缺少用户消息或计算逻辑边界条件

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
--enable-review         启用LLM案例审查模式  
--enable-classification 启用案例分类功能 (默认: True)
--session/-s           会话名称，用于输出组织
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

### 6. 输出与文件管理

#### 6.1 Session-based 文件组织 (v3.1)
```
out/
├── session_{timestamp}/
│   ├── cases_channel_1.json      # 频道1案例数据
│   ├── segmented_channel_1.csv   # 频道1标注消息
│   ├── cases_channel_2.json      # 频道2案例数据
│   ├── segmented_channel_2.csv   # 频道2标注消息
│   └── statistics_{session}.json # 跨频道统计分析
└── debug_output/
    └── {call_label}_{timestamp}.log  # LLM调用日志
```

#### 6.2 JSON案例格式
```json
{
    "channel_url": "频道URL",
    "global_cases": [
        {
            "case_id": "0",
            "message_id_list": [0, 1, 2, 5],
            "summary": "Order 12345 shipping address change request...",
            "status": "resolved",
            "main_category": "Order",
            "sub_category": "Modification",
            "handle_time": 45,
            "first_res_time": 12,
            "usr_msg_num": 3,
            "first_contact_resolution": 1
        }
    ],
    "total_messages": 213,
    "processing_metadata": {...}
}
```

## v3.2 版本更新总结

### 🆕 核心优化内容

**1. 代码结构精简**
* 移除了2个完整模块文件(file_processor.py, chunk.py)
* 清理了Channel类中7个未使用的方法
* 简化了Utils类，移除冗余格式化方法
* 优化命令行参数，移除测试相关的过时选项

**2. 逻辑整合优化**
* 将FileProcessor功能完全整合到Session类
* 消息格式化逻辑集中到Channel._format_messages_for_prompt()
* 案例分割直接在Channel类中处理，无需Chunk抽象层
* 减少方法调用层次，提高代码可读性

**3. 质量验证增强**
* 新增统计验证脚本 `unit_test/test_session_statistics.py`
* 自动验证性能指标计算的准确性
* 提供详细的角落案例分析和频道信息
* 支持会话级别的统计数据完整性检查

**4. Bug修复**
* 修复了Utils.format_one_msg_for_prompt缺失导致的分类错误
* 保持了所有核心功能的完整性和准确性
* 改善了错误处理和调试信息

### 📈 系统成熟度

**架构演进历程**：
1. **v1.0**: 基础实现 - 单频道处理，基本LLM集成
2. **v2.0**: 多频道支持 - 频道分组，避免跨频道污染  
3. **v3.0**: 架构重构 - Session协调器，关注点分离
4. **v3.1**: 统计增强 - 综合分析引擎，层次化报告
5. **v3.2**: 代码优化 - 精简结构，质量验证，维护性提升

**当前状态**：
* ✅ Session-based统一架构
* ✅ 综合统计分析引擎  
* ✅ 层次化分类报告系统
* ✅ 自动化案例分类和性能计算
* ✅ 多供应商LLM集成和调试
* ✅ 完整的质量保障和验证机制
* ✅ **精简的代码结构和更高的维护性**
* ✅ **自动化质量验证工具**

**代码质量特征**：
* **可维护性**: 移除冗余代码，简化抽象层次
* **可验证性**: 内置统计验证工具确保数据准确性
* **健壮性**: 保持核心功能完整性，优化错误处理
* **扩展性**: 精简后的架构更易于功能扩展和修改

**性能特征**：
* **处理能力**: 支持数千案例的批量分析
* **统计精度**: 自动过滤无效数据，提供准确的分析结果  
* **报告质量**: 层次化显示，直观易读的统计报告
* **验证能力**: 自动化验证确保统计计算的正确性

系统现已达到生产就绪状态，具备完整的数据处理、智能分析、报告生成和质量验证能力，代码结构更加精简和易维护，可支持复杂的客户支持运营分析需求。