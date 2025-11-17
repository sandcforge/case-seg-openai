# Cloud Run Jobs 部署指南

本项目配置为 Cloud Run Jobs，每天自动运行一次处理客服消息。

## 项目结构

```
case_seg_openai/
├── src/                      # 所有应用代码
│   ├── __init__.py
│   ├── app_bigquery.py      # Cloud Run 入口
│   ├── main.py              # CLI 入口
│   ├── case.py              # Case 数据模型
│   ├── channel.py           # Channel 处理逻辑
│   ├── llm_client.py        # LLM 客户端
│   ├── session.py           # Session 管理
│   ├── utils.py             # 工具函数
│   ├── vision_processor.py  # 图像处理
│   └── prompts/             # LLM prompts
│       ├── case_classification_prompt.md
│       ├── case_review_prompt.md
│       ├── segmentation_prompt.md
│       ├── tail_summary_prompt.md
│       └── vision_analysis_prompt.md
├── requirements.txt         # Python 依赖
├── Dockerfile              # 容器配置
├── .dockerignore           # Docker 忽略文件
├── deploy.sh               # 部署脚本
├── setup_scheduler.sh      # 定时任务配置
└── DEPLOY.md              # 本文档
```

## 架构说明

- **Cloud Run Jobs**: 运行 `src/app_bigquery.py` 进行消息处理
- **Cloud Scheduler**: 每天凌晨 2 点触发任务
- **串行执行**: parallelism=1 确保同一时间只有一个任务运行
- **跨平台构建**: Mac ARM64 → Linux AMD64
- **按需计费**: 只在任务运行时收费

## 前置条件

### 1. 安装必需工具

```bash
# Google Cloud SDK
brew install google-cloud-sdk

# Docker (支持多平台构建)
brew install --cask docker
```

### 2. 准备环境变量

从 `.env` 文件导出环境变量：

```bash
# 必需变量
export OPENAI_API_KEY="your-openai-api-key"
export BIGQUERY_CREDENTIALS_JSON='{"type":"service_account",...}'

# 可选变量
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export FIREBASE_API_KEY="your-firebase-key"
export FIREBASE_EMAIL="your-email"
export FIREBASE_PASSWORD="your-password"
```

### 3. 注意事项

- **跨平台构建**: 本地 Mac (ARM64) 会构建 Linux AMD64 镜像以兼容 Cloud Run
- Docker 使用 QEMU 进行跨平台构建，首次构建可能较慢
- 后续构建会利用缓存加速

## 部署步骤

### 1. 认证 Google Cloud

```bash
gcloud auth login
gcloud config set project plantstory

# 配置 Artifact Registry 认证（推荐使用）
gcloud auth configure-docker us-docker.pkg.dev
```

### 2. 部署 Cloud Run Job

```bash
./deploy.sh
```

这个脚本会：
1. 配置 Artifact Registry 认证
2. 创建 Docker repository（如果不存在）
3. 构建 Docker 镜像（linux/amd64 平台）
4. 推送到 Artifact Registry
5. 部署为 Cloud Run Job
6. 配置资源限制和环境变量

**配置说明：**
- 内存: 4Gi
- CPU: 2 核
- 超时: 72000 秒 (20 小时)
- Parallelism: 1 (串行执行)
- Tasks: 1

### 3. 配置每日定时任务

```bash
./setup_scheduler.sh
```

这个脚本会创建 Cloud Scheduler：
- **调度时间**: 每天凌晨 2:00 AM (America/Los_Angeles 时区)
- **Cron 表达式**: `0 2 * * *`

修改调度时间：编辑 `setup_scheduler.sh` 中的 `SCHEDULE` 变量

```bash
SCHEDULE="0 2 * * *"      # 每天凌晨 2 点
SCHEDULE="0 */6 * * *"    # 每 6 小时
SCHEDULE="0 0 * * 0"      # 每周日凌晨
```

## 本地测试

### 方式 1: 使用模块运行（推荐）

```bash
# 从项目根目录运行
python -m src.app_bigquery --dry-run

# 查看帮助
python -m src.app_bigquery --help
```

### 方式 2: 进入 src 目录运行

```bash
cd src
python app_bigquery.py --dry-run
```

### 方式 3: 使用 CLI 工具

```bash
python -m src.main --input assets/support_messages.csv --chunk-size 80
```

### 常用参数

```bash
# Dry-run 模式（保存 CSV/JSON，不写入 BigQuery）
python -m src.app_bigquery --dry-run

# 自定义 chunk size
python -m src.app_bigquery --chunk-size 100

# 自定义模型
python -m src.app_bigquery --model gpt-4

# 处理指定 channel
python -m src.app_bigquery --channel-urls "https://example.com/channel1"

# 禁用 vision processing
python -m src.app_bigquery --no-enable-vision-processing
```

## Cloud Run 管理

### 手动触发 Job

```bash
gcloud run jobs execute customer-service-case-analysis --region us-central1 --project plantstory
```

### 查看运行日志

```bash
# 最近 50 条日志
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=customer-service-case-analysis" \
  --limit 50 --project plantstory

# 查看详细日志
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=customer-service-case-analysis" \
  --format="table(timestamp,severity,textPayload)" \
  --limit=100
```

### 查看执行历史

```bash
gcloud run jobs executions list \
  --job=customer-service-case-analysis \
  --region=us-central1 \
  --project plantstory
```

### 查看 Job 配置

```bash
gcloud run jobs describe customer-service-case-analysis \
  --region us-central1 \
  --project plantstory
```

## Cloud Scheduler 管理

### 手动触发 Scheduler

```bash
gcloud scheduler jobs run customer-service-case-analysis-daily \
  --location=us-central1 \
  --project plantstory
```

### 查看 Scheduler 状态

```bash
gcloud scheduler jobs describe customer-service-case-analysis-daily \
  --location=us-central1 \
  --project plantstory
```

### 暂停定时任务

```bash
gcloud scheduler jobs pause customer-service-case-analysis-daily \
  --location=us-central1 \
  --project plantstory
```

### 恢复定时任务

```bash
gcloud scheduler jobs resume customer-service-case-analysis-daily \
  --location=us-central1 \
  --project plantstory
```

### 查看执行历史

```bash
gcloud scheduler jobs list --location=us-central1 --project plantstory
```

## 更新部署

### 代码更新后重新部署

```bash
# 只需运行 deploy.sh，无需重新配置 Scheduler
./deploy.sh
```

### 修改调度时间

```bash
# 编辑 setup_scheduler.sh 中的 SCHEDULE 变量
# 然后重新运行
./setup_scheduler.sh
```

### 修改环境变量

```bash
# 更新环境变量后重新部署
export OPENAI_API_KEY="new-key"
./deploy.sh
```

## 故障排查

### 1. 导入错误 (ImportError)

**问题**: `ImportError: attempted relative import with no known parent package`

**解决**: 使用模块方式运行
```bash
python -m src.app_bigquery  # ✅ 正确
python src/app_bigquery.py  # ❌ 可能失败
```

### 2. Prompts 文件未找到

**问题**: `FileNotFoundError: prompts/segmentation_prompt.md`

**解决**: 已修复，`llm_client.py` 使用动态路径解析

### 3. 跨平台构建失败

**问题**: Docker 构建失败或镜像无法运行

**解决**:
- 确保使用 `--platform linux/amd64`
- 检查 Docker Desktop 是否启用多平台支持

### 4. BigQuery 权限错误

**问题**: `Permission denied` 访问 BigQuery

**解决**:
- 检查 `BIGQUERY_CREDENTIALS_JSON` 是否正确
- 确保服务账号有 BigQuery 读写权限

### 5. 超时错误

**问题**: Job 运行超过 20 小时

**解决**: 增加 `deploy.sh` 中的 `--task-timeout` 值（当前已设为 20 小时）
```bash
--task-timeout 86400  # 24 小时（最大值）
```

### 6. 内存不足

**问题**: `OOMKilled` 错误

**解决**: 增加内存配置
```bash
--memory 8Gi  # 改为 8GB
```

## 成本优化

### 当前配置

- **内存**: 4GB
- **CPU**: 2 核
- **超时**: 72000 秒 (20 小时)
- **频率**: 每天 1 次
- **并发**: 1 (串行执行)

### 预估成本

Cloud Run Jobs 计费：
- 只在任务运行时计费
- 基于 CPU、内存和执行时间
- 每月免费额度: 180,000 vCPU-秒, 360,000 GiB-秒

**示例计算**（假设每次运行 10 分钟）：
- 每天运行 1 次 × 30 天 = 30 次/月
- 每次 10 分钟 = 300 次/月
- 2 vCPU × 600 秒 = 1,200 vCPU-秒
- 4 GiB × 600 秒 = 2,400 GiB-秒
- 月总计: 36,000 vCPU-秒, 72,000 GiB-秒（在免费额度内）

## 安全最佳实践

1. **不要将 .env 文件提交到 Git**
   - 已在 `.gitignore` 中配置

2. **使用 Secret Manager（推荐）**
   ```bash
   # 创建 secret
   echo -n "your-api-key" | gcloud secrets create openai-api-key --data-file=-

   # 在 deploy.sh 中引用
   --set-secrets="OPENAI_API_KEY=openai-api-key:latest"
   ```

3. **最小权限原则**
   - 为服务账号只授予必需的权限
   - BigQuery: `roles/bigquery.dataEditor`
   - Storage: `roles/storage.objectViewer`

4. **定期轮换 API Keys**

## 监控和告警

### 设置日志告警

```bash
# 创建告警策略（错误超过阈值）
gcloud logging metrics create case_processor_errors \
  --description="Case processor errors" \
  --log-filter='resource.type="cloud_run_job"
    resource.labels.job_name="customer-service-case-analysis"
    severity>=ERROR'
```

### 查看指标

```bash
# Cloud Console 查看
https://console.cloud.google.com/run/jobs/details/us-central1/customer-service-case-analysis
```

## 清理资源

### 删除 Cloud Run Job

```bash
gcloud run jobs delete customer-service-case-analysis \
  --region us-central1 \
  --project plantstory
```

### 删除 Cloud Scheduler

```bash
gcloud scheduler jobs delete customer-service-case-analysis-daily \
  --location=us-central1 \
  --project plantstory
```

### 删除 Docker 镜像

```bash
# Artifact Registry (新方式)
gcloud artifacts docker images delete \
  us-docker.pkg.dev/plantstory/cloud-run-apps/customer-service-case-analysis \
  --project plantstory

# 或删除整个 repository
gcloud artifacts repositories delete cloud-run-apps \
  --location=us \
  --project plantstory
```

## 常见问题

**Q: 为什么使用 Cloud Run Jobs 而不是 Cloud Run Service?**

A: Jobs 专为批处理任务设计，任务完成后自动停止，不产生额外费用。Service 需要保持运行状态。

**Q: 可以手动触发多次吗？**

A: 可以，但由于 parallelism=1，新触发会排队等待当前任务完成。

**Q: 如何更改时区？**

A: 编辑 `setup_scheduler.sh` 中的 `TIMEZONE` 变量，然后重新运行脚本。

**Q: 支持哪些 LLM 模型？**

A:
- OpenAI: gpt-4, gpt-5, gpt-4-turbo 等
- Anthropic: claude-3-5-sonnet, claude-3-opus 等
- 通过 `--model` 参数指定

**Q: 如何查看 debug 日志？**

A: 所有 LLM 调用的详细日志保存在容器的 `debug_output/` 目录，可通过 Cloud Logging 查看。

## 参考资料

- [Cloud Run Jobs 文档](https://cloud.google.com/run/docs/create-jobs)
- [Cloud Scheduler 文档](https://cloud.google.com/scheduler/docs)
- [Docker 多平台构建](https://docs.docker.com/build/building/multi-platform/)
- [项目 PRD](prd.md)
