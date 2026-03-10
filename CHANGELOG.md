# Changelog

## v2.5.1 (2026-03-10)

### 🗜️ 上下文智能压缩

解决 Claude Code 频繁出现"继续"按钮的核心问题。

- **智能压缩替代裁剪**：当对话消息超过 30 条或总字符超过 60K 时，自动压缩老消息而非丢弃
  - 工具结果 `Action output: <30KB 文件内容>` → `Action output: [30000 chars, 247 lines] import ...`
  - 助手工具调用 → `[Called read_file(file_path)]`（保留工具名和参数名）
  - 保留因果链语义，减少 70-80% 字符量
- **保留区策略**：few-shot 头部 2 条 + 最近 6 条消息始终保持完整原文

### ⚠️ 截断检测

- **自动检测被截断的响应**：代码块未闭合、XML 标签未闭合时，返回 `stop_reason: "max_tokens"` 让 Claude Code 自动继续，无需手动点击"继续"
- 同时应用于流式和非流式响应

### 🔧 tolerantParse 增强

- **新增第四层正则兜底**：当模型生成的 JSON 工具调用包含未转义双引号（如代码内容参数）导致标准解析和控制字符修复均失败时，使用正则提取 `tool` 名称和 `parameters` 字段
- 解决 `SyntaxError: Expected ',' or '}'` at position 5384 等长参数解析崩溃问题

### 🛡️ 拒绝 Fallback 优化

- 工具模式下拒绝时返回极短文本 `"Let me proceed with the task."`，避免 Claude Code 误判为任务完成

---

## v2.5.0 (2026-03-10)

- OpenAI Responses API (`/v1/responses`) 支持 Cursor IDE Agent 模式
- 跨协议防御对齐（Anthropic + OpenAI handler 共享拒绝检测和重试逻辑）
- 统一图片预处理管道（OCR/Vision API）
