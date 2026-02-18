# 长期记忆系统 (Long Memory)

## 概述

长期记忆是一个跨对话的持久化记忆系统。在聊天过程中自动从用户消息中提取事实，存入向量数据库，并在后续对话中通过语义搜索召回相关记忆，注入到 LLM 上下文中。

## 用户设置

Settings > Personalization：

- **Long Memory** 开关 — 启用/禁用
- **Top-K Recall** — 每次召回记忆条数（默认 3）
- **Manage** 弹窗 > Long Memory 标签页 — 查看、手动添加、编辑、删除记忆

## 核心文件

```
backend/open_webui/long_memory.py           # 核心服务：LongMemoryService
backend/open_webui/long_memory_prompts.py    # LLM 提示词（中文）
backend/open_webui/routers/long_memory.py    # REST API 端点
backend/open_webui/models/memories.py        # 数据库模型（LongMemory 表）
backend/open_webui/utils/middleware.py        # 聊天中间件集成
backend/open_webui/internal/migrations/019_add_long_memory.py  # 数据库迁移
src/lib/apis/memories/index.ts               # 前端 API 客户端
src/lib/components/chat/Settings/Personalization.svelte         # 设置 UI
src/lib/components/chat/Settings/Personalization/ManageModal.svelte  # 记忆管理弹窗
src/lib/components/chat/Chat.svelte          # 聊天组件（feature flag 传递）
```

## 技术栈

- **嵌入模型**：`nomic-ai/nomic-embed-text-v1.5`（768 维，通过 SentenceTransformer 本地运行）
- **向量库**：项目内置的 `VECTOR_DB_CLIENT`（默认 ChromaDB）
- **SQL 存储**：`long_memories` 表（SQLite/PostgreSQL）
- **集合命名**：`user-long-memory-{user_id}`（用户隔离）

## 召回流程（用户发消息时）

```
用户发送消息
    |
    v
check_need() — LLM 判断是否需要召回
    |  输入：用户消息
    |  输出：{ need: bool, queries: [string] }
    |
    v  (need=true 或消息含 "记得"/"我的"/"remember" 等关键词)
recall() — 向量语义搜索
    |  1. 将 queries 向量化
    |  2. 在 user-long-memory-{user_id} 集合中搜索
    |  3. 每个 query 取 top-k 结果，跨 query 去重
    |
    v
注入系统消息："长期记忆：\n- 记忆1\n- 记忆2\n..."
    |
    v
LLM 带着记忆上下文生成回复
```

中间件入口：`middleware.py` 的 `chat_long_memory_handler()`

## 存储流程（收到回复后）

```
助手回复完成
    |
    v
用户消息加入批次队列（按 user_id + chat_id 分组）
    |
    v  触发条件：累积 >= 10 条 或 空闲 >= 600 秒
flush 处理
    |
    +-- check_and_summarize() — LLM 提取值得记住的事实
    |     输入：合并后的用户消息文本
    |     输出：{ need: bool, memories: [string] }
    |
    +-- find_similar_existing_memories() — 向量搜索相似度 >= 0.8 的已有记忆
    |
    +-- merge_memories() — LLM 合并新旧记忆，去重，新的优先
    |     输出：{ memories: [string] }
    |
    +-- 删除旧的重复记忆（SQL + 向量库）
    |
    +-- store() — 写入 SQL 表 + 向量库
```

中间件入口：`middleware.py` 的 `_long_memory_store_task()`

## LLM 提示词

所有提示词定义在 `long_memory_prompts.py`，均为中文：

| 提示词 | 用途 |
|--------|------|
| `CHECK_NEED_SYSTEM_PROMPT` | 判断是否需要召回记忆 |
| `WRITE_CHECK_AND_EXTRACT_SYSTEM_PROMPT` | 判断是否有信息值得存储 + 提取 |
| `SUMMARIZE_SYSTEM_PROMPT` | 从对话中提取稳定事实 |
| `MERGE_MEMORIES_SYSTEM_PROMPT` | 合并去重新旧记忆 |

每条记忆限制 <= 30 个中文字符，每轮最多 30 条。

## API 端点

挂载在 `/api/v1/long_memory`：

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 获取当前用户所有记忆 |
| POST | `/add` | 手动添加记忆 |
| POST | `/{id}/update` | 更新记忆内容 |
| DELETE | `/{id}` | 删除单条记忆 |
| DELETE | `/delete/user` | 清除当前用户所有记忆 |
| POST | `/reset` | 从 SQL 重建向量库索引 |
| POST | `/check` | 测试：检查是否需要召回 |
| POST | `/summarize` | 测试：从文本提取记忆 |
| POST | `/store` | 测试：存储记忆 |
| POST | `/recall` | 测试：召回记忆 |
| POST | `/demo` | 测试：完整管线演示 |

## 配置参数

`LongMemoryConfig`（`long_memory.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用 |
| `k` | `3` | 每次召回的记忆条数 |
| `collection_name_prefix` | `user-long-memory-` | 向量集合前缀 |
| `embedding_model` | `nomic-ai/nomic-embed-text-v1.5` | 嵌入模型 |
| `max_queries` | `5` | 每次召回最多查询数 |
| `max_memories_per_turn` | `30` | 每轮最多存储记忆数 |

批次参数（`middleware.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `LONG_MEMORY_BATCH_SIZE` | `10` | 触发即时 flush 的消息数 |
| `LONG_MEMORY_IDLE_FLUSH_SECONDS` | `600` | 空闲多久触发 flush（秒） |

## 调试

设置环境变量开启详细日志：

```bash
LONG_MEMORY_DEBUG=true
```
