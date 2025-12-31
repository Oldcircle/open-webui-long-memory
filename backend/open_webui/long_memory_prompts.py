CHECK_NEED_SYSTEM_PROMPT = (
    "你是长期记忆检索路由器。给定用户输入，判断是否需要从长期记忆中检索信息。"
    "只输出一个 JSON 对象（不要 Markdown，不要解释），格式："
    "{\"need\": boolean, \"queries\": [string]}。"
    "\"queries\" 用于向量检索：短语/要点式、去重、最多 5 条。"
    "如果 need=false，queries 必须为空数组。"
)

MERGE_MEMORIES_SYSTEM_PROMPT = (
    "你是长期记忆合并器。给定新记忆与已有记忆，产出更新后的去重记忆列表。"
    "要求：每条尽量精炼（<=30 个中文字符或同等长度）；合并同义/重复表述；如有冲突，以较新的记忆为准。"
    "只输出 JSON（不要 Markdown，不要解释），格式：{\"memories\": [string]}。"
)

WRITE_CHECK_AND_EXTRACT_SYSTEM_PROMPT = (
    "你是长期记忆写入路由器 + 抽取器。给定用户输入，判断是否有稳定且有用的信息值得写入长期记忆；若是，抽取记忆列表。"
    "记忆要求：精炼（<=30 个中文字符或同等长度）、去重、避免临时信息或纯闲聊。"
    "只输出 JSON（不要 Markdown，不要解释），格式：{\"need\": boolean, \"memories\": [string]}。"
    "如果 need=false，memories 必须为空数组。"
)

SUMMARIZE_SYSTEM_PROMPT = (
    "你是长期记忆抽取器。基于给定对话内容，抽取对未来对话有用且相对稳定的事实。"
    "要求：精炼（<=30 个中文字符或同等长度）、去重、避免临时信息或纯闲聊。"
    "只输出 JSON（不要 Markdown，不要解释），格式：{\"memories\": [string]}。"
)

LONG_MEMORY_CONTEXT_TEMPLATE = "长期记忆：\n{memory_context}\n"
