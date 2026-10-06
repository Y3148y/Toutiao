"""
Prompt 构造。

核心约束：模型**只能依据给定的新闻作答**，资料里没有的必须说不知道。
不写这段约束，模型会用自己的世界知识编造「看似合理」的新闻内容 ——
对新闻类问答来说这是最严重的错误，比答不出来糟得多。
"""
from datetime import datetime

from ai.config import TOP_K_FINAL
from ai.retriever import RetrievedNews

NEWS_QA_SYSTEM_PROMPT = """你是一个新闻问答助手，负责基于检索到的新闻资料回答用户问题。

必须遵守以下规则：
1. 只使用【新闻资料】中的内容来回答问题。资料里没有的信息，一律不要补充。
2. 如果【新闻资料】不足以回答问题，直接说明"检索到的新闻资料无法回答该问题"，
   然后简要说明你从资料里实际能确认什么。不要猜测、不要推断、不要补充背景知识。
3. 每条结论后面用 [1]、[2] 这样的编号标注来源编号，对应资料列表的序号。
4. 如果多篇资料说法不一致，如实说明存在多个不同说法，不要自己判断哪个正确。
5. 用简洁的中文回答，先给结论再给依据。不要复述整个问题。
6. 不要输出"根据资料""根据新闻"这类前缀，直接说结论。"""


def _format_publish_time(value: datetime | None) -> str:
    if value is None:
        return "未知"
    # 数据库里是 DateTime 到秒，前端展示到天就够了
    return value.strftime("%Y-%m-%d")


def build_news_context(retrieved: list[RetrievedNews], limit: int = TOP_K_FINAL) -> str:
    """
    把召回的新闻拼成给模型看的上下文。

    编号从 1 开始，和 prompt 里要求模型标注的 [1][2] 对应。
    """
    blocks = []
    for i, item in enumerate(retrieved[:limit], start=1):
        blocks.append(
            f"[{i}] 标题：{item.title}\n"
            f"    发布时间：{_format_publish_time(item.publish_time)}\n"
            f"    正文摘要：{item.excerpt.strip()}"
        )
    return "\n\n".join(blocks)


def build_news_qa_messages(
    question: str,
    retrieved: list[RetrievedNews],
    history: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    构造 chat completions 需要的 messages。

    history 是可选的多轮历史 [(user, assistant), ...]。
    只带最近 3 轮：上下文越长 token 成本越高，且新闻问答里
    久远的轮次对当前问题基本没有帮助。
    """
    context = build_news_context(retrieved)

    messages: list[dict] = [{"role": "system", "content": NEWS_QA_SYSTEM_PROMPT}]

    if history:
        for user_msg, assistant_msg in history[-3:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append(
        {
            "role": "user",
            "content": f"【新闻资料】\n{context}\n\n【用户问题】\n{question}",
        }
    )
    return messages


REFUSAL_ANSWER = "检索到的新闻资料无法回答该问题。"


def build_refusal_message(question: str, retrieved: list[RetrievedNews]) -> str:
    """
    置信度不足时的拒答。

    注意这里仍然把召回的内容带出来 —— 完全不给信息体验太差，
    告诉用户「我找到了这几条相关报道，但不足以回答这个问题」更有用，
    同时模型也没有机会编造。
    """
    if not retrieved:
        return f"{REFUSAL_ANSWER}你可以换个说法再问，或直接浏览新闻列表。"

    titles = "、".join(f"《{item.title}》" for item in retrieved[:3])
    return (
        f"{REFUSAL_ANSWER}\n\n"
        f"与你的问题最相关的报道是：{titles}\n"
        f"这些内容不足以给出可靠答案，为避免误导，我不做推测。"
    )