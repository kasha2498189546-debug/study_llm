from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables.history import RunnableWithMessageHistory

model = init_chat_model(
    model="qwen3.8-max",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.YHPILPH.3j9W.MEUCIQDX6Q9FgzLcCcpaJP2pLNQmOF4VAFzuWxJZ7ugC1K7rdwIgKuxzliqLKXhWq8m6GacSnCI5uoMYqkhJ2kHRB5otEd4",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai",
    extra_body={"enable_thinking": False})

summary_model = model.bind(temperature=0.2)

SESSION_ID = "user-001"
MAX_MESSAGES = 10
KEEP_RECENT = 4
SUMMARY_TAG = "【前文摘要】"

ROLE_NAMES = {"human": "用户", "ai": "助手", "system": "系统"}


def get_session_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://127.0.0.1:6379/13")


def compress_if_needed(session_id: str) -> bool:
    history = get_session_history(session_id)
    messages = history.messages
    if len(messages) <= MAX_MESSAGES:
        return False

    old_messages = messages[:-KEEP_RECENT]
    recent_messages = messages[-KEEP_RECENT:]

    # 旧摘要会一起参与总结，形成滚动摘要
    dialog_text = "\n".join(
        f"{ROLE_NAMES.get(m.type, m.type)}：{m.content}" for m in old_messages
    )

    try:
        summary = summary_model.invoke([
            SystemMessage(content=(
                "你是对话摘要助手。请把下面的对话压缩成一段简洁的中文摘要，"
                "保留人物、事实、用户偏好和未完成的事项，不超过 200 字。"
            )),
            HumanMessage(content=dialog_text),
        ]).content
    except Exception as exc:
        # 摘要失败就不动原历史
        print(f"[压缩] 生成摘要失败，保留原历史：{exc}")
        return False

    history.clear()
    history.add_message(SystemMessage(content=f"{SUMMARY_TAG}{summary}"))
    for message in recent_messages:
        history.add_message(message)

    print(f"[压缩] {len(messages)} 条历史 -> 摘要 + 最近 {KEEP_RECENT} 条，共 {len(history.messages)} 条")
    print(f"[压缩] 摘要：{summary}")
    return True


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

