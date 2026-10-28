from fastapi import FastAPI,Request
from pydantic import BaseModel
import uvicorn
from langchain.chat_models import init_chat_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("收到请求")
logger.warning("模型返回空内容")
logger.error("调用失败")


model = init_chat_model(
    model="qwen3.8-flash",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.YHPILPH.3j9W.MEUCIQDX6Q9FgzLcCcpaJP2pLNQmOF4VAFzuWxJZ7ugC1K7rdwIgKuxzliqLKXhWq8m6GacSnCI5uoMYqkhJ2kHRB5otEd4",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai"
)

model2 = init_chat_model(
    model="qwen3.5:9b",
    base_url="http://192.168.1.105:11434/v1",
    api_key="123",
    max_tokens=1000,
    model_provider="openai"
)


app = FastAPI()


@app.post("/users")
async def user_chat(request: Request):

    data = await request.json()

    messages = data.get("messages")
    for i in range(3):
        response = model.invoke(messages)
        if response.content is not None:
            logger.info("使用模型1成功，重试次数为%d", i+1)
            break
        if response.status_code == 500:
            logger.error("模型网络错误，已经使用替换模型")
            response = model2.invoke(messages)
        


    return {"ai_content": response.content}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)