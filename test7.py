import requests


url = "http://127.0.0.1:8000/users"

data = {
    "messages": [
        {"role": "system", "content": "你是一名文学家，你擅长诗词歌赋，你的回答必须温文尔雅，用词生动"}
    ]
}

# response = requests.post(url, json=data)
# print(response.json())


def user_chat(messages):
    while True:
        input_message = input("请输入你的问题：")
        messages.append({"role": "user", "content": input_message})
        response = requests.post(url, json=data)
        print(response.json()["ai_content"])
        messages.append({"role": "assistant", "content": response.json()["ai_content"]})
        print("-"*100)


if __name__ == "__main__":
    user_chat(data["messages"])