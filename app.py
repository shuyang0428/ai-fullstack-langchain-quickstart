import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

# Get the api key from .env
load_dotenv()

# create chat model bot
model = init_chat_model(
    f"anthropic:{os.environ['DEFAULT_LLM_MODEL']}",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

# invoke once and response complete aimessage
response = model.invoke('explain what problem LangChain could solve in 3 sentances')

# print(response.content)

# chat with character and history record 同样的消息也能写成字典，模型不会自动记住上一次调用
messages = [
    SystemMessage('你是一名python教练，回答的时候先讲直觉，再给代码'),
    HumanMessage('列表推导式是什么？'),
    AIMessage('列表推导式是一种用简洁语法创建列表的方式。'),
    HumanMessage('给我一个过滤偶数的例子')
]

response = model.invoke(messages)

# print(response.text)

# 两次invoke相互独立，不会自动共享上下文
model.invoke('My name is Daien')
response = model.invoke('whats my name')
# print(response.text)

# 要想模型理解上下文需要把历史消息也一起传入
messages = [
    HumanMessage('my name is diane please remember my name'),
]

# 保留完整的aimessage，保留assistant角色
first_response = model.invoke(messages)
messages.append(first_response)

messages.append(HumanMessage('what is my name'))

# 再次发送包含历史的完整消息列表
second_response = model.invoke(messages)
print(second_response.text) 