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
messages.append(first_response) # type: ignore

messages.append(HumanMessage('what is my name'))

# 再次发送包含历史的完整消息列表
second_response = model.invoke(messages)
# print(second_response.text) 

# 受用prompttemplate生成文本提示词，PromptTemplate接受一段带占位符的文本
# 调用时传入变量，他会生成填充后的完整内容

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
你是一名专业翻译。请把下面内容翻译成{target_language}，保持原本含义和语气，不要增加解释。

待翻译内容：
{text}
""".strip()
)

result = prompt.invoke({
    "target_language": "English",
    "text": "欢迎使用我们的产品",
})

# response = model.invoke(result)
# print(response.content)

# 如果只想拿到普通字符串也可以用format
text_prompt = prompt.format(
    target_language = 'English',
    text = '订单已被发出'
)
# print(text_prompt)

# 使用ChatPromptTemplate生成消息
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名{domain}领域的顾问。回答要准确、简洁；不确定时明确说明。",
    ),
    (
        "human",
        "请回答下面的问题：\n\n{question}",
    ),
])

prompt_value = prompt.invoke({
    "domain": "Python",
    "question": "生成器为什么比一次性创建列表更省内存？",
})

# for message in prompt_value.to_messages():
#     print(type(message).__name__, message.text)


# 把模版和模型连起来
# prompt模版本身不调用模型，他只负责生成模型输入，可以先生成消息再调用模型：
prompt_value = prompt.invoke({
    "domain": "数据库",
    "question": "索引为什么能提高查询速度？",
})

response = model.invoke(prompt_value)
# print(response.content)

# 或者使用管道操作符把两者组合起来
chain = prompt | model

response = chain.invoke({
    "domain": "数据库",
    "question": "索引为什么能提高查询速度？",
})

# print(response.content)