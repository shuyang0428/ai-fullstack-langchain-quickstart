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

# 如果模板需要接收一组历史消息，不能简单地把 Python 列表塞进 {history}。
# 这样会把列表转换成一段没有角色信息的文字，模型无法准确判断哪些是用户说的，哪些是助手回答的。
from langchain.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名耐心的客服助手。"),
    MessagesPlaceholder("history", optional=True), #调用 prompt.invoke的时候把history这个变量传入就会把掐红的消息逐条插入这里
    ("human", "{question}"),
])

history = [
    HumanMessage("我的订单还没到。"),
    AIMessage("请提供订单号，我来帮你查询。"),
]

prompt_value = prompt.invoke({
    "history": history,
    "question": "订单号是 A1024。",
})

# for message in prompt_value.to_messages():
#     print(type(message).__name__, message.text)


# 固定一部分变量
# ​有些变量对整个应用都不变。例如产品名称、品牌语气和输出语言，没有必要每次调用都重新传入。
# partial 可以先固定这些值，得到一个新的模板：
# partial 不会修改原模板，而是返回一个已经填好部分变量的新模板。
# 这样可以从一个基础模板派生出不同产品或不同语言的版本，又不用复制整段内容。
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{product_name}的客服，统一使用{language}回答。"),
    ("human", "{question}"),
])
# 预先固定每次都相同的变量
customer_service_prompt = base_prompt.partial(
    product_name="云记账",
    language="中文",
)
# 只需要传入尚未固定的变量
prompt_value = customer_service_prompt.invoke({
    "question": "如何导出本月账单？"
})

# 加入Few-Shot
#当一条规则很难用语言解释清楚时，可以在 Prompt 中放少量输入输出示例，让模型直接观察期望模式。
# 这种方法通常叫 Few-shot Prompting。
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "把用户反馈分类为 bug、feature 或 question，只返回分类名称。"
        ),
        ("human", "登录后页面一直空白"),
        ("ai", "bug"),
        ("human", "希望增加深色模式"),
        ("ai", "feature"),
        ("human", "{feedback}"),
    ]
)
# 先生成 fewshot消息再调用模型
response = (prompt | model).invoke({ #把prompt的输出给model
    "feedback": "导出的文件保存在哪里？"
})