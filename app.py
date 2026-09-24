import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Get the api key from .env
load_dotenv()

# create chat model bot
model = init_chat_model(
    f"anthropic:{os.environ['DEFAULT_LLM_MODEL']}",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

# invoke once and response complete aimessage
response = model.invoke('explain what problem LangChain could solve in 3 sentances')

print(response.content)