from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os

# API key for openai - need to setup the api key as an environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Select a model
model = ChatOpenAI(
    model="gpt-4o-mini"
)  # Large Language Models (LLMs) are advanced machine learning models that excel in a wide


system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(  # Class(BaseChatPromptTemplate): prompt template for chat models used to create flexible templated prompts
    [("system", system_template), ("user", "{text}")]
)  # Changed in version 0.2.24: You can pass any Message-like formats supported by ChatPromptTemplate.from_messages() directly to ChatPromptTemplate() init.

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

# print(f"Example output: {prompt}")

prompt.to_messages()

reply = model.invoke(prompt)
print(
    f"Response from the LLM: \n{reply,type(reply)}\n\nReponse string: \n{reply.content}"
)
