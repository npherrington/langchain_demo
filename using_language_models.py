from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import os

# API key for openai - need to setup the api key as an environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Select a model
model = ChatOpenAI(
    model="gpt-4o-mini"
)  # Large Language Models (LLMs) are advanced machine learning models that excel in a wide
# range of language-related tasks such as text generation, translation, summarization, question answering, and more, without needing
# task-specific fine tuning for every scenario.

# Messages to send to the model
messages = [
    SystemMessage(
        "Translate the following from English into Italian"
    ),  # Class(BaseMessage): message for priming the AI behaviour
    HumanMessage("hi!"),  # Class(BaseMessage): message from a human
]

reply = model.invoke(messages)
print(f"Response from the LLM: \n{reply}")

# Alternate formating
print(
    """ 
.invoke can also include the folowing alternate formats:
- model.invoke("Hello")
- model.invoke([{"role": "user", "content": "Hello"}])
- model.invoke([HumanMessage("Hello")])
"""
)

# Streaming:
# Because chat models are Runnables, they expose a standard interface that includes async and streaming modes of invocation.
# This allows us to stream individual tokens from a chat model:

print("Responses can also be streamed (each token is separated by |):\n")
for token in model.stream(messages):
    print(token.content, end="|")
