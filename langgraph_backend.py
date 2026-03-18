from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama

model = ChatOllama(
    model="phi",
    temperature=0,
    max_tokens=100,
    base_url="http://localhost:11434"
)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages = state['messages'] # full conversation history
    response = model.invoke(messages) # send everything
    return {'messages':[response]} # append new reply

checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkpointer)