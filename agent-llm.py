from asyncio import Event
import json
import getpass
import os
from dotenv import load_dotenv
from typing import Annotated
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv() # load from .env file

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


# from langchain_google_genai import ChatGoogleGenerativeAI
# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.5-flash-lite",
# #     temperature=0,
# #     max_tokens=None,
# #     timeout=None,
# #     max_retries=2,
# #     # other params...
# # )
llm = init_chat_model(
    "gemini-2.5-flash-lite", model_provider="google_genai", temperature=0
)

agent_name = "Captain Jack Sparrow"
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful AI assistant, your name is {agent_name}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# class BasicToolNode:
#     # a node that runs the tools requested in the last AIMessage, if there are any
    
#     def __init__(self, tools: list):
#         self.tools_by_name = {tool.__name__: tool for tool in tools} # {name : tool} dict
    
#     def __call__(self, inputs: dict):
#         print("entered ToolNode")
#         if messages := inputs.get("messages", []):
#             target_message = messages[-1]
#         else:
#             raise ValueError("No message found in input")

#         if len(messages) <= 0 or not isinstance(target_message, AIMessage):
#             raise ValueError("Last message must be an AIMessage, or there must be at least one message.")

#         tool_calls = target_message.tool_calls
#         results = []
#         for tool_call in tool_calls:
#             tool = self.tools_by_name[tool_call["name"]]
#             result = tool(**tool_call["args"]) # execute the tool function using the args that the model came up with, ** : dict unpacking
#             # result = tool.invoke(tool_call["args"]) # this is the culprit

#             results.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"], name=tool_call["name"]))
#         return {"messages": results}



class ChatbotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    agent_name: str

graph_builder = StateGraph(ChatbotState)


# functions tools
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def addition(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


tools = [multiply, addition]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# function nodes
def chatbot(state: ChatbotState):
    print("entered chatbot")
    messages = state["messages"]
    prompt = prompt_template.invoke({"messages": messages, 
                                    "agent_name": state["agent_name"]})
    
    return {"messages": [llm_with_tools.invoke(prompt)]}


def tool_router(state: ChatbotState):
    print("entered tool_router")
    if not state.get("messages", []):
        raise ValueError("No message in state")

    # only select the last message, which should always be an AIMessage, since it follows the chatbot node
    action_message = state["messages"][-1] # only select the last one
    print(f"now in router, last message is {action_message}")

    if isinstance(action_message, AIMessage) and \
        action_message.tool_calls:
        # hasattr(action_message, "tool_calls") and \ 

        for tool in action_message.tool_calls:
            print(f"calling tool {tool}")
        return "tools"
    else:
        # if regular AIMessage without tool calls, end the current conversation
        # !!! without this branch, the chatbot will NOT handle general questions that are not related to the tools
        # todo: i still dont understand why LMAO....
        print("going to END")
        print(action_message)
        return END 


# Fix: Use a unique name for the chatbot function node
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tool_router, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(checkpointer=MemorySaver())


config = {"configurable": {"thread_id": "123"}}

def stream_graph_updates(user_input):
    # for event in graph.stream({"messages": conversation_history}):
    for chunk in graph.stream(
        # default initial state
        ChatbotState({"messages": [{"role": "user", "content": user_input}], "agent_name": agent_name}),
        config=config, # memory checkpoint to preserve conversation history
        stream_mode="messages-tuple"
    ):
        # graph.stream() yields events as the graph executes each node
        # for value in Event.values():
        #     # print("Assistant:", value["messages"][-1].content)
        #     value["messages"][-1].pretty_print()
        if chunk.event != "messages":
            continue

        message_chunk, metadata = chunk.data  # (1)!
        if message_chunk["content"]:
            print(message_chunk["content"], end="|", flush=True)

# conversation_history = [] # if wanted to preserve conversation history manually, then use this
while True:
    # try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
    
        # conversation_history.append({"role": "user", "content": user_input})
        stream_graph_updates(user_input)
        
    # except:
    #     # fallback if input() is not available
    #     user_input = "What do you know about LangGraph?"
    #     print("User: " + user_input)
    #     stream_graph_updates(user_input)
    #     break