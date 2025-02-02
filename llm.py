import getpass
import os
from typing import TypedDict, Annotated, Sequence
# from langchain_ollama import ChatOllama
from dotenv import load_dotenv 

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv() # load from .env file

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# llm = ChatOllama(
#     model = "deepseek-r1:1.5b",
#     temperature = 0.8,
#     num_predict = 256
#     # other params ...
# )

AGENT_NAME = "Daddy Lin"

trimmer = trim_messages(
    max_tokens=128,  # limit the content provided to the model's context window
    token_counter=llm,
    strategy="last",
    include_system=True,
    start_on="human",
)

class LLMConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_name: str

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability, your name is {agent_name}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph
workflow = StateGraph(state_schema=LLMConversationState)

# Define the function that calls the model
def call_model(state: LLMConversationState):
    trimmed_msg_context = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_msg_context, 
                                     "agent_name": state["agent_name"]})
    response = llm.invoke(prompt)
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}  

conversation_history = [
    # ("system", "You are a helpful assistance"),
    # ("human", "八百标兵奔北坡，炮兵并排北边跑。"),
    HumanMessage(content="Hi! I'm Bob"),
    AIMessage(content="Okay"),
]

while(True):
    print("================================= Your Message =================================\n")
    query = input("You: ")
    if query == "exit":
        break
    if not query.strip():
        print("Please enter a valid query.")
        continue
    
    conversation_history.append(HumanMessage(content=query))
    # output = app.invoke({"messages": conversation_history, "agent_name": AGENT_NAME}, config) # which then calls llm.invoke 
    consolidated_ai_response = ""
    for chunk, metadata in app.stream(
        {"messages": conversation_history, "agent_name": AGENT_NAME},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            print(chunk.content, end="|", flush=True)   # flush=True to print immediately, no buffering
            consolidated_ai_response += chunk.content
    print() # add a new line
    # ai_response = output["messages"][-1]
    ai_response = AIMessage(content=consolidated_ai_response)
    conversation_history.append(ai_response)    # same as AIMessage(content=ai_response.content)
    # ai_response.pretty_print()  # output contains all messages in state
   

# stream = llm.stream(messages_context)
# full = next(stream)
# for chunk in stream:
#     full += chunk
# print(full)


