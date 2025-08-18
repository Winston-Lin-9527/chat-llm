import getpass
import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import PROMPT_RUNNABLE_NAME
from pydantic import BaseModel, Field

load_dotenv() # load from .env file

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
# llm = init_chat_model("qwen3:1.7b", model_provider="ollama")


import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# embedding

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
)

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)


rag_data_loaded: bool = False

def rag_load_data():
    """Load data from a blog and chunk it into smaller pieces."""
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    print(f"Loaded {len(all_splits)} chunks into the vector store.")


from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

# allow model to re-write user queries to include context based on past messages

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

"""
1. User inputs as HumanMessage
2. query to vector store as AIMessage(with tool calls)
3. query result comes back as ToolMessage
4. finally generate() responds as AIMessage
"""


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """ retrieve information related to a query that's related to lilianweng""" # this comment matters
    global rag_data_loaded
    if not rag_data_loaded:
        rag_load_data()
        rag_data_loaded = True

    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def addition(a: int, b: int):
    """ add two numbers together """
    return a + b
    

# or just replace with this 
# retriever = vector_store.create_retriever_tool()
# from langchain.tools.retriever import create_retriever_tool
# retriever_tool = create_retriever_tool(
#     retriever,
#     "retriever_tool",
#     "search and return information related to a query"
# )


tools_set = [retrieve, addition]
llm_with_tools = llm.bind_tools(tools_set)

GENERAL_PROMPT = """
    You are a helpful assistant that can answer questions and perform tasks using tools.
    But you don't have to use tools if you can answer the question directly. \n
    Here is the conversation history:
    {messages} \n \n
    Answer the last question from the user
"""

# node 1
def query_or_respond(state: MessagesState):
    # function comment here doesn't matter at all
    msgs = state['messages']
    prompt = GENERAL_PROMPT.format(messages=msgs)
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    
    return {"messages" : [response]}

# node 2 - node to do tool calls, adds result as a ToolMessage to the state
tool_node = ToolNode(tools=tools_set)

# node 3 - prompt vs retrieved context grading agent 
GRADE_PROMPT = '''
    you are a grader assessing relevance of a retrieved content to a user question. 
    here is the retrieved content: {context},
    here is the user question: {question},
    if the retrieved content is relevant to the question, then grade it as such.
    return a binary "Yes" if relevant, or "No" if not relevant.
'''

class GradeQuery(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'Yes' if relevant, 'No' if irrelevant"
    )

def grade(state: MessagesState) -> Literal['generate', 'rewrite_query']:
    orig_question = state['messages'][0].content
    retrieved_ctx = state['messages'][-1].content
    
    prompt = GRADE_PROMPT.format(question=orig_question, context=retrieved_ctx)
    response = (llm. # so adding the outer bracket allows multi-line 
            with_structured_output(GradeQuery).
            invoke([HumanMessage(content=prompt)])
    )
    print(f"""================================== Function Grade() ==================================\n
          grade(): relevant? {response.binary_score}""")
    
    if response.binary_score == "Yes":
        return "generate"
    else:
        return "rewrite_query"


# node 4 - rewrite question if deemed not relevant
REWRITE_PROMPT = """
    the initial question is {orig_question}, assess it and try to reason about the underlying sementic intent / meaning. \n
    you've previously generated a query to Tool call, refer to the following conversation history.
    \n {messages_history}
"""

def rewrite_query(state: MessagesState):
    orig_question = state['messages'][0].content
    msgs = state['messages'] # todo: maybe extract the last generated query separately
    
    prompt = REWRITE_PROMPT.format(orig_question=orig_question, messages_history=msgs[1:])
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {'messages': response} # append to state.messages


# node 5
def generate(state: MessagesState):
    # collect all the retrieved context
    # recent_tool_msgs = []
    # for msg in reversed(state['messages']):
    #     if msg.type == 'tool':
    #         recent_tool_msgs.append(msg)
    #     else:
    #         break
    # tool_msgs = recent_tool_msgs[::-1] # reverse
    # context_combined = "\n\n".join(doc.content for doc in tool_msgs)
    
    
    # only collect the last, relevant context that passed grade() 
    context = state['messages'][-1].content
    
    orig_question = state['messages'][0].content
    
    # non tool calls
    # conversation_msgs = [
    #     msg
    #     for msg in state["messages"]
    #     if msg.type in ('system', 'human')
    #     or msg.type == 'ai' and not msg.tool_calls
    # ]
    
    # prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know. Use three sentences maximum and keep the "
    #     "answer concise."
    #     "\n\n"
    #     "question: {orig_question}"
    #     f"context: {context}"
    # )
    
    # past_convo = [SystemMessage(system_message_content)] + conversation_msgs
    # response = llm.invoke(past_convo) 
    
    prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"question: {orig_question}"
        f"context: {context}"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages" : [response]}
    

    
    
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {
        "__end__": END, 
        "tools": "tools"
    },
)
graph_builder.add_conditional_edges(
    "tools",
    grade,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query"
    } 
)
graph_builder.add_edge("generate", END)
graph_builder.add_edge("rewrite_query", "query_or_respond")

graph = graph_builder.compile()



from langchain_core.messages import HumanMessage
# input_message = "what's 89+12"

# for step in graph.stream(
#     {"messages": [HumanMessage(content=input_message)]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
    
    
config = {"configurable": {"thread_id": "123"}}

def stream_graph_updates(user_input):
    # for event in graph.stream({"messages": conversation_history}):
    for event in graph.stream(
        # default initial state
        # {"messages": [{"role": "user", "content": user_input}]},
        MessagesState({"messages": [HumanMessage(content=user_input)]}),
        config=config, # memory checkpoint to preserve conversation history
        stream_mode="values"
    ):
        # graph.stream() yields events as the graph executes each node
        # for value in event.values():
        #     # print("Assistant:", value["messages"][-1].content)
        #     print(value)
        #     value["messages"][-1].pretty_print()
        event["messages"][-1].pretty_print() 
            


# conversation_history = [] # if wanted to preserve conversation history manually, then use this
while True:
    # try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
    
        # conversation_history.append({"role": "user", "content": user_input})
        stream_graph_updates(user_input)