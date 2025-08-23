import getpass
import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import PROMPT_RUNNABLE_NAME
from mcp.server.streamable_http import MCP_PROTOCOL_VERSION_HEADER
from pydantic import BaseModel, Field
import asyncio

load_dotenv()  # load from .env file

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.chat_models import init_chat_model


class RAGAgent:
    """A RAG (Retrieval-Augmented Generation) agent with proper dependency management."""
    
    def __init__(self):
        # Initialize embeddings and vector store
        from langchain_ollama import OllamaEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.rag_data_loaded = False
        
        # Initialize MCP client
        self.mcp_client = MultiServerMCPClient({
            "math": {
                "command": "python",
                "args": ["mcp_servers/math.py"],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": ["mcp_servers/weather.py"],
                "transport": "stdio",
            },
            # "excel": {
            #     "command": "uvx",
            #     "args": ["excel-mcp-server", "stdio"],
            #     "transport": "stdio",
            # }
        })
        
        # LLM instances (will be set during initialization)
        self.llm = None
        self.llm_with_tools = None
        
        # Local tools
        self.local_tools = [self._create_retrieve_tool(), self._create_addition_tool()]
        self.mcp_tools = []
        
        # Graph (will be built after LLM initialization)
        self.graph = None
        
        # Prompts
        self.GENERAL_PROMPT = """
        You are a helpful assistant that can answer questions and perform tasks using tools.
        But you don't have to use tools if you can answer the question directly. \n
        Here is the conversation history:
        {messages} \n \n
        Answer the last question from the user
        """
        
        self.GRADE_PROMPT = '''
        you are a grader assessing relevance of a retrieved content to a user question. 
        here is the retrieved content: {context},
        here is the user question: {question},
        if the retrieved content is relevant to the question, then grade it as such.
        return a binary "Yes" if relevant, or "No" if not relevant.
        '''
        
        self.REWRITE_PROMPT = """
        the initial question is {orig_question}, assess it and try to reason about the underlying semantic intent / meaning. \n
        you've previously generated a query to Tool call, refer to the following conversation history.
        \n {messages_history}
        """
    
    def _create_retrieve_tool(self):
        """Create the retrieve tool with access to instance variables."""
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """retrieve information related to a query that's related to lilianweng"""
            if not self.rag_data_loaded:
                self._load_rag_data()
                self.rag_data_loaded = True

            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        return retrieve
    
    def _create_addition_tool(self):
        """Create the addition tool."""
        @tool
        def addition(a: int, b: int):
            """add two numbers together"""
            return a + b
        
        return addition
    
    def _load_rag_data(self):
        """Load data from a blog and chunk it into smaller pieces."""
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
        _ = self.vector_store.add_documents(documents=all_splits)
        print(f"Loaded {len(all_splits)} chunks into the vector store.")
    
    async def initialize_llm(self, mcp_sessions):
        """Initialize LLM instances with tools."""
        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
        # self.llm = init_chat_model(model="qwen3:0.6b", model_provider="ollama", reasoning=False)

        # sequential loading, make concurrent loading later
        for session in mcp_sessions:
            self.mcp_tools.extend(await load_mcp_tools(session))
        
        # this is where we tell the model about the tools
        self.llm_with_tools = self.llm.bind_tools(self.mcp_tools+self.local_tools)
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        graph_builder = StateGraph(MessagesState)
        
        # Add nodes
        graph_builder.add_node("query_or_respond", self._query_or_respond)
        graph_builder.add_node("tools", ToolNode(tools=self.mcp_tools+self.local_tools)) # this is where the tools are actually added to our usable toolset
        graph_builder.add_node("rewrite_query", self._rewrite_query)
        graph_builder.add_node("generate", self._generate)
        
        # Set entry point
        graph_builder.set_entry_point("query_or_respond")
        
        # Add edges
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
            self._grade,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        graph_builder.add_edge("generate", END)
        graph_builder.add_edge("rewrite_query", "query_or_respond")
        
        return graph_builder.compile()
    
    def _query_or_respond(self, state: MessagesState):
        """Node 1: Query or respond using LLM with tools."""
        msgs = state['messages']
        prompt = self.GENERAL_PROMPT.format(messages=msgs)
        response = self.llm_with_tools.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}
    
    def _grade(self, state: MessagesState) -> Literal['generate', 'rewrite_query']:
        """Node 3: Grade the relevance of retrieved content."""
        orig_question = state['messages'][0].content
        retrieved_ctx = state['messages'][-1].content

        if state['messages'][-1].name != 'retrieve':
            print("early exit'ed from grade()")
            return "generate"
        
        class GradeQuery(BaseModel):
            binary_score: str = Field(
                description="Relevance score: 'Yes' if relevant, 'No' if irrelevant"
            )
        
        prompt = self.GRADE_PROMPT.format(question=orig_question, context=retrieved_ctx)
        response = (self.llm.
                   with_structured_output(GradeQuery).
                   invoke([HumanMessage(content=prompt)])
        )
        print(f"================================== Function Grade() ==================================\n"
              f"grade(): relevant? {response.binary_score}")
        
        if response.binary_score == "Yes":
            return "generate"
        else:
            return "rewrite_query"
    
    def _rewrite_query(self, state: MessagesState):
        """Node 4: Rewrite question if deemed not relevant."""
        orig_question = state['messages'][0].content
        msgs = state['messages']
        
        prompt = self.REWRITE_PROMPT.format(orig_question=orig_question, messages_history=msgs[1:])
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {'messages': [response]}
    
    def _generate(self, state: MessagesState):
        """Node 5: Generate final answer using retrieved context."""
        context = state['messages'][-1].content
        orig_question = state['messages'][0].content
        
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
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}
    
    async def stream_graph_updates(self, user_input: str, config: dict):
        """Stream graph execution updates."""
        async for event in self.graph.astream(
            MessagesState({"messages": [HumanMessage(content=user_input)]}),
            config=config,
            stream_mode="values"
        ):
            event["messages"][-1].pretty_print()
    
    async def run(self):
        """Main execution loop."""
        config = {"configurable": {"thread_id": "123"}}

        async with self.mcp_client.session("math") as math_session, self.mcp_client.session("weather") as weather_session:
            mcp_sessions = [math_session, weather_session]
            
            await self.initialize_llm(mcp_sessions)
            # Build graph after LLM initialization
            self.graph = self._build_graph()
            
            print("RAG Agent initialized. Type 'quit', 'exit', or 'q' to stop.")
            
            while True:
                try:
                    user_input = input("User: ")
                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        break
                    
                    await self.stream_graph_updates(user_input, config)
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue


# Alternative approach using dependency injection
class LLMManager:
    """Manages LLM instances and provides them to other components."""
    
    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
    
    async def initialize(self, tools):
        """Initialize LLM instances."""
        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
        self.llm_with_tools = self.llm.bind_tools(tools)
    
    def get_llm(self):
        """Get the base LLM instance."""
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        return self.llm
    
    def get_llm_with_tools(self):
        """Get the LLM instance with tools bound."""
        if self.llm_with_tools is None:
            raise RuntimeError("LLM with tools not initialized. Call initialize() first.")
        return self.llm_with_tools


# # Factory function approach
# async def create_rag_agent():
#     """Factory function to create and initialize a RAG agent."""
#     agent = RAGAgent()
#     return agent


# Main execution
async def main():
    agent = RAGAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())