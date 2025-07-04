from typing import Any, Dict, Generator, Optional, Union, List
import uuid
import yaml
import mlflow
from mlflow.models import ModelConfig
from databricks_langchain import (
    ChatDatabricks,
    DatabricksVectorSearch
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from typing import TypedDict
import os
import json

############################################
# Configuration handling
############################################
try:
    mlflow_config = ModelConfig()
    if mlflow_config.get("LLM_ENDPOINT_NAME"):
        print("Using MLflow ModelConfig")
        config_dict = mlflow_config.to_dict()
    else:
        raise ValueError("ModelConfig is empty")
except:
    print("Loading config from local file")
    # Updated path - go up two levels from src/agent_modules/ to reach config/
    with open("../../config/agent_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

# print(f"Using LLM endpoint: {config_dict.get('LLM_ENDPOINT_NAME')}")
# print(f"Using catalog: {config_dict.get('VECTOR_CATALOG')}")
# print(f"Using schema: {config_dict.get('VECTOR_SCHEMA')}")
# print(f"Using vector index: {config_dict.get('VECTOR_INDEX_NAME')}")

# Assemble the full vector index path from components
catalog = config_dict.get("VECTOR_CATALOG")
schema = config_dict.get("VECTOR_SCHEMA")
base_name = config_dict.get("VECTOR_INDEX_NAME")
vector_index_full_path = f"{catalog}.{schema}.{base_name}"

# Initialize components
LLM_ENDPOINT_NAME = config_dict.get("LLM_ENDPOINT_NAME")
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
system_prompt = config_dict.get("SYSTEM_PROMPT")

# Get vector search columns from config, with defaults matching your schema
vector_columns = config_dict.get("VECTOR_COLUMNS")
num_results = config_dict.get("NUM_RESULTS", 3)

# Create vector store
vector_store = DatabricksVectorSearch(
    index_name=vector_index_full_path,
    columns=vector_columns
)

#####################
# State and workflow
#####################


class DirectRetrievalState(TypedDict):
    messages: List[Dict]
    context: Optional[str] = None
    user_query: Optional[Dict] = None
    source_documents: Optional[List[Dict]] = None
    sources: Optional[List[str]] = None  # Store simplified source list
    assistant_response: Optional[Dict] = None


def call_retrieval(state: DirectRetrievalState) -> DirectRetrievalState:
    """Retrieve relevant documents and format sources"""
    # Get latest user query
    user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
    if not user_messages:
        return {"messages": [{"role": "assistant", "content": "No user query found."}]}

    latest_query = user_messages[-1]["content"]

    # Get documents from vector store
    retrieved_docs = vector_store.similarity_search(
        latest_query, k=num_results)

    # Format context and extract sources in one pass
    context_parts = []
    source_documents = []
    sources = []

    for doc in retrieved_docs:
        # Format context
        context_parts.append(f"Content: {doc.page_content}")
        if hasattr(doc, 'metadata') and doc.metadata:
            context_parts.append(f"Metadata: {doc.metadata}")
        context_parts.append("---")

        # Process document for sources
        source_doc = {"page_content": doc.page_content, "metadata": {}}

        # Extract metadata and source
        source = "No source found"
        web_url = None
        if hasattr(doc, 'metadata') and doc.metadata:
            source_doc["metadata"] = doc.metadata.copy()

            # Set chunk_id
            if "id" in doc.metadata and "chunk_id" not in doc.metadata:
                source_doc["metadata"]["chunk_id"] = doc.metadata["id"]
            elif "chunk_id" not in doc.metadata:
                source_doc["metadata"]["chunk_id"] = str(uuid.uuid4())

            # Get source from product_id first, then fallback to other fields
            if "product_id" in doc.metadata:
                source = doc.metadata["product_id"]
            else:
                # Fallback to other possible source fields
                for field in ["indexed_doc", "source", "path", "product_doc"]:
                    if field in doc.metadata:
                        source = doc.metadata[field]
                        break

            # Capture web_url separately
            if "web_url" in doc.metadata:
                web_url = doc.metadata["web_url"]
        else:
            source_doc["metadata"]["chunk_id"] = str(uuid.uuid4())

        # Clean up source path for display
        if isinstance(source, str) and '/' in source:
            source = os.path.basename(source)

        source_doc["metadata"]["source"] = source
        source_doc["metadata"]["web_url"] = web_url
        source_documents.append(source_doc)

        # Store source information as a dictionary with display name and URL
        source_info = {"name": source}
        if web_url:
            source_info["url"] = web_url
        sources.append(source_info)

    return {
        "messages": state["messages"],
        "context": "\n".join(context_parts),
        "user_query": {"role": "user", "content": latest_query},
        "source_documents": source_documents,
        "sources": sources
    }


def generate_response(state: DirectRetrievalState) -> DirectRetrievalState:
    """Generate LLM response with context"""
    # Build prompt with context
    context_str = f"\nRetrieved context:\n{state['context']}"

    # Prepare messages for LLM
    messages = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": system_prompt + context_str})

    # Add conversation history (excluding system messages)
    user_assistant_messages = [
        msg for msg in state["messages"] if msg["role"] != "system"]
    messages.extend(user_assistant_messages)

    # Generate response
    response = llm.invoke(messages)
    content = response.content

    # Clean content (remove any "Sources:" section added by LLM)
    if "\n\nSources:" in content:
        content = content.split("\n\nSources:")[0]

    # Create response with UUID
    response_id = str(uuid.uuid4())
    assistant_message = {
        "role": "assistant",
        "content": content,
        "id": response_id
    }

    return {
        "messages": state["messages"] + [assistant_message],
        "source_documents": state.get("source_documents", []),
        "sources": state.get("sources", []),
        "assistant_response": assistant_message
    }


def create_direct_retrieval_agent(
    model: LanguageModelLike,
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create and compile the agent workflow"""
    workflow = StateGraph(DirectRetrievalState)

    workflow.add_node("retrieval", RunnableLambda(call_retrieval))
    workflow.add_node("response", RunnableLambda(generate_response))

    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "response")
    workflow.add_edge("response", END)

    return workflow.compile()


#####################
# Chat agent implementation
#####################

class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: Union[list[ChatAgentMessage], dict],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Generate response with sources in a single pass"""
        # Process input messages
        if isinstance(messages, dict) and "messages" in messages:
            msg_list = messages["messages"]
            # Ensure all messages have IDs
            for msg in msg_list:
                if isinstance(msg, dict) and "id" not in msg:
                    msg["id"] = str(uuid.uuid4())
            request = {"messages": msg_list}
        else:
            request = {"messages": [msg.model_dump() for msg in messages]}

        # Execute agent and get final state
        final_state = self.agent.invoke(request)

        # Extract response and sources from final state
        assistant_message = final_state.get("assistant_response")
        sources = final_state.get("sources", [])

        # Debug output to verify we have sources
        # print(f"Sources found: {sources}")

        if assistant_message:
            # Create message without sources in content
            message = ChatAgentMessage(
                role=assistant_message.get("role", "assistant"),
                content=assistant_message.get("content", ""),
                id=assistant_message.get("id")
            )

            # Create response with sources in custom_outputs
            response = ChatAgentResponse(
                messages=[message],
                custom_outputs={"sources": sources} if sources else None
            )
            # Debug final response to verify we have sources
            # print(f"Final response with custom_outputs: {response}")
            return response
        else:
            # Fallback if no message was found
            fallback = ChatAgentMessage(
                role="assistant",
                content="I apologise, but I wasn't able to generate a response.",
                id=str(uuid.uuid4())
            )
            return ChatAgentResponse(messages=[fallback])

    def predict_stream(
        self,
        messages: Union[list[ChatAgentMessage], dict],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Stream response with sources in a single pass"""
        # Process input messages
        if isinstance(messages, dict) and "messages" in messages:
            msg_list = messages["messages"]
            # Ensure all messages have IDs
            for msg in msg_list:
                if isinstance(msg, dict) and "id" not in msg:
                    msg["id"] = str(uuid.uuid4())
            request = {"messages": msg_list}
        else:
            request = {"messages": [msg.model_dump() for msg in messages]}

        # Pre-fetch sources for streaming
        try:
            final_state = self.agent.invoke(request)
            sources = final_state.get("sources", [])
            print(f"Pre-fetched sources for streaming: {sources}")

            # Store sources for later use by the frontend
            # This doesn't affect the streaming but ensures they're accessible
            # after streaming is complete
            self._last_sources = sources
        except Exception as e:
            print(f"Error pre-fetching sources: {str(e)}")
            self._last_sources = []

        # Variables for streaming
        response_id = None
        full_content = ""  # Track full content for final chunk

        # Stream updates from the agent
        for event in self.agent.stream(request, stream_mode="updates"):
            # Extract sources from retrieval node if we didn't pre-fetch them
            if not hasattr(self, '_last_sources') or not self._last_sources:
                if "retrieval" in event and "sources" in event["retrieval"]:
                    self._last_sources = event["retrieval"].get("sources", [])
                    print(
                        f"Extracted sources during streaming: {self._last_sources}")

            # Process response node chunks
            if "response" in event and "assistant_response" in event["response"]:
                msg = event["response"]["assistant_response"]
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    # Get or set response ID
                    if response_id is None:
                        response_id = msg.get("id", str(uuid.uuid4()))

                    # Create delta with cleaned content
                    content = msg.get("content", "")
                    if "\n\nSources:" in content:
                        content = content.split("\n\nSources:")[0]

                    # Track the full content
                    full_content = content

                    delta = {
                        "role": "assistant",
                        "content": content,
                        "id": response_id
                    }

                    # Create and yield the chunk
                    chunk = ChatAgentChunk(**{"delta": delta})
                    yield chunk

        # After streaming completes, yield a final chunk with sources in custom_outputs
        # This matches the API spec which states sources will be in the final complete response
        if response_id is not None:
            final_delta = {
                "role": "assistant",
                "content": "",  # Empty content for final chunk
                "id": response_id
            }

            # Create the final chunk with sources in custom_outputs
            final_chunk = ChatAgentChunk(
                delta=final_delta,
                custom_outputs={
                    "sources": self._last_sources} if self._last_sources else None
            )

            yield final_chunk


#####################
# Main agent setup
#####################
mlflow.langchain.autolog()
agent = create_direct_retrieval_agent(llm, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
