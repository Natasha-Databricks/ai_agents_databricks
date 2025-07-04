import functools
import os
import uuid
import yaml
from typing import Any, Optional, Generator, Literal

import mlflow
from pydantic import BaseModel

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatAgentChunk, ChatContext
from mlflow.pyfunc import ChatAgent

# PubMed imports
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# --- Load Config ---
try:
    from mlflow.models import ModelConfig
    config_dict = ModelConfig().to_dict()
except:
    # Try environment variable first, then fallback to relative paths
    config_paths = []

    # Check if environment variable is set
    if "AGENT_CONFIG_PATH" in os.environ:
        config_paths.append(os.environ["AGENT_CONFIG_PATH"])

    # Add fallback paths
    config_paths.extend([
        "agent_config.yaml",              # Current working directory
        "../../config/agent_config.yaml",  # From src/agent_modules/
        "../config/agent_config.yaml",    # From notebooks/
        "config/agent_config.yaml",       # From project root
    ])

    config_dict = None
    for config_path in config_paths:
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                print(f"Successfully loaded config from: {config_path}")
                break
        except FileNotFoundError:
            continue

    if config_dict is None:
        raise FileNotFoundError(
            "Could not find agent_config.yaml in any of the expected locations"
        )

# Core Configuration
GENIE_SPACE_ID = config_dict["GENIE_SPACE_ID"]
LLM_ENDPOINT_NAME = config_dict["LLM_ENDPOINT_NAME"]
VECTOR_INDEX_NAME = f"{config_dict['VECTOR_CATALOG']}.{config_dict['VECTOR_SCHEMA']}.{config_dict['VECTOR_INDEX_NAME']}"
VECTOR_COLUMNS = config_dict.get("VECTOR_COLUMNS")
NUM_RESULTS = config_dict.get("NUM_RESULTS", 3)

# Agent Configuration
MAX_ITERATIONS = config_dict.get("MAX_ITERATIONS", 3)
PUBMED_MAX_DOCS = config_dict.get("PUBMED_MAX_DOCS", 5)

# Prompts Configuration
SYSTEM_PROMPT = config_dict.get("SYSTEM_PROMPT", "")
SUPERVISOR_SYSTEM_PROMPT = config_dict.get("SUPERVISOR_SYSTEM_PROMPT")
PUBMED_SYSTEM_PROMPT = config_dict.get("PUBMED_SYSTEM_PROMPT")
FINAL_ANSWER_PROMPT = config_dict.get("FINAL_ANSWER_PROMPT")

# Worker Descriptions
WORKER_DESCRIPTIONS = config_dict.get("WORKER_DESCRIPTIONS", {
    "Genie": "Structured data expert",
    "Retrieval": "Document search agent",
    "PubMed": "Literature search agent"
})

# Agent Names
AGENT_NAMES = config_dict.get("AGENT_NAMES")

# Error Messages
ERROR_MESSAGES = config_dict.get("ERROR_MESSAGES")

# --- LLM Setup ---
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# --- Genie Agent ---
genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name=AGENT_NAMES["genie"],
    description=WORKER_DESCRIPTIONS["Genie"],
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    )
)

# --- Retrieval Agent ---
vector_store = DatabricksVectorSearch(
    index_name=VECTOR_INDEX_NAME, columns=VECTOR_COLUMNS)


class RetrievalAgent:
    def invoke(self, state):
        user_msg = [m for m in state["messages"] if m["role"] == "user"][-1]
        query = user_msg["content"]

        docs = vector_store.similarity_search(query, k=NUM_RESULTS)
        context_parts = []
        sources = []

        for doc in docs:
            context_parts.append(f"Content: {doc.page_content}")
            if doc.metadata:
                context_parts.append(f"Metadata: {doc.metadata}")
                if "source" in doc.metadata:
                    sources.append(
                        {"name": os.path.basename(doc.metadata["source"])})
            context_parts.append("---")

        context = "\n".join(context_parts)
        prompt = f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context}"
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = llm.invoke(messages)

        return {
            "messages": [{
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": response.content.strip() if response.content else ERROR_MESSAGES["no_response"],
                "name": AGENT_NAMES["retrieval"]
            }]
        }


retrieval_agent = RetrievalAgent()

# --- PubMed Agent ---


class PubMedAgent:
    def __init__(self, max_docs=None):
        self.pubmed_tool = PubmedQueryRun()
        self.max_docs = max_docs or PUBMED_MAX_DOCS

    def invoke(self, state):
        # Get the latest user message
        user_msg = [m for m in state["messages"] if m["role"] == "user"][-1]
        query = user_msg["content"]

        try:
            # Query PubMed
            pubmed_results = self.pubmed_tool.invoke(query)

            # Prepare context with PubMed results
            context = f"PubMed Search Results for query: '{query}'\n\n{pubmed_results}"

            # Create messages for LLM
            messages = [
                {"role": "system",
                    "content": f"{PUBMED_SYSTEM_PROMPT}\n\nContext:\n{context}"}
            ] + state["messages"]

            # Get LLM response
            response = llm.invoke(messages)

            return {
                "messages": [{
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": response.content.strip() if response.content else ERROR_MESSAGES["pubmed_no_results"],
                    "name": AGENT_NAMES["pubmed"]
                }]
            }

        except Exception as e:
            return {
                "messages": [{
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": f"{ERROR_MESSAGES['pubmed_error']}: {str(e)}",
                    "name": AGENT_NAMES["pubmed"]
                }]
            }


pubmed_agent = PubMedAgent()

# --- Supervisor Agent ---
formatted_descriptions = "\n".join(
    f"- {k}: {v}" for k, v in WORKER_DESCRIPTIONS.items())
system_decision_prompt = f"{SUPERVISOR_SYSTEM_PROMPT}\n{formatted_descriptions}"
options = ["FINISH"] + list(WORKER_DESCRIPTIONS.keys())

FINISH = {"next_node": "FINISH"}


def supervisor_agent(state):
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH

    class NextNode(BaseModel):
        next_node: Literal[tuple(options)]

    messages = [
        {"role": "system", "content": system_decision_prompt}] + state["messages"]
    supervisor_chain = RunnableLambda(
        lambda _: messages) | llm.with_structured_output(NextNode)
    next_node = supervisor_chain.invoke(state).next_node

    if state.get("next_node") == next_node and count > 1:
        return FINISH

    return {"iteration_count": count, "next_node": next_node}


def agent_node(state, agent, name):
    try:
        result = agent.invoke(state)
        messages = result.get("messages", [])

        if not messages:
            content = f"[{name}] {ERROR_MESSAGES['no_message']}"
        else:
            last_msg = messages[-1]
            content = (
                last_msg.get("content", "").strip()
                if isinstance(last_msg, dict)
                else getattr(last_msg, "content", "").strip()
            )
            if not content:
                content = f"[{name}] {ERROR_MESSAGES['empty_message']}"

    except Exception as e:
        content = f"[{name}] {ERROR_MESSAGES['agent_error']}: {str(e)}"

    return {
        "messages": [{
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": content,
            "name": name
        }]
    }


def final_answer(state):
    messages = state["messages"] + \
        [{"role": "user", "content": FINAL_ANSWER_PROMPT}]
    final_response = llm.invoke(messages)

    return {
        "messages": [{
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": final_response.content,
            "name": AGENT_NAMES["final_answer"]
        }]
    }


# --- LangGraph StateGraph ---
class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int


workflow = StateGraph(AgentState)
workflow.add_node("Genie", functools.partial(
    agent_node, agent=genie_agent, name=AGENT_NAMES["genie"]))
workflow.add_node("Retrieval", functools.partial(
    agent_node, agent=retrieval_agent, name=AGENT_NAMES["retrieval"]))
workflow.add_node("PubMed", functools.partial(
    agent_node, agent=pubmed_agent, name=AGENT_NAMES["pubmed"]))
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("supervisor")
for node in WORKER_DESCRIPTIONS.keys():
    workflow.add_edge(node, "supervisor")

workflow.add_conditional_edges("supervisor", lambda x: x["next_node"], {
    "Genie": "Genie",
    "Retrieval": "Retrieval",
    "PubMed": "PubMed",
    "FINISH": "final_answer"
})
workflow.add_edge("final_answer", END)
multi_agent = workflow.compile()

# --- ChatAgent Wrapper ---


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        collected_messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                for msg in node_data.get("messages", []):
                    if not msg.get("id"):
                        msg["id"] = str(uuid.uuid4())
                    collected_messages.append(ChatAgentMessage(**msg))
        return ChatAgentResponse(messages=collected_messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {
            "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                for msg in node_data.get("messages", []):
                    if not msg.get("id"):
                        msg["id"] = str(uuid.uuid4())
                    yield ChatAgentChunk(delta=msg)


# --- Final Setup ---
mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)
