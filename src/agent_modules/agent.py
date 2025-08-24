# https://www.mlflow.org/docs/3.2.0/genai/serving/responses-agent/
import functools
import os
import uuid
import yaml
from typing import Any, Callable, Generator, Optional, Literal

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from pydantic import BaseModel

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from databricks_langchain.genie import GenieAgent
from langchain_community.tools.pubmed.tool import PubmedQueryRun


# --- Load Config ---
def load_config():
    """Load configuration from multiple possible locations."""
    try:
        from mlflow.models import ModelConfig
        return ModelConfig().to_dict()
    except:
        config_paths = [
            os.environ.get("AGENT_CONFIG_PATH"),
            "agent_config.yaml",
            "../../config/agent_config.yaml",
            "../config/agent_config.yaml",
            "config/agent_config.yaml"
        ]
        
        for config_path in filter(None, config_paths):
            try:
                with open(config_path, "r") as f:
                    print(f"Successfully loaded config from: {config_path}")
                    return yaml.safe_load(f)
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError("Could not find agent_config.yaml in any expected location")

config_dict = load_config()

# Configuration
GENIE_SPACE_ID = config_dict["GENIE_SPACE_ID"]
LLM_ENDPOINT_NAME = config_dict["LLM_ENDPOINT_NAME"]
VECTOR_INDEX_NAME = f"{config_dict['VECTOR_CATALOG']}.{config_dict['VECTOR_SCHEMA']}.{config_dict['VECTOR_INDEX_NAME']}"
VECTOR_COLUMNS = config_dict.get("VECTOR_COLUMNS")
NUM_RESULTS = config_dict.get("NUM_RESULTS", 3)
MAX_ITERATIONS = config_dict.get("MAX_ITERATIONS", 3)

# Prompts
SYSTEM_PROMPT = config_dict.get("SYSTEM_PROMPT", "You are a helpful assistant.")
SUPERVISOR_SYSTEM_PROMPT = config_dict.get("SUPERVISOR_SYSTEM_PROMPT", "You are a supervisor deciding which agent to use.")
PUBMED_SYSTEM_PROMPT = config_dict.get("PUBMED_SYSTEM_PROMPT", "You are a PubMed research assistant.")
FINAL_ANSWER_PROMPT = config_dict.get("FINAL_ANSWER_PROMPT", "Please provide a comprehensive final answer.")

# Agent configuration
WORKER_DESCRIPTIONS = config_dict.get("WORKER_DESCRIPTIONS", {
    "Genie": "Structured data expert for database queries",
    "Retrieval": "Document search agent for finding relevant documents", 
    "PubMed": "Literature search agent for scientific publications"
})

AGENT_NAMES = config_dict.get("AGENT_NAMES", {
    "genie": "GenieAgent",
    "retrieval": "RetrievalAgent",
    "pubmed": "PubMedAgent",
    "final_answer": "FinalAnswer"
})

# Initialize LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


class AgentInfo(BaseModel):
    """Class representing an agent tool."""
    name: str
    description: str
    exec_fn: Callable


class MultiAgentResponsesAgent(ResponsesAgent):
    """Multi-agent system using ResponsesAgent pattern."""

    def __init__(self):
        """Initialize all agents."""
        # Initialize Genie Agent
        self.genie_agent = GenieAgent(
            genie_space_id=GENIE_SPACE_ID,
            genie_agent_name=AGENT_NAMES["genie"],
            description=WORKER_DESCRIPTIONS["Genie"],
            client=WorkspaceClient(
                host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
                token=os.getenv("DATABRICKS_GENIE_PAT"),
            )
        )
        
        # Initialize Vector Search
        self.vector_store = DatabricksVectorSearch(
            index_name=VECTOR_INDEX_NAME,
            columns=VECTOR_COLUMNS
        )
        
        # Initialize PubMed
        self.pubmed_tool = PubmedQueryRun()
        
        # Agent registry
        self._agents_dict = {
            "Genie": AgentInfo(
                name="Genie",
                description=WORKER_DESCRIPTIONS["Genie"],
                exec_fn=self.execute_genie
            ),
            "Retrieval": AgentInfo(
                name="Retrieval", 
                description=WORKER_DESCRIPTIONS["Retrieval"],
                exec_fn=self.execute_retrieval
            ),
            "PubMed": AgentInfo(
                name="PubMed",
                description=WORKER_DESCRIPTIONS["PubMed"],
                exec_fn=self.execute_pubmed
            )
        }

    @mlflow.trace(span_type=SpanType.AGENT)
    def execute_genie(self, query: str, messages: list) -> str:
        """Execute Genie agent."""
        try:
            state = {"messages": messages}
            result = self.genie_agent.invoke(state)
            return result.get("messages", [{}])[-1].get("content", "No response from Genie")
        except Exception as e:
            return f"Genie agent error: {str(e)}"

    @mlflow.trace(span_type=SpanType.AGENT) 
    def execute_retrieval(self, query: str, messages: list) -> str:
        """Execute Retrieval agent."""
        try:
            docs = self.vector_store.similarity_search(query, k=NUM_RESULTS)
            context_parts = []
            
            for doc in docs:
                context_parts.append(f"Content: {doc.page_content}")
                if doc.metadata:
                    context_parts.append(f"Metadata: {doc.metadata}")
                context_parts.append("---")
            
            context = "\n".join(context_parts)
            prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query}\n\nRetrieved context:\n{context}\n\nPlease provide a helpful answer based on this context."
            
            response = llm.invoke([{"role": "system", "content": prompt}])
            return response.content.strip() if response.content else "No response from Retrieval"
            
        except Exception as e:
            return f"Retrieval agent error: {str(e)}"

    @mlflow.trace(span_type=SpanType.AGENT)
    def execute_pubmed(self, query: str, messages: list) -> str:
        """Execute PubMed agent."""
        try:
            pubmed_results = self.pubmed_tool.invoke(query)
            prompt = f"{PUBMED_SYSTEM_PROMPT}\n\nQuestion: {query}\n\nPubMed Results:\n{pubmed_results}\n\nPlease provide a helpful answer based on these research findings."
            
            response = llm.invoke([{"role": "system", "content": prompt}])
            return response.content.strip() if response.content else "No PubMed results found"
            
        except Exception as e:
            return f"PubMed agent error: {str(e)}"

    @mlflow.trace(span_type=SpanType.LLM)
    def supervisor_decision(self, messages: list) -> str:
        """Make routing decision using LLM."""
        options = ["Genie", "Retrieval", "PubMed", "FINISH"]
        
        decision_prompt = f"""{SUPERVISOR_SYSTEM_PROMPT}

Available agents:
{chr(10).join(f"- {k}: {v}" for k, v in WORKER_DESCRIPTIONS.items())}

Based on the conversation, decide which agent should handle this request. 
Respond with exactly one of: {', '.join(options)}

Choose:
- Genie: for structured data queries, database questions
- Retrieval: for document search, general knowledge questions  
- PubMed: for clinical trials, research studies, scientific literature
- FINISH: if sufficient information has been gathered"""

        routing_messages = [{"role": "system", "content": decision_prompt}] + messages
        routing_messages.append({
            "role": "user",
            "content": "Which agent should handle this request? Respond with only: Genie, Retrieval, PubMed, or FINISH"
        })
        
        try:
            response = llm.invoke(routing_messages)
            response_text = response.content.strip() if response.content else ""
            
            # Parse response
            for option in options:
                if option.lower() == response_text.lower().strip():
                    return option
                if option.lower() in response_text.lower():
                    return option
            
            return "FINISH"  # Default fallback
            
        except Exception as e:
            print(f"Supervisor error: {e}")
            return "FINISH"

    def orchestrate_agents(self, input_messages: list, max_iter: int = None) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Orchestrate multiple agents to handle the request."""
        max_iter = max_iter or MAX_ITERATIONS
        messages = input_messages.copy()
        
        # Get the user query
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        
        agent_responses = []
        
        for iteration in range(max_iter):
            #print(f"Iteration {iteration + 1}")
            
            # Get supervisor decision
            next_agent = self.supervisor_decision(messages)
            #print(f"Supervisor decision: {next_agent}")
            
            if next_agent == "FINISH" or next_agent not in self._agents_dict:
                break
            
            # Avoid loops - if we just used this agent, finish
            if agent_responses and agent_responses[-1]["agent"] == next_agent:
                #print(f"Avoiding loop with {next_agent}, finishing")
                break
            
            # Execute the selected agent
            agent_info = self._agents_dict[next_agent]
            result = agent_info.exec_fn(user_query, messages)
            
            # Store agent response
            agent_response = {
                "agent": next_agent,
                "content": result,
                "call_id": f"call_{len(agent_responses) + 1}",
                "id": f"fc_{len(agent_responses) + 1}"
            }
            agent_responses.append(agent_response)
            
            # Yield the agent call
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_function_call_item(
                    id=agent_response["id"],
                    call_id=agent_response["call_id"],
                    name=next_agent,
                    arguments=f'{{"query": "{user_query}"}}'
                )
            )
            
            # Yield the agent output
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done", 
                item=self.create_function_call_output_item(
                    call_id=agent_response["call_id"],
                    output=result
                )
            )
            
            # Add agent response to conversation
            messages.append({
                "role": "assistant",
                "content": result,
                "name": next_agent
            })
        
        # Generate final comprehensive answer
        if agent_responses:
            context = "\n\nInformation gathered by agents:\n"
            for response in agent_responses:
                if not response["content"].startswith("[") and "error:" not in response["content"].lower():
                    context += f"\n{response['agent']}: {response['content']}\n"
            
            final_prompt = f"""{FINAL_ANSWER_PROMPT}

Original question: {user_query}
{context}

Please synthesize this information into a clear, helpful response that directly addresses the user's question."""
            
            try:
                response = llm.invoke([{"role": "system", "content": final_prompt}])
                final_text = response.content if response.content else "I apologize, but I wasn't able to generate a proper response."
            except Exception as e:
                final_text = f"Error generating final response: {str(e)}"
        else:
            final_text = "I wasn't able to gather sufficient information to answer your question."
        
        # Yield final answer
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                text=final_text,
                id="final_response"
            )
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Main predict method."""
        input_messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in request.input]
        
        outputs = [
            event.item 
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        
        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs or {}
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming predict method."""
        input_messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in request.input]
        
        # Add system message if not present
        if not any(msg.get("role") == "system" for msg in input_messages):
            input_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        yield from self.orchestrate_agents(input_messages)


# --- Final Setup ---
mlflow.langchain.autolog()
AGENT = MultiAgentResponsesAgent()
mlflow.models.set_model(AGENT)