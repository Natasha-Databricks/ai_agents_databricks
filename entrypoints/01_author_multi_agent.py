# Databricks notebook source
# MAGIC %md
# MAGIC # Clinical Research Multi-Agent System
# MAGIC
# MAGIC A sophisticated multi-agent system built with LangGraph and MLflow for clinical research data analysis, combining structured clinical trial data, document retrieval, and PubMed literature search capabilities.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This system orchestrates three specialized agents through a supervisor to provide comprehensive clinical research insights:
# MAGIC
# MAGIC - **Genie Agent**: Queries structured clinical trials data using natural language
# MAGIC - **Retrieval Agent**: Searches internal clinical trial documents and extracts metadata
# MAGIC - **PubMed Agent**: Searches biomedical literature for peer-reviewed research
# MAGIC - **Supervisor Agent**: Routes queries to appropriate agents and manages workflow
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC User Query → Supervisor Agent → [Genie | Retrieval | PubMed] → Final Answer
# MAGIC ```
# MAGIC ![agent_arch](../media/agent_arch.png)
# MAGIC
# MAGIC The supervisor intelligently routes queries based on content and iteratively calls agents until a satisfactory answer is reached (max 3 iterations).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow[databricks]>3.0 databricks-langchain databricks-agents uv langgraph pyyaml xmltodict
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import libraries

# COMMAND ----------

# Standrd Libraries
import os
import yaml 
import json
import requests
import time
import pandas as pd 
# Databricks
# from databricks_langchain import VectorSearchRetrieverTool

# MLFlow
import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.types.agent import ChatAgentMessage, ChatContext

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Configuration

# COMMAND ----------

dbutils.widgets.text("schema", "agents")
dbutils.widgets.text("catalog", "agents_workshop")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("config_yml", "../config/agent_config.yaml")  # Updated path

# Get values from widgets
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
environment = dbutils.widgets.get("environment")
config_file = dbutils.widgets.get("config_yml")

# Get user name to customise model registration
user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "").lower()[:35]

print(f"Widget values:")
print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Environment: {environment}")
print(f"Config file: {config_file}")
print(f"User Name: {user_name}")


# Set Secrets
secret_scope_name = "your_secret_scope"
secret_key_name_pat = "PAT_token"
secret_key_name_host = "Genie_host"

os.environ["DATABRICKS_GENIE_PAT"] = dbutils.secrets.get(
    scope=secret_scope_name, key=secret_key_name_pat
)

os.environ["DB_MODEL_SERVING_HOST_URL"] = dbutils.secrets.get(
    scope=secret_scope_name, key=secret_key_name_host
)

assert os.environ["DATABRICKS_GENIE_PAT"] is not None, (
    "The DATABRICKS_GENIE_PAT was not properly set to the PAT secret"
)

# COMMAND ----------

# Load other configuration from YAML if needed
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

for key, value in config.items():
    if key == "SYSTEM_PROMPT":
        print(f"{key}: [System prompt configured]")
    else:
        print(f"{key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5. Setup Config File Path for Agent Import
# MAGIC
# MAGIC Ensure the agent module can find the configuration file by copying it to a known location.

# COMMAND ----------

import shutil
import os

# Copy config file to current directory for reliable access
current_dir = os.getcwd()
config_target = os.path.join(current_dir, "agent_config.yaml")

try:
    shutil.copy2(config_file, config_target)
    print(f"Config file copied to: {config_target}")
except Exception as e:
    print(f"Warning: Could not copy config file: {e}")
    print(f"Agent will try to load from original path: {config_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6. Alternative: Set Environment Variable for Config Path
# MAGIC
# MAGIC Provide an absolute path to the config file via environment variable.

# COMMAND ----------

# Set environment variable for the config file path so agent.py can find it
import os
config_absolute_path = os.path.abspath(config_file)
os.environ["AGENT_CONFIG_PATH"] = config_absolute_path
print(f"Config path set in environment: {config_absolute_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create / Load Parameterized Agent
# MAGIC
# MAGIC Let's create a parameterized agent that reads settings from our configuration file. In this initial example, we load our baseline Agent from `agent.py`.

# COMMAND ----------

import sys
sys.path.append('../src/')
from agent_modules.agent import AGENT

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test the loaded Agent, DEVELOPMENT ONLY
# MAGIC
# MAGIC Let's test our agent with sample questions to validate its behavior and check traces.

# COMMAND ----------

test_question = "Find recent publications on rare diseases "
example_input = {"messages": [{"role": "user", "content": test_question}]}
AGENT.predict(example_input)

# COMMAND ----------

test_question = "Any internal clinical trials on rare diseases? What were the outcomes of those using the structured database"
example_input = {"messages": [{"role": "user", "content": test_question}]}
AGENT.predict(example_input)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate the Agent with ground truth
# MAGIC
# MAGIC Let's test our agent with sample questions before registering it. Please inspect the traces as they will contain information about the LLM judge outcome. We recommend you also click through the traces to understand each component of data flow within the agent and how the input results in the output. **Have you noticed the judge icon 👩🏻‍⚖️ which was added as opposed to a simple `agent.predict()`?**

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

with open("../data/eval_dataset.json", "r") as file:
    eval_data = json.load(file)

eval_dataset = pd.DataFrame([
    {
        "inputs": {
            "inputs": {"messages": [{"role": "user", "content": item["question"]}]}
        },
        "expected_response": item["answer"]
    }
    for item in eval_data
])

# COMMAND ----------

def predict_fn(inputs):
    return AGENT.predict(inputs)

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[RelevanceToQuery(), Safety()],
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log Agent as MLflow Model with Configuration & Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model()`.

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
#    DatabricksFunction
#   DatabricksSQLWarehouse,
#   DatabricksTable,
)
from pkg_resources import get_distribution

# Load configuration from YAML 
with open(config_file, "r") as f:     
    config_dict = yaml.safe_load(f)  


# Add resources for automatic passthrough
vs_index_name = f"{config_dict['VECTOR_CATALOG']}.{config_dict['VECTOR_SCHEMA']}.{config_dict['VECTOR_INDEX_NAME']}"

resources = [#DatabricksServingEndpoint(endpoint_name=config_dict["LLM_ENDPOINT_NAME"]),
                 #DatabricksGenieSpace(genie_space_id=config_dict["GENIE_SPACE_ID"]),
                 DatabricksVectorSearchIndex(index_name=vs_index_name)
]


# Log the model with the config dictionary
with mlflow.start_run() as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../src/agent_modules/agent.py",  # Path to agent.py file
        model_config=config_dict,  # Pass config as dictionary
        pip_requirements=[
            f"mlflow=={get_distribution('mlflow').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-agents=={get_distribution('databricks-agents').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"pyyaml=={get_distribution('pyyaml').version}",
            f"xmltodict=={get_distribution('xmltodict').version}",
        ],
        resources=resources,
        input_example=example_input
    )

print(f"Model and configuration logged with run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Pre-deployment Agent Validation

# COMMAND ----------

import mlflow
import sys
from mlflow.types.agent import ChatAgentMessage
import uuid


mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Register the Model to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

UC_MODEL_NAME = f"{catalog}.{schema}.{config['MODEL_NAME']}_{user_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Set production alias for prod version tracking

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Set alias for the model version
client = MlflowClient()
client.set_registered_model_alias(
    name=UC_MODEL_NAME,
    alias="Production",
    version=uc_registered_model_info.version
)

print(f"Model {UC_MODEL_NAME} version {uc_registered_model_info.version} registered and transitioned to Production alias")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Deploy the Agent to Mosaic Model Serving
# MAGIC
# MAGIC **!!!! Note: Uncomment the following code block if you want to test deployment.!!!**
# MAGIC
# MAGIC > **Warning: It is possible that the workspace will throw an error indicating that too many endpoints were created.**

# COMMAND ----------

# from databricks import agents

# # Deploy the agent
# deployment = agents.deploy(
#     UC_MODEL_NAME,
#     uc_registered_model_info.version,
#     llm_config={
#         "llm_parameters": {
#            "enable_safety_filter": True,
#         }},
#     tags={"endpointSource": "agent_chatbot"},
#     environment_vars={
#         "DATABRICKS_GENIE_PAT": f"{{{{secrets/{secret_scope_name}/{secret_key_name_pat}}}}}",
#         "DB_MODEL_SERVING_HOST_URL": f"{{{{secrets/{secret_scope_name}/{secret_key_name_host}}}}}",

#     },
# )

# print(f"Agent deployed: {deployment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Poll the Deployed Agent and Send Example Request

# COMMAND ----------

# import time
# import json
# import requests
# import os
# import sys
# from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
# from databricks.sdk import WorkspaceClient

# # Initialize the Databricks SDK client
# w = WorkspaceClient()

# # Maximum wait time in seconds (30 minutes)
# max_wait_time = 1800
# start_time = time.time()
# wait_interval = 30  # Check every 30 seconds

# print(f"Waiting for deployment {deployment.endpoint_name} to be ready...")

# while True:
#     # Check if we've exceeded the maximum wait time
#     current_time = time.time()
#     elapsed_time = current_time - start_time
#     elapsed_minutes = elapsed_time / 60
    
#     if elapsed_time > max_wait_time:
#         print(f"Exceeded maximum wait time of {max_wait_time/60:.1f} minutes. Continuing anyway...")
#         break
    
#     try:
#         # Get the current endpoint status
#         endpoint_status = w.serving_endpoints.get(deployment.endpoint_name)
        
#         # Check if the endpoint is ready
#         if (endpoint_status.state.ready == EndpointStateReady.READY and
#             endpoint_status.state.config_update != EndpointStateConfigUpdate.IN_PROGRESS):
            
#             print(f"✅ Deployment ready after {elapsed_minutes:.1f} minutes!")
            
#             # Add a small buffer for full initialization
#             print("Waiting 15 more seconds for full initialization...")
#             time.sleep(15)
#             break
#         else:
#             status_msg = f"Status: {endpoint_status.state.ready}, Config update: {endpoint_status.state.config_update}"
#             print(f"Waiting... ({elapsed_minutes:.1f} min elapsed) - {status_msg}")
#             time.sleep(wait_interval)
            
#     except Exception as e:
#         print(f"Error checking endpoint status: {e}")
#         time.sleep(wait_interval)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC After your agent is deployed, you can:
# MAGIC
# MAGIC 1. Chat with it in AI playground for additional testing
# MAGIC 2. Share it with SMEs in your organization for feedback
# MAGIC 3. Embed it in production applications using the REST API
# MAGIC 4. Configure monitoring and logging to track performance
# MAGIC 5. Set up a CI/CD pipeline for continuous updates
