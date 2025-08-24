# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Config Generator
# MAGIC
# MAGIC
# MAGIC **Purpose:**
# MAGIC - Creates a consistent configuration file with correct UC paths
# MAGIC - Maintains a single source of truth for catalog and schema names
# MAGIC - Ensures agent.py has the correct configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Widgets and Parameters

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Widget setup
dbutils.widgets.text("catalog", "agents_workshop")
dbutils.widgets.text("schema", "agents")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("output_path", "../config/agent_config.yaml")  # Updated path

# Get parameters from widgets
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
environment = dbutils.widgets.get("environment")
output_path = dbutils.widgets.get("output_path")

print(f"Generating config with catalog: {catalog}, schema: {schema}, environment: {environment}")
print(f"Output path: {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Configuration

# COMMAND ----------

# Create the config dictionary
config = {
    # Agent Configuration
    
    # LLM Model Endpoint Configuration
    "LLM_ENDPOINT_NAME": "databricks-llama-4-maverick", #"databricks-meta-llama-3-3-70b-instruct",
    
    # Vector Search Configuration
    "VECTOR_CATALOG": catalog,
    "VECTOR_SCHEMA": schema,
    "VECTOR_INDEX_NAME": "clinical_studies_vs",
    "VECTOR_SEARCH_ENDPOINT": "one-env-shared-endpoint-10",
    "QUERY_TYPE": "hybrid",  # Adding default query type, other option is ANN
    "NUM_RESULTS": 3,        # Adding default num_results
    "TEMPERATURE": 0.3,
    "VECTOR_COLUMNS": ["nct_id", "official_title","detailed_description"],  # Configurable vector search columns

    # Optional: Add additional vector search parameters
    # "VECTOR_SEARCH_FILTERS": "",
    
    # Unity Catalog Configuration
    "MODEL_NAME": "CLINICAL_STUDIES_AGENT",
    
    # Genie Configuration
    "GENIE_SPACE_ID": "<enter Genie space ID>",

    # PubMed Configuration
    "PUBMED_MAX_DOCS": 5,
    
    # Supervisor Configuration
    "MAX_ITERATIONS": 3,
    "SUPERVISOR_TEMPERATURE": 0.1,
    "SUPERVISOR_MAX_TOKENS": 100,
    "DEBUG_MODE": True,
    
    # Worker Agent Descriptions
    "WORKER_DESCRIPTIONS": {
        "Genie": "Structured data expert using clinical trials data details table. When given ntc_id values, it uses them to query related structured data about location, outcomes, start and end dates",
        "Retrieval": "Searches internal clinical trials for broad information. Extracts ntc_id and official_title metadata for structured data follow-up.",
        "PubMed": "Searches PubMed biomedical literature database for published research papers, reviews, and clinical studies. Best for finding peer-reviewed scientific evidence and medical research."
    },
    
    # System Prompts
    "SYSTEM_PROMPT": """You are a clinical research assistant on clinical studies and research documentation, it contains information about clinical trials, including their eligibility criterias, interventions and primary outcomes. Respond truthfully and do not hallucinate.""",
    
    "SUPERVISOR_SYSTEM_PROMPT": "You are a supervisor agent. Choose one of the workers or finish based on the user's query and previous interactions. Consider the strengths of each agent when making your decision.",
    
    "PUBMED_SYSTEM_PROMPT": """You are a biomedical research assistant. Based on the PubMed search results provided, answer the user's question with accurate, evidence-based information. Always cite the sources when possible and mention that the information comes from PubMed literature.""",
    
    "FINAL_ANSWER_PROMPT": "Using only the content in the messages, respond to the previous user question using the answer given by the other assistant messages.",
    
    # Agent Names (for consistency)
    "AGENT_NAMES": {
        "genie": "Genie",
        "retrieval": "RetrievalAgent", 
        "pubmed": "PubMedAgent",
        "supervisor": "SupervisorAgent",
        "final_answer": "FinalAnswer"
    },
    
    # Error Messages
    "ERROR_MESSAGES": {
        "no_message": "did not return any message.",
        "empty_message": "returned an empty message.",
        "agent_error": "encountered an error",
        "no_response": "No response generated.",
        "pubmed_no_results": "No PubMed results found.",
        "pubmed_error": "PubMed search encountered an error"
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Configuration File

# COMMAND ----------

from pathlib import Path
import yaml

# Ensure parent directories exist
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# Write to output file
with open(output_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Config file generated at: {output_path}")

# Display the generated config
with open(output_path, 'r') as f:
    config_content = f.read()
    
print("Generated config content:")
print("-------------------------")
print(config_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Completion
# MAGIC
# MAGIC The configuration file has been generated successfully. This will be used by:
# MAGIC
# MAGIC 1. The agent.py script to initialize the RAG components
# MAGIC 2. Other notebooks in the workflow that need consistent configuration
# MAGIC
# MAGIC The generated file includes both the separated components (catalog, schema, base name) and the full UC path for backward compatibility.

# COMMAND ----------

# Return success
dbutils.notebook.exit("Config generated successfully")
