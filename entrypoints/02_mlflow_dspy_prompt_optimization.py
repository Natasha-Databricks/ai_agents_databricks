# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Prompt Optimization with Databricks LLM Endpoints
# MAGIC
# MAGIC This notebook demonstrates how to use MLflow's `optimize_prompt()` API with DSPy's MIPROv2 algorithm 
# MAGIC to automatically improve prompts for your product documentation system.
# MAGIC
# MAGIC ## Key Benefits
# MAGIC - **Unified Interface**: Access state-of-the-art prompt optimization through MLflow
# MAGIC - **Prompt Management**: Integrate with MLflow Prompt Registry for version control
# MAGIC - **Evaluation**: Comprehensive prompt performance evaluation
# MAGIC - **Databricks Integration**: Works with your existing Databricks LLM endpoints

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Dependencies

# COMMAND ----------

# MAGIC %pip install dspy>=2.6.0 mlflow[databricks]>=3.1 pyyaml databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Configuration and MLflow

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

# COMMAND ----------

import os
import yaml
import json
from typing import Any
import mlflow
from mlflow.genai.scorers import scorer
from mlflow.genai.optimize import OptimizerConfig, LLMParams
from databricks_langchain import VectorSearchRetrieverTool

# Load your existing configuration
with open(config_file, "r") as f:  # Now uses the widget value
    config = yaml.safe_load(f)

# Set MLflow tracking URI for Databricks
mlflow.set_tracking_uri("databricks")

print(f"Using LLM endpoint: {config['LLM_ENDPOINT_NAME']}")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Custom Scorer for Medial Research agent
# MAGIC
# MAGIC Create a scorer that evaluates how well the optimized prompt performs on your Medial Research agent task.

# COMMAND ----------

@scorer
def iou_scorer(expectations: dict[str, Any], outputs: dict[str, Any]) -> float:
    """
    Custom scorer for documentation responses.
    Measures word overlap between expected and actual responses.
    """
    expected_words = set(expectations["answer"].lower().split())
    predicted_words = set(outputs["answer"].lower().split())
    
    if len(expected_words) == 0:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = expected_words.intersection(predicted_words)
    union = expected_words.union(predicted_words)
    
    return len(intersection) / len(union) if len(union) > 0 else 0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Initial Prompt
# MAGIC
# MAGIC Register your baseline product documentation prompt in MLflow.

# COMMAND ----------

# Define the initial prompt template for product documentation
initial_template = f"""{config["SYSTEM_PROMPT"]}

Question: {{{{question}}}}

Return your response in JSON format: {{"answer": "your_response_here"}}"""

print(f"Baseline prompt we want to optimise:\n {initial_template}")

# Register the initial prompt in MLflow

prompt = mlflow.genai.register_prompt(
    name=f"{catalog}.{schema}.baseline_prompt_{user_name}",
    template=initial_template,
)

print(f"Registered prompt: {prompt.uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Training and Evaluation Data
# MAGIC
# MAGIC Use your real product documentation examples for training and evaluation.

# COMMAND ----------

import json

# Load training and evaluation data from JSON file
with open('../data/eval_dataset.json', 'r') as f:
    data = json.load(f)

# Split data into training and evaluation sets
train_data = [
    {
        "inputs": {"question": item["question"]},
        "expectations": {"answer": item["answer"]}
    }
    for item in data[:7]  # Assuming first 7 items for training
]

eval_data = [
    {
        "inputs": {"question": item["question"]},
        "expectations": {"answer": item["answer"]}
    }
    for item in data[7:]  # Assuming remaining items for evaluation
]

print(f"Training examples: {len(train_data)}")
print(f"Evaluation examples: {len(eval_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Prompt Optimization
# MAGIC
# MAGIC Use MLflow's `optimize_prompt()` with your Databricks LLM endpoint to optimize the prompt.

# COMMAND ----------

optimizer_config = OptimizerConfig(
    num_instruction_candidates=10,  # Number of prompt variations to try
    max_few_show_examples=0,       # Determine how many few-shot examples to include
    verbose=True,                  # Show optimization logs
    autolog=True,                  # Automatically log results to MLflow
)

# Define the target LLM parameters for your Databricks endpoint
target_llm_params = LLMParams(
    model_name=f"databricks/{config['LLM_ENDPOINT_NAME']}",
    temperature=config.get('TEMPERATURE', 0.7),
)

print("Starting prompt optimization...")
print(f"Target model: {target_llm_params.model_name}")
print(f"Training examples: {len(train_data)}")
print(f"Evaluation examples: {len(eval_data)}")

# Run the optimization
result = mlflow.genai.optimize_prompt(
    target_llm_params=target_llm_params,
    prompt=prompt,
    train_data=train_data,
    eval_data=eval_data,
    scorers=[iou_scorer],
    optimizer_config=optimizer_config,
)

print(f"Optimization completed!")
print(f"Optimized prompt URI: {result.prompt.uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Optimization Results
# MAGIC
# MAGIC Display the optimized prompt and performance metrics.

# COMMAND ----------

# Load and display the optimized prompt
optimized_prompt = mlflow.genai.load_prompt(result.prompt.uri)

print("OPTIMIZED PROMPT TEMPLATE:")
print("=" * 80)
print(optimized_prompt.template)
print("=" * 80)

# Display performance metrics if available
if hasattr(result, 'metrics'):
    print("\nPERFORMANCE METRICS:")
    for metric_name, metric_value in result.metrics.items():
        print(f"{metric_name}: {metric_value:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Optimized Configuration
# MAGIC
# MAGIC Export the optimized prompt for use in your production agent.

# COMMAND ----------

# Load the optimized prompt template
optimized_prompt = mlflow.genai.load_prompt(result.prompt.uri)

# Create updated configuration with the optimized prompt
optimized_config = config.copy()
optimized_config['SYSTEM_PROMPT'] = optimized_prompt
optimized_config['MLFLOW_PROMPT_URI'] = result.prompt.uri

# Save the optimized configuration
with open("../config/agent_config_mlflow_optimized.yaml", 'w') as f:  # Updated path
    yaml.dump(optimized_config, f, default_flow_style=False, width=1000)

print(f"Optimized configuration saved to: ../config/agent_config_mlflow_optimized.yaml")
print(f"Prompt length: {len(optimized_prompt.template)} characters")
print(f"MLflow prompt URI: {result.prompt.uri}")

# Display the key parts of the optimized config
print("\nKEY CONFIGURATION UPDATES:")
print(f"- SYSTEM_PROMPT: Updated with optimized instructions")
print(f"- MLFLOW_PROMPT_URI: {result.prompt.uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC ### Using Your Optimized Prompt
# MAGIC 1. **Replace your agent configuration**: Use `agent_config_mlflow_optimized.yaml` in your production system
# MAGIC 2. **Monitor performance**: Track the optimized prompt's performance in production
# MAGIC 3. **Version control**: The prompt is automatically versioned in MLflow Prompt Registry
# MAGIC 4. **Iterate**: Re-run optimization as you collect more training examples
# MAGIC
# MAGIC ### MLflow Integration Benefits
# MAGIC - **Prompt Registry**: All prompt versions are tracked and can be retrieved
# MAGIC - **Experiment Tracking**: Optimization runs are logged with metrics and artifacts
# MAGIC - **Model Management**: Easy deployment and rollback of prompt versions
# MAGIC - **Lineage**: Full traceability from data to optimized prompts
# MAGIC
# MAGIC ### Loading the Optimized Prompt in Production
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC # Load the optimized prompt
# MAGIC prompt = mlflow.genai.load_prompt("models:/<prompt_name>/latest")
# MAGIC
# MAGIC # Use in your application
# MAGIC formatted_prompt = prompt.format(question="user's question")
# MAGIC ```
# MAGIC
# MAGIC ### Monitoring and Improvement
# MAGIC - Set up A/B testing between prompt versions
# MAGIC - Collect user feedback on response quality  
# MAGIC - Use MLflow's evaluation features for ongoing assessment
# MAGIC - Re-optimize regularly with new training examples
