# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy System Prompt Optimization with MIPROv2
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook demonstrates how to use DSPy's MIPROv2 optimizer to automatically improve system prompts for RAG (Retrieval-Augmented Generation) applications. The optimization process uses training examples to iteratively refine prompts, resulting in better performance on your specific use case.
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC 1. **Sets up DSPy with Databricks LLM endpoints** - Configures the DSPy framework to work with your Databricks model serving endpoints
# MAGIC 2. **Integrates with Databricks Vector Search** - Connects to your existing vector search index for document retrieval
# MAGIC 3. **Defines a RAG system** - Creates a modular RAG pipeline using DSPy signatures and modules
# MAGIC 4. **Optimizes prompts automatically** - Uses MIPROv2 to generate better prompt instructions based on your training data
# MAGIC 5. **Evaluates performance** - Compares baseline vs optimized performance and saves the results
# MAGIC 6. **Exports optimized configuration** - Saves the improved prompt to a configuration file for production use
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with LLM endpoint configured
# MAGIC - Vector search index with your product documentation
# MAGIC - `agent_config.yaml` file with configuration parameters
# MAGIC
# MAGIC ## Expected Outcome
# MAGIC An optimized system prompt saved to `agent_config_optimized.yaml` that performs better on your specific use case

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Required Dependencies
# MAGIC
# MAGIC Install the necessary packages for DSPy optimization and Databricks integration.

# COMMAND ----------

# MAGIC %pip install dspy pyyaml databricks-langchain

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Import Libraries and Configure MLflow Tracking
# MAGIC
# MAGIC Import required libraries and enable MLflow autologging to track the optimization process.

# COMMAND ----------

import dspy
import yaml
from databricks_langchain import VectorSearchRetrieverTool
import mlflow
# Enable MLflow autolog for DSPy
mlflow.dspy.autolog()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load Configuration and Setup DSPy Environment
# MAGIC
# MAGIC Load your configuration file and set up the DSPy environment. Your `agent_config.yaml` should contain:
# MAGIC - **LLM_ENDPOINT_NAME**: Name of your Databricks model serving endpoint
# MAGIC - **VECTOR_CATALOG/SCHEMA/INDEX_NAME**: Vector search index location
# MAGIC - **VECTOR_SEARCH_ENDPOINT**: Vector search endpoint name
# MAGIC - **QUERY_TYPE**: Type of vector search query (e.g., "similarity")
# MAGIC - **NUM_RESULTS**: Number of documents to retrieve per query
# MAGIC - **VECTOR_COLUMNS**: Columns to return from vector search
# MAGIC - **TEMPERATURE**: Model temperature for generation

# COMMAND ----------

# Load config
with open("../config/agent_config.yaml", "r") as f:  # Updated path
    config = yaml.safe_load(f)

# Configure DSPy
lm = dspy.LM(model=f"databricks/{config['LLM_ENDPOINT_NAME']}", max_tokens=500)
dspy.settings.configure(lm=lm)

# Setup vector search
vector_search_tool = VectorSearchRetrieverTool(
    index_name=f"{config['VECTOR_CATALOG']}.{config['VECTOR_SCHEMA']}.{config['VECTOR_INDEX_NAME']}",
    endpoint_name=config['VECTOR_SEARCH_ENDPOINT'],
    query_type=config['QUERY_TYPE'],
    num_results=config['NUM_RESULTS'],
    columns=config['VECTOR_COLUMNS']
)

def search(question, k=3):
    return vector_search_tool.invoke(question)

print(f"Using LLM: {config['LLM_ENDPOINT_NAME']}")
print(f"Vector index: {config['VECTOR_CATALOG']}.{config['VECTOR_SCHEMA']}.{config['VECTOR_INDEX_NAME']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Define RAG System Architecture
# MAGIC
# MAGIC Create the RAG system using DSPy's modular approach:
# MAGIC - **Signature**: Defines the input/output specification and task description
# MAGIC - **Module**: Combines retrieval and generation into a cohesive system
# MAGIC
# MAGIC The signature will be automatically optimized by MIPROv2 to improve performance.

# COMMAND ----------

# RAG signature and module
class ProductRAGSignature(dspy.Signature):
    """Product documentation RAG that retrieves and responds to questions."""
    question = dspy.InputField(desc="User's question about products")
    response = dspy.OutputField(desc="Accurate answer based on retrieved context")

class ProductRAG(dspy.Module):
    def __init__(self, num_docs=3):
        super().__init__()
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought(ProductRAGSignature)
        
    def forward(self, question: str):
        context = search(question, k=self.num_docs)
        result = self.respond(question=question)
        result.context = context
        return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Prepare Training and Validation Data
# MAGIC
# MAGIC **CRITICAL: Replace these examples with your actual use case data.**
# MAGIC
# MAGIC ### Guidelines for High-Quality Training Data:
# MAGIC - **Use 10-50 examples** that represent typical user questions
# MAGIC - **Include diverse question types** (factual, comparison, troubleshooting, etc.)
# MAGIC - **Ensure accuracy** - responses should reflect your actual documentation
# MAGIC - **Be specific** - questions should be domain-specific, not generic
# MAGIC - **Validation set** should be different from training examples
# MAGIC
# MAGIC ### Training Data Best Practices:
# MAGIC - Cover different product categories and use cases
# MAGIC - Include both simple and complex questions
# MAGIC - Represent the actual language your users employ
# MAGIC - Ensure responses are complete and helpful

# COMMAND ----------

import json
import dspy

# Or keep it in notebooks if it's test data
with open('../data/eval_dataset.json', 'r') as f:
    eval_data = json.load(f)

# Convert eval data to DSPy examples
eval_examples = []
for item in eval_data:
    eval_examples.append(
        dspy.Example(
            question=item["question"],
            response=item["answer"]
        ).with_inputs("question")
    )

print(f"Evaluation examples: {len(eval_examples)}")

# Optional: Split eval data into train/val if needed
# Using 70/30 split for better balance
split_point = int(len(eval_examples) * 0.7)

train_examples_from_eval = eval_examples[:split_point]  # First 7 examples
val_examples_from_eval = eval_examples[split_point:]    # Last 3 examples

print(f"Training examples from eval data: {len(train_examples_from_eval)}")
print(f"Validation examples from eval data: {len(val_examples_from_eval)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Define Evaluation Metric
# MAGIC
# MAGIC The evaluation metric determines how we measure prompt performance. This simple word overlap metric 
# MAGIC can be replaced with more sophisticated metrics like semantic similarity or domain-specific scoring.

# COMMAND ----------

# Accuracy metric
def accuracy_metric(example, pred, trace=None):
    expected = set(example.response.lower().split())
    predicted = set(pred.response.lower().split())
    if len(expected) == 0:
        return 0.0
    return len(expected.intersection(predicted)) / len(expected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Run MIPROv2 Optimization
# MAGIC
# MAGIC MIPROv2 (Multi-Prompt Instruction Proposal Optimizer v2) automatically generates and tests 
# MAGIC multiple prompt variations to find the best performing version.
# MAGIC
# MAGIC ### How MIPROv2 Works:
# MAGIC 1. **Generate Candidates**: Creates multiple prompt instruction variations
# MAGIC 2. **Evaluate Performance**: Tests each prompt on your training data
# MAGIC 3. **Select Best**: Chooses the highest-performing prompt based on your metric
# MAGIC 4. **Iterative Refinement**: Continues improving through multiple optimization rounds
# MAGIC
# MAGIC ### Optimization Process:
# MAGIC - Uses two separate models: one for prompt generation, one for task execution
# MAGIC - Tests prompts against your training examples
# MAGIC - Selects the prompt that maximizes your evaluation metric
# MAGIC - Can take 5-15 minutes depending on data size and model speed

# COMMAND ----------

# MIPROv2 optimization
print("Starting MIPROv2 optimization...")

optimizer = dspy.MIPROv2(
    prompt_model=dspy.LM(model=f"databricks/{config['LLM_ENDPOINT_NAME']}", max_tokens=1000),
    task_model=dspy.LM(model=f"databricks/{config['LLM_ENDPOINT_NAME']}", max_tokens=500),
    metric=accuracy_metric,
    #num_candidates=20,
    init_temperature=config['TEMPERATURE'],
)

optimized_rag = optimizer.compile(
    student=ProductRAG(),
    trainset=train_examples_from_eval,
    valset=val_examples_from_eval,
)

print("Optimization completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Extract and Save Optimized Prompt
# MAGIC
# MAGIC Extract the optimized prompt instructions and create a production-ready configuration file.
# MAGIC The optimized prompt will contain improved instructions that should perform better on your specific use case.

# COMMAND ----------

# Extract optimized prompt
try:
    optimized_instructions = optimized_rag.respond.predict.signature.instructions
    print("OPTIMIZED INSTRUCTIONS:")
    print("-" * 60)
    print(optimized_instructions)
    print("-" * 60)
    
    # Create final system prompt
    final_system_prompt = f"""{optimized_instructions} Respond truthfully and do not hallucinate."""
    
    # Save optimized config
    optimized_config = config.copy()
    optimized_config['SYSTEM_PROMPT'] = final_system_prompt
    
    with open("../config/agent_config_optimized.yaml", 'w') as f:  # Updated path
        yaml.dump(optimized_config, f, default_flow_style=False, width=1000)
    
    print(f"Optimized config saved: {len(final_system_prompt)} characters")
    print(f"File: agent_config_optimized.yaml")
    
except Exception as e:
    print(f"Error extracting prompt: {e}")
    optimized_instructions = None

# COMMAND ----------

if optimized_instructions:
    print("Performance test:")
    
    baseline_rag = ProductRAG()
    test_example = val_examples_from_eval[0]
    
    # Test both models
    baseline_pred = baseline_rag(question=test_example.question)
    baseline_score = accuracy_metric(test_example, baseline_pred)
    
    optimized_pred = optimized_rag(question=test_example.question)
    optimized_score = accuracy_metric(test_example, optimized_pred)
    
    print(f"Baseline: {baseline_score:.3f}")
    print(f"Optimized: {optimized_score:.3f}")
    print(f"Improvement: +{optimized_score - baseline_score:.3f}")
    
    print(f"\nTest question: {test_example.question}")
    print(f"Expected: {test_example.response}")
    print(f"Baseline: {baseline_pred.response}")
    print(f"Optimized: {optimized_pred.response}")
    
    print("\nUse agent_config_optimized.yaml in your agent")
