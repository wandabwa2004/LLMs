
# Predictive Maintenance Optimization and Advisory System

## Overview

This repository contains a **predictive maintenance** workflow that:
1. **Ingests** equipment data (e.g., transformers, poles, insulators, etc.).
2. **Analyzes** it locally (summary statistics, risk measures).
3. **Runs an optimization** step (Mixed-Integer Programming, MIP) to decide which equipment to maintain under a given budget.
4. **Summarizes** the large maintenance schedule to avoid token-limit issues.
5. **Generates** high-level **financial** and **risk** recommendations using **LLM-based agents**.
6. **Combines** these advisors’ outputs into a coherent **executive summary** via a **communicator** agent.

The **key** is that all heavy data handling (thousands of items) happens **locally** with Python/pandas, and **only** small textual summaries are passed to GPT-4. This prevents the “Request too large” errors, while still leveraging LLM-based strategic advice.

---

## Project Files

1. **`maintenance_pipeline.py`**  
   - **Data Ingestion**:  
     - `ingest_data(file_path: str) -> pd.DataFrame` reads CSV data into a pandas DataFrame.  
   - **Local Analysis**:  
     - `local_analysis(df: pd.DataFrame) -> dict` returns high-level statistics (e.g., average failure probability, how many items are high risk).  
   - **MIP Optimization** (`MaintenanceOptimizer`):  
     - Uses **PuLP** to minimize overall failure risk under a fixed **budget** (and optional **manpower** constraints).  
     - Returns a new DataFrame with a binary column `maintain` (0/1), plus `solution_status` and `optimized_risk`.  
   - **`run_pipeline(file_path, budget) -> dict`**  
     - Ties everything together: ingestion, analysis, optimization.  
     - Returns a `dict` containing `analysis_summary` and `plan_summary` (which in turn contains `maintenance_schedule`).

2. **`strategy_agents.py`**  
   - Defines **three** LLM-based agents:  
     1. **Financial Agent**: Provides **cost/budget** guidance.  
     2. **Risk Agent**: Provides **compliance/safety** guidance.  
     3. **Communicator Agent**: Synthesizes the outputs from the first two agents into an **executive summary**.  
   - Each agent is initialized with **LangChain** (`initialize_agent`) using **ChatOpenAI** (GPT-4).  
   - Tools are empty (`tools=[]`), because we don’t need the agent to call external functions.  
   - The functions `get_financial_advice`, `get_risk_advice`, and `get_communicator_report` produce prompt strings and invoke the agents’ `.run(...)` method.

3. **`local_summarizer.py`**  
   - **Summarizes** the large maintenance schedule (often thousands of lines) to **avoid token-limit** issues.  
   - `summarize_maintenance_plan(schedule_df, top_n=10) -> dict`:  
     - Aggregates overall stats (# of items, how many are maintained, total cost for maintained items, etc.).  
     - Groups by `equipment_type` for distribution.  
     - Extracts the **top-N** costliest items for a small sample.  
   - `generate_text_summary(summary_dict) -> str`:  
     - Converts that dictionary into a concise string for the LLM prompt.

4. **`main.py`**  
   - The main **entry point** for the pipeline.  
   - 1) **Runs** the local pipeline (`run_pipeline`) → obtains `analysis_summary` + `plan_summary`.  
   - 2) **Converts** `plan_summary["maintenance_schedule"]` into a DataFrame.  
   - 3) **Calls** `summarize_maintenance_plan(...)` + `generate_text_summary(...)` → small text snippet.  
   - 4) **Initializes** the **Financial** + **Risk** + **Communicator** agents.  
   - 5) **Gathers** advice from financial/risk agents (passing only the short summary).  
   - 6) **Combines** them via the communicator agent to produce a **final** executive summary.  

---

## Architecture & Data Flow

1. **Data** (CSV) → `maintenance_pipeline.py` → returns `analysis_summary` (stats) + `plan_summary` (the raw schedule).  
2. **Large Schedule** → *Locally Summarized* in `local_summarizer.py` → yields a brief dictionary (stats, top items).  
3. **Brief Dictionary** → `generate_text_summary(...)` → a short text string.  
4. **Agents** in `strategy_agents.py`:  
   - **Financial Agent**: Focus on cost optimization.  
   - **Risk Agent**: Focus on compliance/safety/regulatory issues.  
   - **Communicator Agent**: Merges both advices into one strategic write-up.  
5. **LLM** (GPT-4) sees only short textual summaries, preventing 429 token-limit errors.

---

## Mixed-Integer Programming (MIP) Details

- Implemented with **PuLP** in the `MaintenanceOptimizer` class.  
- **Decision Variables**: A binary \( x_i \) for each item \( i \), meaning “maintain” (1) or “do not maintain” (0).  
- **Objective**: Minimize expected risk \(\sum_i [ p_i \times \text{risk_impact}_i \times (1 - x_i) ]\) plus optionally scaled by \(\alpha\) if maintained.  
- **Budget Constraint**: \(\sum_i (x_i \times \text{cost}_i) \leq \text{budget}\).  
- **Optional**: Manpower constraint \(\sum_i (x_i \times \text{labor}_i) \leq \text{manpower_limit}\).  

The solver returns an **optimal** or **feasible** solution, marking each item with `maintain=1/0` and computing the **“optimized_risk.”**

---

## Installation

1. **Clone** this repository:

   ```
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo```

2. Install Python dependencies:
   `pip install -r requirements.txt`
   Ensure you have Python 3.8+ and a valid OpenAI account/API key for GPT-4 usage.

**Usage**

1. Obtain an OpenAI API key from [OpenAI](https://platform.openai.com/).
2. Prepare your CSV data (e.g., equipment_data.csv) with columns like:
- equipment_id, equipment_type, age, cost, failure_probability, etc. This can be customized for your usecase.  
3. Run the main script:
   `python main.py <csv_file_path> <budget> <openai_api_key>`
  For example:
  `python main.py equipment_data.csv 500000 sk-XXXXXX`
4. Results:
  - A plan_summary.json is saved locally, containing the raw schedule + cost/risk.
  - The script prints:
    - Financial Advice (from the `create_financial_agent`).
    - Risk & Compliance Advice (from the `create_risk_agent`).  
    - Communicator Summary merges the above into a final, high-level strategy.

**Extending the Project**
1. Scenario Analysis: You can rerun run_pipeline with different budgets or constraints, then again call the agents for new advice.
2. Charts/Dashboards: Integrate with a plotting library or Streamlit to visualize the maintenance schedule or risk distribution.
3 Further Constraints: In MaintenanceOptimizer, add constraints for capacity (e.g., “max X items per month”), region constraints, or must-maintain sets of equipment.
4. More Agents: E.g., a “Scenario Explorer Agent,” or “Final Presenter Agent” specialized in generating slide decks, etc.

  


