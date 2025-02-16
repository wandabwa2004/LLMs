# Equipment Maintenance Planning with Deepseek and CrewAI

## Overview

This repository contains a comprehensive framework for optimizing equipment maintenance planning using mathematical optimization combined with qualitative reasoning. The system integrates:
- **Data Pipeline and Optimization Module:** Ingests equipment data, performs local analysis, and optimizes the maintenance schedule using PuLP.
- **Agentic Reasoning with Deepseek LLM:** Uses specialized agents to generate strategic advice (financial, risk, and executive summaries) based on optimization results.
- **Crew Orchestration:** Employs the CrewAI framework to orchestrate agents and tasks.

The framework leverages Deepseek—a local reasoning LLM running in Ollama—to provide advanced qualitative insights that complement the quantitative optimization.

## Backstory

During a session with a senior manager at a power generation company in NSW, I was presented with a challenging scenario: using equipment maintenance data to derive insights that could inform the strategic direction of the company. At the time, the problem seemed daunting. Two years later, I revisited this scenario from an agentic AI perspective, which led me to develop this integrated framework combining optimization with strategic reasoning.

## File Structure

- `main.py`: Main entry point for the application.
- `maintenance_pipeline.py`: Contains functions for data ingestion, local analysis, and optimization using PuLP.
- `strategy_agents.py`: Defines specialized agents (Financial Advisor, Risk & Compliance Specialist, Strategic Communicator) and task execution.
- `local_summarizer.py`: Implements functions to summarize the maintenance schedule.
- `ollama_config.yaml`: Configuration file for the Deepseek LLM in Ollama.
- `prompts/`: Directory containing YAML prompt configuration files:
  - `communicator_agent.yaml`
  - `financial_agent.yaml`
  - `risk_agent.yaml`

## Environment Setup

### Prerequisites

- **Python 3.9 or later**
- **CrewAI:** An open-source framework for building and orchestrating AI agents. Installation instructions are available at [CrewAI Open Source](https://www.crewai.com/open-source).
- **Ollama:** Required to run Deepseek locally. Visit [Ollama](https://ollama.ai) for installation instructions.

### Creating a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Run the following commands:

```bash
python3 -m venv crewai-env
source crewai-env/bin/activate    # On Windows: crewai-env\Scripts\activate
