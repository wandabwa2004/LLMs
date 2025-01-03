# main.py

import sys
import json
import pandas as pd

# Import your pipeline and agents
from maintenance_pipeline import run_pipeline
from strategy_agents import (
    create_financial_agent,
    create_risk_agent,
    create_communicator_agent,
    get_financial_advice,
    get_risk_advice,
    get_communicator_report
)

# Import local summarizer
from local_summarizer import summarize_maintenance_plan, generate_text_summary

def main(file_path: str, budget: float, openai_api_key: str):
    """
    1) Run the local pipeline to get analysis and plan.
    2) Summarize the large 'maintenance_schedule' so we don't exceed GPT-4 token limits.
    3) Pass the short summary text to Financial + Risk agents.
    4) Combine both advices via the Communicator agent.
    """
    # 1) Local pipeline
    pipeline_results = run_pipeline(file_path, budget)
    analysis_summary = pipeline_results["analysis_summary"]
    plan_summary = pipeline_results["plan_summary"]

    # Save the raw plan_summary to disk for debugging - choice  
    # with open('plan_summary.json', 'w') as file:
    #     json.dump(plan_summary, file, indent=2)

    # Convert plan_summary["maintenance_schedule"] to a DataFrame
    schedule_list = plan_summary["maintenance_schedule"]
    schedule_df = pd.DataFrame(schedule_list)

    # 2) Summarize schedule locally
    summary_dict = summarize_maintenance_plan(schedule_df, top_n=10)
    summary_text = generate_text_summary(summary_dict)

    # 3) Create the agents
    financial_agent = create_financial_agent(openai_api_key)
    risk_agent = create_risk_agent(openai_api_key)
    communicator_agent = create_communicator_agent(openai_api_key)

    # 4) Gather advice from Financial + Risk -- pass only the summarized text
    print("=== FINANCIAL ADVICE ===")
    fin_advice = get_financial_advice(financial_agent, {"summary": summary_text})
    print(fin_advice)

    print("\n=== RISK & COMPLIANCE ADVICE ===")
    risk_advice = get_risk_advice(risk_agent, analysis_summary, {"summary": summary_text})
    print(risk_advice)

    # 5) Communicator merges final insights
    print("\n=== COMMUNICATOR SUMMARY ===")
    final_summary = get_communicator_report(communicator_agent, fin_advice, risk_advice)
    print(final_summary)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python main.py <csv_file_path> <budget> <openai_api_key>")
        sys.exit(1)

    csv_file = sys.argv[1]
    budget_value = float(sys.argv[2])
    openai_key = sys.argv[3]

    main(csv_file, budget_value, openai_key)