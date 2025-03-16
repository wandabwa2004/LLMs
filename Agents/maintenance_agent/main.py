import sys
import json
import pandas as pd
import os

# Import your pipeline and agent functions
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


def main(file_path: str, budget: float):
    """
    1) Run the local pipeline to get analysis and plan.
    2) Summarize the large 'maintenance_schedule' so we don't exceed token limits.
    3) Pass the summary to Financial and Risk agents.
    4) Combine their advice using the Communicator agent.
    """
    # 1) Run the local pipeline
    pipeline_results = run_pipeline(file_path, budget)
    analysis_summary = pipeline_results["analysis_summary"]
    plan_summary = pipeline_results["plan_summary"]
    print(analysis_summary)
    print("------------------------")
    print(plan_summary)

    # Convert maintenance_schedule to a DataFrame and summarize it
    schedule_list = plan_summary["maintenance_schedule"]
    schedule_df = pd.DataFrame(schedule_list)
    summary_dict = summarize_maintenance_plan(schedule_df, top_n=10) #top 10 items due  for maintenance
    summary_text = generate_text_summary(summary_dict)
    print(summary_text)


    # 3) Create the agents
    financial_agent = create_financial_agent()
    risk_agent = create_risk_agent()
    communicator_agent = create_communicator_agent()

    # 4) Gather advice
    print("=== FINANCIAL ADVICE ===")
    fin_advice = get_financial_advice(financial_agent, {"summary": summary_text})
    print(fin_advice)

    print("\n=== RISK & COMPLIANCE ADVICE ===")
    risk_advice = get_risk_advice(risk_agent, analysis_summary, {"summary": summary_text})
    print(risk_advice)

    print("\n=== COMMUNICATOR SUMMARY ===")
    final_summary = get_communicator_report(communicator_agent, fin_advice, risk_advice)
    print(final_summary)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <csv_file_path> <budget>")
        sys.exit(1)

    csv_file = sys.argv[1]
    budget_value = float(sys.argv[2])
    main(csv_file, budget_value)