# local_summarizer.py

import pandas as pd

def summarize_maintenance_plan(schedule_df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Creates a condensed summary of a large maintenance schedule.
    Returns a small dictionary that won't exceed GPT-4 token limits.
    """
    total_items = len(schedule_df)
    total_maintained = schedule_df["maintain"].sum()
    total_cost_maintained = (schedule_df["cost"] * schedule_df["maintain"]).sum()
    avg_risk = schedule_df["failure_probability"].mean() if "failure_probability" in schedule_df.columns else 0

    # Group by equipment_type for distribution
    group_stats = schedule_df.groupby("equipment_type").agg(
        total_items_by_type=("equipment_id", "count"),
        maintained_by_type=("maintain", "sum"),
        avg_cost_by_type=("cost", "mean"),
        avg_failure_prob_by_type=("failure_probability", "mean")
    ).reset_index()

    # Top-N by cost
    top_n_costly = schedule_df.nlargest(top_n, "cost")

    summary_dict = {
        "overall_stats": {
            "total_equipment": int(total_items),
            "items_maintained": int(total_maintained),
            "total_cost_for_maintained": float(total_cost_maintained),
            "average_failure_probability": float(avg_risk),
        },
        "group_by_equipment_type": group_stats.to_dict(orient="records"),
        "top_n_costly_items": top_n_costly[[
            "equipment_id", "equipment_type", "cost", "maintain", "failure_probability"
        ]].to_dict(orient="records")
    }
    return summary_dict

def generate_text_summary(summary_dict: dict) -> str:
    """
    Converts the summary dictionary into a short text/markdown string
    for an LLM or for direct display.
    """
    overall = summary_dict["overall_stats"]
    group_list = summary_dict["group_by_equipment_type"]
    top_n_list = summary_dict["top_n_costly_items"]

    text_parts = []
    text_parts.append("**High-Level Maintenance Plan Stats:**")
    text_parts.append(f"- Total equipment: {overall['total_equipment']}")
    text_parts.append(f"- Maintained items: {overall['items_maintained']}")
    text_parts.append(f"- Total cost for maintained: ${overall['total_cost_for_maintained']:.2f}")
    text_parts.append(f"- Average failure probability: {overall['average_failure_probability']:.3f}")

    text_parts.append("\n**By Equipment Type:**")
    for row in group_list:
        eq_type = row["equipment_type"]
        eq_count = row["total_items_by_type"]
        eq_maint = row["maintained_by_type"]
        avg_cost = row["avg_cost_by_type"]
        avg_fail = row["avg_failure_prob_by_type"]
        text_parts.append(
            f"- {eq_type}: {eq_count} items, {eq_maint} maintained, "
            f"avg cost=${avg_cost:.1f}, avg fail prob={avg_fail:.3f}"
        )

    text_parts.append("\n**Top-Cost Items (sample):**")
    for item in top_n_list:
        text_parts.append(
            f"- ID {item['equipment_id']}, type={item['equipment_type']}, "
            f"cost=${item['cost']}, maintain={item['maintain']}, "
            f"fail_prob={item['failure_probability']:.3f}"
        )

    text_parts.append(
        "\n(Truncated schedule. Full data is stored locally.)"
    )
    return "\n".join(text_parts)