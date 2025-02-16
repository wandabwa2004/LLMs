# maintenance_pipeline.py
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus

class MaintenanceOptimizer:
    def __init__(self, alpha=0.0, include_manpower_constraint=False, manpower_limit=None):
        self.alpha = alpha
        self.include_manpower_constraint = include_manpower_constraint
        self.manpower_limit = manpower_limit

    def optimize_schedule(self, risk_df: pd.DataFrame, budget: float, cost_col: str = "cost", fail_prob_col: str = "failure_probability", risk_impact_col: str = "risk_impact", labor_col: str = "labor_hours"):
        if cost_col not in risk_df.columns or fail_prob_col not in risk_df.columns:
            raise ValueError(f"Missing required columns: {cost_col}, {fail_prob_col}.")

        if risk_impact_col not in risk_df.columns:
            risk_df[risk_impact_col] = 1.0

        df = risk_df.copy()

        problem = LpProblem("MaintenanceOptimization", LpMinimize)

        df["var"] = [LpVariable(f"x_{eid}", cat=LpBinary) for eid in df["equipment_id"]]

        objective_terms = []
        for i, row in df.iterrows():
            x_var = row["var"]
            p_fail = row[fail_prob_col]
            imp = row[risk_impact_col]

            risk_if_not = p_fail * imp
            risk_if_maint = self.alpha * p_fail * imp

            expression = risk_if_not * (1 - x_var) + risk_if_maint * (x_var)
            objective_terms.append(expression)

        problem += lpSum(objective_terms), "TotalRisk"

        cost_expr = lpSum(row[cost_col] * row["var"] for _, row in df.iterrows())
        problem += cost_expr <= budget, "BudgetConstraint"

        if self.include_manpower_constraint and labor_col in df.columns and self.manpower_limit is not None:
            labor_expr = lpSum(row[labor_col] * row["var"] for _, row in df.iterrows())
            problem += labor_expr <= self.manpower_limit, "ManpowerConstraint"

        problem.solve()

        df["maintain"] = [int(var.varValue) for var in df["var"]]
        sol_status = LpStatus[problem.status]

        total_risk_value = 0.0
        for _, row in df.iterrows():
            if row["maintain"] == 1:
                total_risk_value += self.alpha * row[fail_prob_col] * row[risk_impact_col]
            else:
                total_risk_value += row[fail_prob_col] * row[risk_impact_col]

        df.drop(columns=["var"], inplace=True)
        df["solution_status"] = sol_status
        df["optimized_risk"] = total_risk_value

        return df

def ingest_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def local_analysis(df: pd.DataFrame) -> dict:
    total_equipment = len(df)
    avg_failure = df["failure_probability"].mean() if "failure_probability" in df.columns else 0
    high_risk_count = (df["failure_probability"] > 0.7).sum() if "failure_probability" in df.columns else 0

    summary = {
        "total_equipment": total_equipment,
        "avg_failure_probability": float(avg_failure),
        "high_risk_count": int(high_risk_count),
    }
    print(summary)
    return summary

def local_optimization(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    optimizer = MaintenanceOptimizer(alpha=0.0, include_manpower_constraint=True, manpower_limit=5000)
    schedule_df = optimizer.optimize_schedule(
        risk_df=df,
        budget=budget,
        cost_col="cost",
        fail_prob_col="failure_probability",
        risk_impact_col="risk_impact",
        labor_col="labor_hours"
    )
    return schedule_df

def run_pipeline(file_path: str, budget: float) -> dict:
    df = ingest_data(file_path)
    analysis = local_analysis(df)
    schedule_df = local_optimization(df, budget)

    plan_summary = {
        "maintenance_schedule": schedule_df.to_dict(orient="records"),
        "optimized_risk": float(schedule_df["optimized_risk"].iloc[0]) if len(schedule_df) else 0.0,
        "solution_status": schedule_df["solution_status"].iloc[0] if len(schedule_df) else "Unknown",
    }

    return {
        "analysis_summary": analysis,
        "plan_summary": plan_summary
    }