# strategy_agents.py

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

def create_financial_agent(openai_api_key: str):
    """
    Agent specialized in cost & budget optimization advice.
    """
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
        openai_api_key=openai_api_key
    )
    tools = []  # no tools needed
    system_msg = (
        "You are a Financial Advisor for a utility company. "
        "Provide cost-based recommendations to maximize ROI within the allocated budget."
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(),
        verbose=True,
        handle_parsing_errors=True,
        system_message=system_msg
    )
    return agent

def create_risk_agent(openai_api_key: str):
    """
    Agent specialized in risk & compliance aspects.
    """
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
        openai_api_key=openai_api_key
    )
    tools = []
    system_msg = (
        "You are a Risk and Compliance Specialist. Focus on safety, regulatory requirements, "
        "and potential legal issues in the proposed plan."
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(),
        verbose=True,
        handle_parsing_errors=True,
        system_message=system_msg
    )
    return agent

def create_communicator_agent(openai_api_key: str):
    """
    Agent that consolidates final recommendations into an executive summary.
    This agent sees the outputs from the financial and risk agents
    and produces a polished, high-level strategy statement.
    """
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
        openai_api_key=openai_api_key
    )
    tools = []
    system_msg = (
        "You are a Strategic Communicator for a utility maintenance department. "
        "Your role is to merge various advisors' insights into a coherent, "
        "executive-level summary that is clear and actionable."
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(),
        verbose=True,
        handle_parsing_errors=True,
        system_message=system_msg
    )
    return agent

def get_financial_advice(agent, plan_summary: dict) -> str:
    """
    Sends cost/budget info to the Financial Agent.
    """
    # prompt = (
    #     f"Maiintenance  Plan Summary: {plan_summary}\n\n"
    #     "Please provide cost-oriented recommendations to stay on or under budget "
    #     "and maximize return on investment."
    # )

    prompt = (
        f"Review the following maintenance plan details: {plan_summary}\n\n"
        """Please provide a detailed financial analysis addressing:
            1. Cost breakdown and budget allocation recommendations
            2. Potential cost savings and efficiency improvements
            3. ROI calculations for major expenditures
            4. Resource optimization strategies
            5. Long-term financial implications
            Focus on quantifiable metrics and specific cost-saving opportunities while maintaining service quality."""
              )

    response = agent.run(prompt)
    return response


def get_risk_advice(agent, analysis_summary: dict, plan_summary: dict) -> str:
    """
    Sends analysis and plan info to the Risk Agent for compliance/safety insights.
    """
    # prompt = (
    #     f"Analysis summary: {analysis_summary}\n\n"
    #     f"Plan summary: {plan_summary}\n\n"
    #     "Identify major risk/compliance concerns and suggest improvements."
    # )

    prompt = f"""Review the following analysis and plan details:

        ANALYSIS SUMMARY:
        {analysis_summary}

        PLAN SUMMARY:
        {plan_summary}

        Please provide a comprehensive risk assessment addressing:
        1. Safety compliance concerns and mitigation strategies
        2. Regulatory requirements and compliance gaps
        3. Environmental impact considerations
        4. Emergency response preparedness
        5. Workforce safety requirements

        Prioritize risks by severity and likelihood, and provide specific mitigation recommendations."""

  
    response = agent.run(prompt)
    return response

def get_communicator_report(agent, financial_advice: str, risk_advice: str) -> str:
    """
    Combines the final advices into an executive-level summary.
    """
    # prompt = (
    #     "FINANCIAL ADVISOR SAYS:\n"
    #     f"{financial_advice}\n\n"
    #     "RISK ADVISOR SAYS:\n"
    #     f"{risk_advice}\n\n"
    #     "Please merge these points into a cohesive, concise executive summary."
    # )

    prompt = f"""Synthesize the following expert recommendations into an executive summary:

            "FINANCIAL ANALYSIS:\n"
            f"{financial_advice}\n\n"

            "RISK ASSESSMENT:\n"
            f"{risk_advice}\n\n"

            Create a strategic summary that:
            1. Highlights critical findings and recommendations
            2. Prioritizes actions based on urgency and impact
            3. Provides clear implementation steps
            4. Identifies key decision points and timelines
            5. Balances financial and risk considerations

            Focus on actionable insights and measurable outcomes. Structure the response with clear sections for immediate actions, short-term priorities, and long-term strategies."""
    response = agent.run(prompt)
    return response