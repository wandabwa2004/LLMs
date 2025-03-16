from crewai import Agent, Task, LLM
from yaml import safe_load

def load_prompt(file_path: str, prompt_key: str) -> dict:
    with open(file_path) as f:
        prompts = safe_load(f)
    return prompts[prompt_key]

# Create a single LLM instance for all agents using the Deepseek model
deepseek_llm = LLM(
    model="ollama/deepseek-r1:7b",
    base_url="http://localhost:11434",
    api_key=""  # Empty because we're using a local model
)

def create_financial_agent():
    financial_prompt = load_prompt('prompts/financial_agent.yaml', 'financial_analysis')
    financial_agent = Agent(
        role='Financial Advisor',
        goal='Maximize ROI within the allocated budget',
        backstory='Expert in utility sector economics with CPA certification',
        llm=deepseek_llm,  # Use the unified Deepseek LLM instance
        verbose=True
    )
    return financial_agent

def create_risk_agent():
    risk_prompt = load_prompt('prompts/risk_agent.yaml', 'risk_assessment')
    risk_agent = Agent(
        role='Risk and Compliance Specialist',
        goal='Focus on safety, regulatory requirements, and potential legal issues',
        backstory='Former utility regulator with safety compliance expertise',
        llm=deepseek_llm,  # Use the unified Deepseek LLM instance
        verbose=True
    )
    return risk_agent

def create_communicator_agent():
    comm_prompt = load_prompt('prompts/communicator_agent.yaml', 'executive_summary')
    communicator_agent = Agent(
        role='Strategic Communicator',
        goal='Merge various advisors\' insights into a coherent, executive-level summary',
        backstory='Former management consultant with technical translation expertise',
        llm=deepseek_llm,  # Use the unified Deepseek LLM instance
        verbose=True
    )
    return communicator_agent

def get_financial_advice(agent, plan_summary: dict) -> str:
    prompt_config = load_prompt('prompts/financial_agent.yaml', 'financial_analysis')
    prompt = prompt_config['prompt'].format(content=plan_summary['summary'])
    task = Task(
        description=prompt_config['description'],
        agent=agent,
        expected_output=prompt_config['expected_output'],
        context=[{
            "description": "Maintenance Plan Summary",
            "content": plan_summary['summary'],
            "expected_output": "Markdown report with cost breakdowns and 3-year ROI projections"
        }],
        config={'llm': 'ollama_config.yaml'}
    )
    output = task.execute_sync(context=prompt)
    return output.raw if output and output.raw is not None else ""

def get_risk_advice(agent, analysis_summary: dict, plan_summary: dict) -> str:
    prompt_config = load_prompt('prompts/risk_agent.yaml', 'risk_assessment')
    prompt = prompt_config['prompt'].format(
        analysis_content=analysis_summary,
        plan_content=plan_summary['summary']
    )
    task = Task(
        description=prompt_config['description'],
        agent=agent,
        expected_output=prompt_config['expected_output'],
        context=[
            {
                "description": "Analysis Summary",
                "content": analysis_summary,
                "expected_output": "Risk matrix with probability/impact scores and mitigation strategies"
            },
            {
                "description": "Maintenance Plan Summary",
                "content": plan_summary['summary'],
                "expected_output": "Risk matrix with probability/impact scores and mitigation strategies"
            }
        ],
        config={'llm': 'ollama_config.yaml'}
    )
    output = task.execute_sync(context=prompt)
    return output.raw if output and output.raw is not None else ""

def get_communicator_report(agent, financial_advice: str, risk_advice: str) -> str:
    prompt_config = load_prompt('prompts/communicator_agent.yaml', 'executive_summary')
    prompt = prompt_config['prompt'].format(
        financial_content=financial_advice,
        risk_content=risk_advice
    )
    task = Task(
        description=prompt_config['description'],
        agent=agent,
        expected_output=prompt_config['expected_output'],
        context=[
            {
                "description": "Financial Advice",
                "content": financial_advice,
                "expected_output": "Executive memo with prioritized action items and risk tradeoffs"
            },
            {
                "description": "Risk Advice",
                "content": risk_advice,
                "expected_output": "Executive memo with prioritized action items and risk tradeoffs"
            }
        ],
        config={'llm': 'ollama_config.yaml'}
    )
    output = task.execute_sync(context=prompt)
    return output.raw if output and output.raw is not None else ""
