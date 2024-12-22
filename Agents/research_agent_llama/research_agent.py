import time
from typing import Dict, List
import torch  # For device selection

# Hugging Face and LangChain imports
from langchain.llms import HuggingFacePipeline
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class ResearchAgent:
    """
    A research agent that uses DuckDuckGo for web searches and a Falcon-based
    language model for summarization and final answers. Uses a REACT-based
    agent with conversation buffering.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", hf_token: str = None):
        # Determine the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

        # Load model with appropriate device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            device_map="auto" if torch.cuda.is_available() else None,  # Automatically maps layers if GPU is available
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 on GPU, FP32 on CPU
        )

        # Initialize Hugging Face pipeline
        hf_pipeline = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Memory to store the conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create tools
        self.search_tool = self._create_rate_limited_search()
        self.summarize_tool = self._create_summarize_tool()

        # Initialize the ReAct agent
        self.agent = self._initialize_agent()

    def _create_rate_limited_search(self) -> Tool:
        """
        Create a DuckDuckGo web search tool with a 1-second sleep to avoid
        hitting rate limits.
        """
        search = DuckDuckGoSearchRun()

        def rate_limited_search(query: str) -> str:
            time.sleep(1)
            try:
                return search.run(query)
            except Exception as e:
                return f"Error during search: {str(e)}"

        return Tool(
            name="Web Search",
            func=rate_limited_search,
            description="Searches the internet for current information."
        )

    def _create_summarize_tool(self) -> Tool:
        """
        Create a summarization tool using a simple LLMChain with a prompt
        that instructs the LLM to summarize text.
        """
        summarize_prompt = ChatPromptTemplate.from_template(
            "Please summarize the following text:\n\n{text}"
        )
        summarize_chain = LLMChain(llm=self.llm, prompt=summarize_prompt)

        return Tool(
            name="Summarize",
            func=lambda text: summarize_chain.run(text),
            description="Useful for summarizing long texts."
        )

    def _initialize_agent(self) -> AgentExecutor:
        """
        Create the ReAct agent that knows how to:
        - Use the Web Search tool for up-to-date info
        - Use the Summarize tool for text condensation
        - Maintain conversation history in memory
        """
        # Define a ChatPromptTemplate for the ReAct agent
        react_prompt = ChatPromptTemplate.from_template(
            """Assistant, you are a helpful AI research agent. You have access to the following tools:

{tools}

Tool Names: {tool_names}

To use a tool, please use the following format:
Action: [the action to take, should be one of [{tool_names}]]
Action Input: [the input to the action]
Observation: [the result of the action]

When you have a final answer, respond with:
Final Answer: [your final answer here]

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought: Let me approach this step by step
{agent_scratchpad}"""
        )

        # Create the ReAct agent
        react_agent = create_react_agent(
            llm=self.llm,
            tools=[self.search_tool, self.summarize_tool],
            prompt=react_prompt
        )

        # Wrap the agent with memory & iteration control
        return AgentExecutor(
            agent=react_agent,
            tools=[self.search_tool, self.summarize_tool],
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )

    def research(self, query: str) -> Dict:
        """
        Execute a 'Research' query using the agent. The agent may search
        and summarize as needed, then respond with a final result.
        Returns a dictionary containing success status, result text, and errors.
        """
        try:
            result = self.agent.invoke({
                "input": f"Research this topic and provide a summary: {query}"
            })
            return {
                "success": True,
                "result": result["output"],
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }

    def get_chat_history(self) -> List[str]:
        """
        Retrieve the list of conversation messages from memory,
        including both user and agent messages.
        """
        return self.memory.chat_memory.messages