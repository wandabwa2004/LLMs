import os  # For interacting with the operating system, specifically for environment variables
import streamlit as st  # For creating the web application interface
from dotenv import load_dotenv  # For loading environment variables from a .env file
from crewai import Agent, Task, Crew, Process  # Core components from the crewAI framework for creating agentic workflows
from langchain_openai import ChatOpenAI  # Specific class for interacting with OpenAI's chat models via LangChain

# Load environment variables from a .env file in the same directory
# This is useful for keeping sensitive information like API keys out of the code
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key was successfully loaded; raise an error if not
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Set up the Large Language Model (LLM) to be used by the agents
# Using OpenAI's gpt-4o-mini model, known for its balance of capability and cost
# Temperature controls randomness (0.4 is relatively focused)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=openai_api_key)

# Define a simple function to assess the risk level in the user's message
def is_high_risk(text):
    """Assess if user text indicates high risk based on keywords."""
    # List of keywords associated with potential crisis situations
    # Added "hopeless" to broaden the scope slightly
    risk_keywords = [
        "suicide", "kill myself", "end my life", "harm myself",
        "hurt myself", "die", "death", "no point", "give up", "hopeless", # Added 'hopeless'
        "abuse", "hitting me", "beating me", "harming me",
        "scared", "terrified", "trapped", "emergency"
    ]

    # Convert the input text to lowercase for case-insensitive matching
    text_lower = text.lower()
    # Calculate a simple risk score: count occurrences of keywords and normalize by the total number of keywords
    # This is a basic heuristic and might need refinement for accuracy
    # A hit on any keyword will now result in a score > 0
    hits = sum([1 for word in risk_keywords if word in text_lower])
    if hits == 0:
        return 0.0
    # Simple scoring: base score + increment per hit, capped at 1.0
    # Ensures any hit gives a noticeable score.
    risk_score = min(0.1 + 0.1 * hits, 1.0)

    return risk_score

# Define a function to create the different specialized AI agents using crewAI
def create_agents():
    """Create the specialized therapy agents with enhanced roles"""

    # Agent focused on understanding and responding appropriately within the Kenyan cultural context
    cultural_agent = Agent(
        role="Cultural Context Expert",
        goal="Ensure responses are culturally sensitive, acknowledging Kenyan teen realities like stigma, family/school pressures, and economic context, while respecting diverse backgrounds (including spirituality if mentioned by user).",
        backstory="""You are an expert in Kenyan youth culture, language (Sheng, Kiswahili, English), and social dynamics.
        You understand the pressures related to education (KCPE/KCSE), family expectations, potential economic hardship, and the stigma surrounding mental health.
        You can translate when needed and ensure communication is respectful and relevant. You acknowledge spiritual beliefs respectfully *if* the user brings them up, without imposing any views.""",
        verbose=True,  # Enables detailed logging of the agent's thought process
        llm=llm  # Assign the pre-configured LLM to this agent
    )

    # Agent focused on validating feelings and showing understanding, including relational aspects
    empathy_agent = Agent(
        role="Empathetic Listener",
        goal="Show deep understanding and empathy towards teen struggles, validating feelings related to personal issues, family, peers, and school. Create a safe, non-judgmental space.",
        backstory="""You are a deeply empathetic listener who understands the complex emotional world of teenagers, including relationship dynamics (family, friends, romance).
        You validate feelings without judgment, fostering a safe space for expression. You respond with warmth and sensitivity, recognizing potential underlying distress or trauma indicators without diagnosing.""",
        verbose=True,
        llm=llm
    )

    # Agent focused on Cognitive Behavioral Therapy techniques, including identity and psychoeducation
    cbt_agent = Agent(
        role="CBT Therapist Helper",
        goal="Help teens gently identify links between thoughts, feelings, and behaviors, challenge unhelpful patterns, and explore how thoughts impact self-perception and identity. Provide simple psychoeducation.",
        backstory="""You are a helper grounded in cognitive behavioral principles. You guide teens to notice negative or unhelpful thinking patterns and gently explore alternative perspectives.
        You can offer simple explanations (psychoeducation) about how thoughts influence feelings and actions. You support self-discovery and building a healthier self-concept.""",
        verbose=True,
        llm=llm
    )

    # Agent focused on providing practical coping mechanisms, including relational and psychoeducation
    coping_agent = Agent(
        role="Coping Skills Coach",
        goal="Teach practical, accessible coping strategies for stress, anxiety, and difficult emotions, including those arising from relational issues. Briefly explain *why* a technique might help.",
        backstory="""You specialize in teaching practical, easy-to-implement coping mechanisms suitable for teens in a Kenyan context.
        You offer strategies for managing anxiety, stress, anger, and sadness, including skills for navigating difficult interactions. You provide brief, simple explanations (psychoeducation) for how these techniques work (e.g., calming the nervous system).""",
        verbose=True,
        llm=llm
    )

    # Agent focused on helping teens set and plan goals linked to values and identity
    goal_agent = Agent(
        role="Goal-Setting Guide",
        goal="Help teens clarify meaningful, achievable goals (SMART), break them down into steps, and connect them to their values and future aspirations.",
        backstory="""You help teens articulate their aspirations and translate them into concrete, manageable goals.
        You encourage exploration of the 'why' behind goals, linking them to personal values and identity. You motivate and support planning, fostering a sense of agency and hope.""",
        verbose=True,
        llm=llm
    )

    # Agent focused on suggesting relevant local resources with nuance
    resource_agent = Agent(
        role="Resource Connector",
        goal="Connect teens with appropriate, relevant, and potentially accessible Kenyan resources (helplines, NGOs, online info, community support types), acknowledging potential access barriers.",
        backstory="""You are knowledgeable about Kenyan youth support resources, including helplines (like 116, Befrienders Kenya), community centers, reputable NGOs, and online information.
        You aim to suggest specific resources relevant to the teen's expressed need, considering potential accessibility issues (cost, location, internet) where possible, while always respecting privacy.""",
        verbose=True,
        llm=llm
    )

    # Agent specialized in handling immediate crisis situations with a trauma-informed sensitivity
    crisis_agent = Agent(
        role="Crisis Support Specialist",
        goal="Provide immediate, calm, empathetic support in crisis situations, prioritize safety, and clearly provide emergency resources (Kenyan hotlines).",
        backstory="""You are trained to respond calmly and supportively to teens in acute distress or crisis (e.g., expressing suicidal thoughts, reporting immediate danger).
        Your priority is safety and connection to immediate help. You provide clear, actionable information about emergency resources (like Childline 116, Befrienders Kenya) while showing care and maintaining a non-judgmental stance.""",
        verbose=True,
        llm=llm
    )

    # Return a dictionary mapping agent names to their corresponding Agent objects
    return {
        "cultural": cultural_agent,
        "empathy": empathy_agent,
        "cbt": cbt_agent,
        "coping": coping_agent,
        "goal": goal_agent,
        "resource": resource_agent,
        "crisis": crisis_agent
    }

# Define a function to create tasks for the agents based on the user's message and conversation history
def create_tasks(agents, user_msg, history):
    """Create specialized tasks for each agent based on user message and enhanced roles"""

    # Format the conversation history into a single string for context
    history_text = "\n".join([f"{role}: {msg}" for role, msg in history])

    # Task for the cultural agent: assess language, ensure cultural appropriateness considering context
    cultural_task = Task(
        description=f"""
        1. Check user message language (English/Kiswahili/Sheng). If not English, note key terms/phrases for context but proceed assuming core message is understandable.
        2. Analyze the message for cultural nuances relevant to Kenyan teens (e.g., mentions of school pressure, family roles, stigma, economic stress).
        3. Briefly note any relevant cultural context or language use that subsequent agents should be aware of. Ensure the overall approach remains sensitive.
        4. If the user mentions spiritual beliefs, note this for respectful acknowledgment later, but do not elaborate unless directly asked.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["cultural"],  # Assign this task to the cultural agent
        expected_output="Brief analysis of language and cultural context relevant to the user's message, highlighting points for sensitivity (e.g., 'User mentions KCSE stress', 'User used Sheng term X meaning Y', 'User mentions prayer as coping')."
    )

    # Task for the empathy agent: validate feelings, create safety, acknowledge relational context
    empathy_task = Task(
        description=f"""
        1. Deeply listen to the user's message and the preceding context.
        2. Identify the core emotions being expressed (e.g., sadness, anxiety, frustration, confusion).
        3. Validate these feelings in 1-2 warm, genuine sentences. Acknowledge any mentioned relational context (family, friends) briefly.
        4. Reinforce that this is a safe space to talk. Avoid giving advice.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["empathy"],
        expected_output="A short (1-3 sentences), warm, empathetic response validating the user's expressed feelings and reinforcing safety. Example: 'It sounds like you're feeling really [emotion] about [situation], especially with [relational context if mentioned]. It's okay to feel that way, and I'm here to listen.'"
    )

    # Task for the CBT agent: identify thoughts, suggest reframing, offer simple psychoeducation
    cbt_task = Task(
        description=f"""
        1. Identify a potential unhelpful thought pattern mentioned or implied in the user's message (e.g., negative self-talk, catastrophizing).
        2. Gently guide the user to notice this thought.
        3. Briefly explain the link between thoughts and feelings (simple psychoeducation).
        4. Suggest exploring an alternative, more balanced perspective or ask a gentle question to prompt reflection. Keep it simple and supportive.
        5. Synthesize key points from previous steps (empathy) into a coherent response.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["cbt"],
        expected_output="A supportive response that includes: validation (building on empathy), identification of a thought pattern, a simple explanation of thought-feeling link, and a gentle prompt for reflection/reframing. Example: 'I hear how tough that is. Sometimes our thoughts can make feelings stronger, like thinking [negative thought]. What might be another way to look at this situation?'"
    )

    # Task for the coping agent: recommend relevant coping strategies with brief explanation
    coping_task = Task(
        description=f"""
        1. Based on the user's expressed feelings (e.g., stress, anxiety, sadness) and situation, suggest ONE specific, practical coping technique suitable for a teen (e.g., 4-7-8 breathing, grounding, simple mindfulness, positive self-talk prompt, brief physical activity).
        2. Briefly explain *why* this technique might help in 1 sentence (e.g., 'helps calm your body', 'brings focus to the present').
        3. Ensure the suggestion is accessible and easy to try.
        4. Synthesize key points from previous steps (empathy) into a coherent response.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["coping"],
        expected_output="A supportive response that includes: validation (building on empathy), ONE practical coping skill suggestion, and a brief (1 sentence) explanation of its benefit. Example: 'It makes sense you're feeling overwhelmed. Sometimes taking a moment to focus on your breath can help calm your body. You could try the 4-7-8 breathing technique: breathe in for 4, hold for 7, out for 8.'"
    )

    # Task for the goal agent: help clarify goals linked to values, break down steps
    goal_task = Task(
        description=f"""
        1. Acknowledge the user's interest in goals/planning.
        2. Help clarify the goal or ask a question to make it more specific and achievable (SMART).
        3. Gently prompt reflection on *why* this goal is important to them (link to values/identity).
        4. Suggest breaking it down into 1-2 small, initial steps.
        5. Maintain an encouraging tone.
        6. Synthesize key points from previous steps (empathy) into a coherent response.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["goal"],
        expected_output="An encouraging response that includes: validation (building on empathy), help clarifying the goal, a prompt about its importance, and suggestion for 1-2 initial steps. Example: 'It's great you're thinking about [goal area]. What's one small thing you could do this week towards that? Thinking about why it matters to you can also be motivating.'"
    )

    # Task for the resource agent: suggest relevant, varied Kenyan resources with nuance
    resource_task = Task(
        description=f"""
        1. Based on the user's issue, identify 1-2 relevant Kenyan support resources.
        2. Aim for variety if possible (e.g., a hotline AND an online resource, or a general youth support org).
        3. Include key contact info (phone number like 116 or Befrienders, website if applicable).
        4. Briefly state what the resource offers (e.g., 'free confidential listening', 'info on mental health').
        5. Acknowledge that accessing resources can sometimes be tricky, but help is available.
        6. Synthesize key points from previous steps (empathy) into a coherent response.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["resource"],
        expected_output="A supportive response that includes: validation (building on empathy), 1-2 specific Kenyan resources with contact info/website and purpose, and a brief note acknowledging potential access challenges. Example: 'It takes courage to reach out. For confidential support, you could contact Childline Kenya by calling 116 (it's free), or check out [relevant website] for information on [topic].'"
    )

    # Task for the crisis agent: assess crisis level, provide immediate support/resources calmly
    crisis_task = Task(
        description=f"""
        1. Acknowledge the user's distress with calm empathy.
        2. Prioritize safety: Gently express concern and the importance of immediate help.
        3. Clearly provide Kenyan emergency contact information: Childline (116) and Befrienders Kenya (+254 722 178 177). Mention they offer immediate, confidential support.
        4. Avoid judgment and complex questions. Keep the message focused, supportive, and directive towards seeking help.
        5. Reinforce that they are not alone and help is available *now*.

        User's message: {user_msg}
        Conversation history: {history_text}
        """,
        agent=agents["crisis"],
        expected_output="A calm, direct, and supportive crisis response that includes: empathy, clear statement of concern for safety, Kenyan emergency hotline numbers (116, Befrienders Kenya) with encouragement to call immediately, and reassurance. Example: 'I hear how much pain you're in right now, and I'm concerned about you. It's really important to talk to someone who can help immediately. Please reach out to Childline Kenya by calling 116 or Befrienders Kenya at +254 722 178 177. They are there to support you right now. You don't have to carry this alone.'"
    )

    # Return a dictionary mapping task names to their corresponding Task objects
    return {
        "cultural": cultural_task,
        "empathy": empathy_task,
        "cbt": cbt_task,
        "coping": coping_task,
        "goal": goal_task,
        "resource": resource_task,
        "crisis": crisis_task
    }

# Function to synthesize the outputs from different agents into a single, coherent response
# Refined to handle potentially richer outputs and prioritize crisis response clearly.
def combine_agent_responses(crew_result, risk_score):
    """
    Combine multiple agent responses into a coherent chat message.
    Prioritizes crisis response if risk is high.
    Otherwise, attempts to weave together empathy, advice/resources, and a supportive closing.
    """
    responses = []
    final_response = "I'm here to listen. How can I help you today?" # Default fallback

    # --- 1. Extract Raw Outputs ---
    if isinstance(crew_result, str):
        # Handle case where crew returns a simple string (e.g., error or simple output)
         return crew_result
    elif hasattr(crew_result, 'tasks_output') and crew_result.tasks_output:
        # Extract raw text from each task output object
        for task_output in crew_result.tasks_output:
            if hasattr(task_output, 'raw') and task_output.raw:
                responses.append(task_output.raw.strip()) # Store stripped responses
            # Fallback if 'raw' isn't present but output is stringifiable
            elif task_output:
                 responses.append(str(task_output).strip())
    elif hasattr(crew_result, 'raw_output') and crew_result.raw_output:
         # Handle cases where the result might be directly in raw_output
         return crew_result.raw_output
    elif responses:
         # If we got responses some other way
         pass
    else:
        # If no usable output found, return the raw result stringified
        print(f"Warning: Could not extract structured responses from crew_result: {crew_result}")
        return str(crew_result) # Return raw string representation as fallback

    # --- 2. Handle High-Risk (Crisis) Scenario ---
    # Use the risk score threshold defined for routing (0.15)
    if risk_score >= 0.15:
        print("Combine Responses: High risk detected, prioritizing crisis output.")
        # In the crisis flow (Cultural -> Empathy -> Crisis -> Coping), the Crisis response is likely the 3rd output.
        # The Coping response (4th) might offer grounding, but the Crisis message is paramount.
        crisis_response = "It sounds like you're going through a very difficult time. Please reach out for immediate support by calling Childline Kenya at 116 or Befrienders Kenya at +254 722 178 177. They are available to help you right now." # Safer default
        if len(responses) >= 3:
            # Assume the third response is from the Crisis agent
            # Check if it contains hotline numbers; if so, use it directly.
            if "116" in responses[2] or "Befrienders" in responses[2].lower():
                 crisis_response = responses[2]
            # If not, maybe the 4th (coping) response has them? Less likely but check.
            elif len(responses) >= 4 and ("116" in responses[3] or "Befrienders" in responses[3].lower()):
                 crisis_response = responses[3]
            # If neither contains hotlines, stick to the default crisis message.

        # Ensure essential hotline info is present
        if "116" not in crisis_response:
            crisis_response += "\nEmergency Help: Call 116 (Childline)"
        if "Befrienders" not in crisis_response:
             crisis_response += "\nOr call +254 722 178 177 (Befrienders Kenya)"

        return crisis_response.strip()

    # --- 3. Handle Standard (Non-Crisis) Scenario ---
    print(f"Combine Responses: Standard flow detected. Responses received: {len(responses)}")
    if not responses:
         return final_response # Return default if no responses somehow

    # In standard flow (Cultural -> Empathy -> [CBT/Coping/Goal/Resource]), expect at least 3 responses.
    # The first is cultural context (often internal note), second is empathy, third is the specialized response.

    # Start with Empathy (usually the second response)
    empathy_part = ""
    if len(responses) >= 2:
        # Take the empathy response, usually short and validating.
        empathy_part = responses[1]
        # Basic check if it looks like validation
        if not ("understand" in empathy_part.lower() or "hear you" in empathy_part.lower() or "sounds like" in empathy_part.lower() or "okay to feel" in empathy_part.lower()):
             empathy_part = f"I hear you. {empathy_part}" # Add a generic validation if needed

    # Get the main content (from CBT, Coping, Goal, or Resource agent - usually the last response)
    main_content_part = ""
    if len(responses) >= 3:
        main_content_part = responses[-1] # Assume the last agent provides the core advice/resource/prompt
        # Remove potential redundancy if it repeats the empathy part
        if empathy_part and main_content_part.startswith(empathy_part.split('.')[0]): # Check if starts similarly
             pass # Keep it as is, agent might have synthesized well
        elif empathy_part:
             main_content_part = "\n\n" + main_content_part # Add spacing if distinct

    # Construct the final response
    if empathy_part and main_content_part:
        final_response = empathy_part + main_content_part
    elif main_content_part: # Only got main content
        final_response = main_content_part
    elif empathy_part: # Only got empathy
        final_response = empathy_part + "\n\nWhat else is on your mind?" # Add a prompt
    elif responses: # Fallback to last response if logic failed
         final_response = responses[-1]

    # Add a general supportive closing, avoiding redundancy if already present
    closing_statement = "\n\nRemember, taking care of yourself is important, and you don't have to figure everything out alone. I'm here to support you."
    if not ("remember" in final_response.lower() or "alone" in final_response.lower() or "support you" in final_response.lower()):
         final_response += closing_statement

    return final_response.strip()


# Main function to process the user's message using the crewAI setup
def process_message(user_msg, history):
    """Process user message using enhanced crewAI agents and tasks, and return the response and risk score."""

    # 1. Create all defined agents with enhanced roles
    agents = create_agents()

    # 2. Create all defined tasks with enhanced descriptions
    tasks = create_tasks(agents, user_msg, history)

    # 3. Assess the risk level of the user's message using the updated function
    risk_score = is_high_risk(user_msg)
    print(f"Calculated Risk Score: {risk_score}") # Log risk score

    # 4. Determine the workflow (which tasks to run) based on risk and message content
    selected_tasks = []

    # Always include cultural context analysis first
    selected_tasks.append(tasks["cultural"])

    # Always include empathy second
    selected_tasks.append(tasks["empathy"])

    # --- Routing Logic ---
    # If risk score is above the threshold, activate the crisis workflow
    # Using 0.15 threshold for crisis intervention
    if risk_score >= 0.15:
        print(f"High risk detected (score: {risk_score}). Activating crisis workflow.")
        selected_tasks.append(tasks["crisis"]) # Add crisis assessment task
        selected_tasks.append(tasks["coping"])  # Add coping skills task even in crisis for immediate grounding/support

        # Define the crew specifically for crisis situations
        # Ensure agents match the selected tasks
        crew = Crew(
            agents=[agents["cultural"], agents["empathy"], agents["crisis"], agents["coping"]],
            tasks=selected_tasks, # Tasks selected for crisis
            verbose=True, # Enable detailed logging
            process=Process.sequential  # Ensure tasks run in the defined order
        )
        # Execute the crew's tasks
        result = crew.kickoff()
        print(f"Crisis Crew Result: {result}") # Log the raw result
        # Combine the results using the specialized function and return
        return combine_agent_responses(result, risk_score), risk_score

    # If not a crisis, determine the primary focus based on keywords in the user message
    else:
        print(f"Low risk detected (score: {risk_score}). Activating standard workflow.")
        text = user_msg.lower() # Use lowercase for keyword matching

        # Determine the primary path - run ONE specialized agent after empathy
        # More sophisticated routing could run multiple, but sequential process makes this tricky.
        # Prioritize based on common teen issues.
        if any(k in text for k in ["goal", "target", "plan", "future", "want to", "achieve", "improve"]):
            print("Routing to: Goal Setting")
            selected_tasks.append(tasks["goal"])
        elif any(k in text for k in ["stress", "anxious", "worry", "overwhelmed", "coping", "relax", "breathe", "calm down", "panic"]):
            print("Routing to: Coping Skills")
            selected_tasks.append(tasks["coping"])
        elif any(k in text for k in ["think", "thought", "negative", "bad", "sad", "depressed", "unhappy", "feel down", "self-esteem", "confidence"]):
            print("Routing to: CBT")
            selected_tasks.append(tasks["cbt"])
        elif any(k in text for k in ["friend", "family", "parents", "relationship", "dating", "argue", "fight", "lonely", "bullying"]):
             # If relational keywords detected, default to Coping skills which now includes relational aspects
             print("Routing to: Coping Skills (Relational Focus)")
             selected_tasks.append(tasks["coping"])
        # Default path: If no specific keywords match strongly, provide general support (coping + resources)
        else:
            print("Routing to: General Support (Coping + Resources)")
            # In sequential, we can only add one more task easily. Let's prioritize Coping as generally useful.
            # The resource agent's goal could be integrated into the Coping agent's task description for default cases,
            # or we accept that resources are only explicitly sought if 'resource' task is triggered.
            # Let's add Resource task here for broader default support.
            selected_tasks.append(tasks["coping"])
            selected_tasks.append(tasks["resource"]) # Add resource task in default path

    # Create and run the crew with the selected agents and tasks for non-crisis situations
    # Dynamically select only the agents whose tasks are included
    selected_agent_objects = [task.agent for task in selected_tasks if task.agent is not None]
    # Ensure unique agents if multiple tasks use the same one (though not the case here)
    unique_agents = list({agent.role: agent for agent in selected_agent_objects}.values())

    crew = Crew(
        agents=unique_agents,
        tasks=selected_tasks,
        verbose=True, # Enable detailed logging
        process=Process.sequential  # Run tasks sequentially
    )

    # Execute the crew's tasks
    result = crew.kickoff()
    print(f"Standard Crew Result: {result}") # Log the raw result
    # Combine the results using the specialized function and return
    return combine_agent_responses(result, risk_score), risk_score

# ==============================================================================
# Main function to run the Streamlit web application (EMOJI UPDATES HERE)
# ==============================================================================
def main():
    """Main Streamlit application function with updated emojis"""

    # Configure the Streamlit page settings
    # Using '🫂' (People Hugging) for a more supportive feel than '💬'
    st.set_page_config(page_title="Teen Therapy Chat", page_icon="🫂", layout="wide")

    # Display the main title and a subtitle
    # Using '🫂' in the title as well
    st.title("🫂 Teen Therapy Chat (Kenya)")
    st.markdown("A safe space to share your feelings and get support. (AI Assistant)") # Clarify it's AI

    # Initialize the conversation history in Streamlit's session state if it doesn't exist
    if "history" not in st.session_state:
        st.session_state.history = []  # Stores tuples of (role, message_text)

    # Create a container for the chat messages for better layout control
    chat_container = st.container()

    # Display the existing conversation history within the container
    with chat_container:
        for role, msg in st.session_state.history:
            # Use Streamlit's chat_message context manager
            if role == "user":
                # Using '👤' (Bust in Silhouette) for a neutral user representation
                st.chat_message("user", avatar="👤").write(msg)
            else:
                # Using '🦉' (Owl) for the assistant, implying wisdom/guidance
                st.chat_message("assistant", avatar="🦉").write(msg)

    # Create a form for user input
    with st.form("message_form", clear_on_submit=True):
        # Text area for the user to type their message
        user_input = st.text_area("Type your message here (English/Kiswahili/Sheng)...", height=100, key="user_input")
        # Submit button for the form
        submitted = st.form_submit_button("Send")

    # Process the input only if the form was submitted and the input is not empty/whitespace
    if submitted and user_input.strip():
        # 1. Add the user's message to the session state history
        st.session_state.history.append(("user", user_input))

        # 2. Display the user's message immediately in the chat container
        with chat_container:
            # Using '👤' for user message display
            st.chat_message("user", avatar="👤").write(user_input)

        # 3. Process the message and get the assistant's response
        try:
            # Show a "Thinking..." indicator while processing
            with chat_container:
                # Using '🦉' for assistant avatar during thinking
                with st.chat_message("assistant", avatar="🦉"):
                    thinking_placeholder = st.empty()
                    # Using '💭' (Thought Bubble) for the thinking indicator
                    thinking_placeholder.markdown("💭 _Thinking..._")

            # Call the main processing function with the user input and current history
            response, risk_score = process_message(user_input, st.session_state.history)

            # 4. Add the assistant's generated response to the history
            st.session_state.history.append(("assistant", response))

            # 5. Display the assistant's reply, replacing the "Thinking..." indicator
            with chat_container:
                 # Using '🦉' for the final assistant message display
                 with st.chat_message("assistant", avatar="🦉"):
                      thinking_placeholder.markdown(response) # Update the placeholder

            # 6. Display a crisis banner if the risk score is high
            if risk_score >= 0.15:
                # Keeping '⚠️' (Warning) for the crisis banner for clarity and impact
                st.error("⚠️ If you're feeling overwhelmed or unsafe, please reach out for immediate help. Consider calling **116** (Childline Kenya - free call) or **+254 722 178 177** (Befrienders Kenya). You are not alone!")

        # Handle potential errors during processing
        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            print(f"ERROR: {error_message}") # Log the full error server-side
            st.error("I'm sorry, I encountered a technical issue and couldn't process your message properly. Please try sending it again in a moment.") # Display user-friendly error
            # Add a generic error message to the chat history
            st.session_state.history.append(("assistant", "Sorry, I ran into a technical problem. Could you please try sending your message again?"))
            # Display the error message in the chat interface
            with chat_container:
                # Using '🦉' for error message display
                st.chat_message("assistant", avatar="🦉").write("Sorry, I ran into a technical problem. Could you please try sending your message again?")

# Standard Python entry point
if __name__ == "__main__":
    main()
