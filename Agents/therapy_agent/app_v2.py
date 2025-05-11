import os  # For interacting with the operating system, specifically for environment variables
import streamlit as st  # For creating the web application interface
from dotenv import load_dotenv  # For loading environment variables from a .env file
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Load environment variables from a .env file in the same directory
# This is useful for keeping sensitive information like API keys out of the code
load_dotenv()

# Get the OpenAI API key from environment variablesstrea
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

# Function to synthesize the outputs from different agents into a single, coherent response
# Refined to handle potentially richer outputs and prioritize crisis response clearly.
def combine_agent_responses(step_responses, risk_score):
    """
    Combine multiple agent responses into a coherent chat message.
    Prioritizes crisis response if risk is high.
    Otherwise, attempts to weave together empathy, advice/resources, and a supportive closing.
    Args:
        step_responses (list[str]): A list of strings, where each string is the output
                                    of a sequential LLM call simulating an agent's task.
        risk_score (float): The calculated risk score for the user's message.
    """
    responses = []
    final_response = "I'm here to listen. How can I help you today?" # Default fallback

    # --- 1. Extract Raw Outputs ---
    # Ensure step_responses is a list of strings
    if isinstance(step_responses, list) and all(isinstance(r, str) for r in step_responses):
        responses = [r.strip() for r in step_responses if r and r.strip()]
    else:
        print(f"Warning: Invalid step_responses format received: {step_responses}")
        # Attempt to use the input directly if it's a string, otherwise return default
        return str(step_responses) if isinstance(step_responses, str) else final_response

    # --- 2. Handle High-Risk (Crisis) Scenario ---
    # Use the risk score threshold defined for routing (0.15)
    if risk_score >= 0.15:
        print("Combine Responses: High risk detected, prioritizing crisis output.")
        # In the crisis flow (Cultural -> Empathy -> Crisis -> Coping), the Crisis response is likely the 3rd output.
        # The Coping response (4th) might offer grounding, but the Crisis message is paramount.
        crisis_response = "It sounds like you're going through a very difficult time. Please reach out for immediate support by calling Childline Kenya at 116 or Befrienders Kenya at +254 722 178 177. They are available to help you right now." # Safer default crisis message
        if len(responses) >= 3:
            # Assume the third response (index 2) is from the Crisis step
            # Check if it contains hotline numbers; if so, use it directly.
            if "116" in responses[2] or "Befrienders" in responses[2].lower():
                 crisis_response = responses[2]
            # If not, maybe the 4th (coping) response (index 3) has them? Less likely but check.
            elif len(responses) >= 4 and ("116" in responses[3] or "Befrienders" in responses[3].lower()):
                 crisis_response = responses[3]
            # If neither contains hotlines, stick to the default crisis message.

        # Ensure essential hotline info is present in the final crisis response
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
    # The first (index 0) is cultural context (often internal note), second (index 1) is empathy, third (index 2) is the specialized response.

    # Start with Empathy (usually the second response)
    empathy_part = ""
    if len(responses) >= 2:
        # Take the empathy response (index 1), usually short and validating.
        empathy_part = responses[1]
        # Basic check if it looks like validation
        if not ("understand" in empathy_part.lower() or "hear you" in empathy_part.lower() or "sounds like" in empathy_part.lower() or "okay to feel" in empathy_part.lower()):
             empathy_part = f"I hear you. {empathy_part}" # Add a generic validation if needed

    # Get the main content (from CBT, Coping, Goal, or Resource agent - usually the last response)
    main_content_part = ""
    if len(responses) >= 3:
        main_content_part = responses[-1] # Assume the last step provides the core advice/resource/prompt
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

# Helper function to invoke the LLM with a specific persona and task
def invoke_llm_step(persona_prompt, task_prompt, user_msg, history, previous_step_output=None):
    """Invokes the LLM for a single step in the sequence."""
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    messages = [
        SystemMessage(content=persona_prompt),
        HumanMessage(content=f"""
        Conversation History:
        {history_text}

        Previous Step Output (Context):
        {previous_step_output if previous_step_output else 'N/A'}

        Current User Message:
        {user_msg}

        Your Task Now:
        {task_prompt}
        """)
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return f"Sorry, I encountered an error trying to process this step: {e}"

# Main function to process the user's message using the crewAI setup
def process_message(user_msg, history):
    """
    Process user message using sequential LLM calls simulating specialized agents,
    and return the combined response and risk score.
    """

    # 1. Assess Risk
    risk_score = is_high_risk(user_msg)
    print(f"Calculated Risk Score: {risk_score}") # Log risk score

    # 2. Define Personas and Task Instructions (derived from Agents/Tasks)
    personas = {
        "cultural": """You are an expert in Kenyan youth culture, language (Sheng, Kiswahili, English), and social dynamics.
        You understand the pressures related to education (KCPE/KCSE), family expectations, potential economic hardship, and the stigma surrounding mental health.
        You can translate when needed and ensure communication is respectful and relevant. You acknowledge spiritual beliefs respectfully *if* the user brings them up, without imposing any views.
        Your goal: Ensure responses are culturally sensitive, acknowledging Kenyan teen realities.""",
        "empathy": """You are a deeply empathetic listener who understands the complex emotional world of teenagers, including relationship dynamics (family, friends, romance).
        You validate feelings without judgment, fostering a safe space for expression. You respond with warmth and sensitivity, recognizing potential underlying distress or trauma indicators without diagnosing.
        Your goal: Show deep understanding and empathy towards teen struggles, validating feelings. Create a safe, non-judgmental space.""",
        "cbt": """You are a helper grounded in cognitive behavioral principles. You guide teens to notice negative or unhelpful thinking patterns and gently explore alternative perspectives.
        You can offer simple explanations (psychoeducation) about how thoughts influence feelings and actions. You support self-discovery and building a healthier self-concept.
        Your goal: Help teens gently identify links between thoughts, feelings, and behaviors, challenge unhelpful patterns, and explore how thoughts impact self-perception and identity. Provide simple psychoeducation.""",
        "coping": """You specialize in teaching practical, easy-to-implement coping mechanisms suitable for teens in a Kenyan context.
        You offer strategies for managing anxiety, stress, anger, and sadness, including skills for navigating difficult interactions. You provide brief, simple explanations (psychoeducation) for how these techniques work (e.g., calming the nervous system).
        Your goal: Teach practical, accessible coping strategies for stress, anxiety, and difficult emotions, including those arising from relational issues. Briefly explain *why* a technique might help.""",
        "goal": """You help teens articulate their aspirations and translate them into concrete, manageable goals.
        You encourage exploration of the 'why' behind goals, linking them to personal values and identity. You motivate and support planning, fostering a sense of agency and hope.
        Your goal: Help teens clarify meaningful, achievable goals (SMART), break them down into steps, and connect them to their values and future aspirations.""",
        "resource": """You are knowledgeable about Kenyan youth support resources, including helplines (like 116, Befrienders Kenya), community centers, reputable NGOs, and online information.
        You aim to suggest specific resources relevant to the teen's expressed need, considering potential accessibility issues (cost, location, internet) where possible, while always respecting privacy.
        Your goal: Connect teens with appropriate, relevant, and potentially accessible Kenyan resources (helplines, NGOs, online info, community support types), acknowledging potential access barriers.""",
        "crisis": """You are trained to respond calmly and supportively to teens in acute distress or crisis (e.g., expressing suicidal thoughts, reporting immediate danger).
        Your priority is safety and connection to immediate help. You provide clear, actionable information about emergency resources (like Childline 116, Befrienders Kenya) while showing care and maintaining a non-judgmental stance.
        Your goal: Provide immediate, calm, empathetic support in crisis situations, prioritize safety, and clearly provide emergency resources (Kenyan hotlines)."""
    }

    tasks_instructions = {
        "cultural": """1. Check user message language (English/Kiswahili/Sheng). Note key terms/phrases for context.
        2. Analyze for cultural nuances (school pressure, family roles, stigma, economic stress).
        3. Briefly note relevant cultural context/language for subsequent steps. Ensure sensitivity.
        4. Note mentions of spiritual beliefs for respectful acknowledgment later.
        Expected Output: Brief analysis of language and cultural context (e.g., 'User mentions KCSE stress', 'User used Sheng term X meaning Y'). Keep this concise and focused on context for the *next* step.""",
        "empathy": """1. Deeply listen to the user's message and preceding context (cultural analysis).
        2. Identify core emotions.
        3. Validate these feelings in 1-2 warm, genuine sentences. Acknowledge relational context briefly if mentioned.
        4. Reinforce safety. Avoid giving advice.
        Expected Output: A short (1-3 sentences), warm, empathetic response validating feelings and reinforcing safety.""",
        "cbt": """1. Identify a potential unhelpful thought pattern.
        2. Gently guide the user to notice this thought.
        3. Briefly explain the thought-feeling link (simple psychoeducation).
        4. Suggest exploring an alternative perspective or ask a gentle reflection question. Keep it simple.
        5. Build upon the empathy expressed in the previous step.
        Expected Output: A supportive response including: validation (building on empathy), identification of a thought pattern, simple explanation, and a gentle prompt for reflection/reframing.""",
        "coping": """1. Based on expressed feelings/situation, suggest ONE specific, practical coping technique (e.g., breathing, grounding, mindfulness, self-talk, activity).
        2. Briefly explain *why* it might help (1 sentence).
        3. Ensure it's accessible and easy.
        4. Build upon the empathy expressed in the previous step.
        Expected Output: A supportive response including: validation (building on empathy), ONE practical coping skill, and a brief explanation of its benefit.""",
        "coping_crisis": """1. Acknowledge distress with calm empathy (building on previous step).
        2. Suggest ONE simple, immediate grounding or calming technique (e.g., 5 senses, deep slow breath).
        3. Briefly explain its purpose (e.g., 'to help feel a bit calmer right now').
        4. Keep it very brief and supportive, secondary to the main crisis message.
        Expected Output: A very short (1-2 sentences) suggestion for an immediate grounding technique, offered supportively.""",
        "goal": """1. Acknowledge interest in goals/planning.
        2. Help clarify the goal or ask a question to make it more specific/achievable (SMART).
        3. Gently prompt reflection on *why* it's important (values/identity).
        4. Suggest 1-2 small initial steps.
        5. Maintain an encouraging tone. Build upon empathy.
        Expected Output: An encouraging response including: validation, help clarifying goal, prompt about importance, suggestion for initial steps.""",
        "resource": """1. Based on the issue, identify 1-2 relevant Kenyan support resources.
        2. Aim for variety (hotline, online, org).
        3. Include key contact info (116, Befrienders, website).
        4. Briefly state what the resource offers.
        5. Acknowledge potential access barriers briefly. Build upon empathy.
        Expected Output: A supportive response including: validation, 1-2 specific Kenyan resources with contact/purpose, and brief note on access challenges.""",
        "crisis": """1. Acknowledge distress with calm empathy (building on previous step).
        2. Prioritize safety: Gently express concern and importance of immediate help.
        3. Clearly provide Kenyan emergency contacts: Childline (116) and Befrienders Kenya (+254 722 178 177). Mention immediate, confidential support.
        4. Avoid judgment/complex questions. Keep message focused, supportive, directive towards help.
        5. Reinforce they are not alone and help is available *now*.
        Expected Output: A calm, direct, supportive crisis response: empathy, safety concern, Kenyan emergency numbers (116, Befrienders) with encouragement to call, reassurance."""
    }

    # 3. Execute Sequential Workflow
    step_outputs = []
    current_context = None # To store output of the previous step

    # Step 1: Cultural Analysis (Internal Context)
    print("Running Step: Cultural Analysis")
    cultural_output = invoke_llm_step(
        personas["cultural"], tasks_instructions["cultural"], user_msg, history, current_context
    )
    step_outputs.append(cultural_output)
    current_context = cultural_output # Pass to next step
    print(f"Cultural Output: {cultural_output[:100]}...") # Log snippet

    # Step 2: Empathy
    print("Running Step: Empathy")
    empathy_output = invoke_llm_step(
        personas["empathy"], tasks_instructions["empathy"], user_msg, history, current_context
    )
    step_outputs.append(empathy_output)
    current_context = empathy_output # Pass to next step
    print(f"Empathy Output: {empathy_output[:100]}...") # Log snippet

    # Step 3+ : Routing based on Risk and Keywords
    if risk_score >= 0.15:
        print(f"High risk detected (score: {risk_score}). Activating crisis workflow.")
        # Step 3: Crisis Response
        print("Running Step: Crisis")
        crisis_output = invoke_llm_step(
            personas["crisis"], tasks_instructions["crisis"], user_msg, history, current_context
        )
        step_outputs.append(crisis_output)
        current_context = crisis_output # Pass crisis output as primary context now
        print(f"Crisis Output: {crisis_output[:100]}...") # Log snippet

        # Step 4: Coping (Simple Grounding in Crisis)
        print("Running Step: Coping (Crisis Context)")
        coping_crisis_output = invoke_llm_step(
             personas["coping"], tasks_instructions["coping_crisis"], user_msg, history, current_context
        )
        step_outputs.append(coping_crisis_output) # Add coping suggestion, but crisis message takes priority
        print(f"Coping (Crisis) Output: {coping_crisis_output[:100]}...") # Log snippet

    else:
        print(f"Low risk detected (score: {risk_score}). Activating standard workflow.")
        text = user_msg.lower() # Use lowercase for keyword matching
        specialized_step_output = None

        # Determine and run ONE specialized step after empathy
        if any(k in text for k in ["goal", "target", "plan", "future", "want to", "achieve", "improve"]):
            print("Routing to: Goal Setting")
            specialized_step_output = invoke_llm_step(
                personas["goal"], tasks_instructions["goal"], user_msg, history, current_context
            )
        elif any(k in text for k in ["stress", "anxious", "worry", "overwhelmed", "coping", "relax", "breathe", "calm down", "panic"]):
            print("Routing to: Coping Skills")
            specialized_step_output = invoke_llm_step(
                personas["coping"], tasks_instructions["coping"], user_msg, history, current_context
            )
        elif any(k in text for k in ["think", "thought", "negative", "bad", "sad", "depressed", "unhappy", "feel down", "self-esteem", "confidence"]):
            print("Routing to: CBT")
            specialized_step_output = invoke_llm_step(
                personas["cbt"], tasks_instructions["cbt"], user_msg, history, current_context
            )
        elif any(k in text for k in ["friend", "family", "parents", "relationship", "dating", "argue", "fight", "lonely", "bullying"]):
             # If relational keywords detected, default to Coping skills which now includes relational aspects
             print("Routing to: Coping Skills (Relational Focus)")
             specialized_step_output = invoke_llm_step(
                 personas["coping"], tasks_instructions["coping"], user_msg, history, current_context
             )
        else:
            print("Routing to: General Support (Coping + Resources)")
            # Run Coping first
            coping_output = invoke_llm_step(
                personas["coping"], tasks_instructions["coping"], user_msg, history, current_context
            )
            step_outputs.append(coping_output)
            current_context = coping_output # Update context for the resource step
            print(f"Default Coping Output: {coping_output[:100]}...") # Log snippet
            # Then run Resource
            resource_output = invoke_llm_step(
                personas["resource"], tasks_instructions["resource"], user_msg, history, current_context
            )
            step_outputs.append(resource_output)
            print(f"Default Resource Output: {resource_output[:100]}...") # Log snippet
            specialized_step_output = None # Handled separately above

        # Add the output of the single specialized step if one was run
        if specialized_step_output:
            step_outputs.append(specialized_step_output)
            print(f"Specialized Output: {specialized_step_output[:100]}...") # Log snippet

    # 4. Combine results
    print(f"Final Step Outputs List: {step_outputs}") # Log the list of outputs before combining
    final_response = combine_agent_responses(step_outputs, risk_score)

    return final_response, risk_score

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
        # Store LangChain message objects for better context management
        st.session_state.history = [] # Stores AIMessage and HumanMessage objects
        # Add initial greeting
        initial_greeting = "Hello! I'm here to listen and support you. How are you feeling today?"
        st.session_state.history.append(AIMessage(content=initial_greeting))

    # Create a container for the chat messages for better layout control
    chat_container = st.container()

    # Display the existing conversation history within the container
    with chat_container:
        for msg_obj in st.session_state.history: # Iterate directly over message objects
            if isinstance(msg_obj, HumanMessage):
                # Using '👤' (Bust in Silhouette) for a neutral user representation
                st.chat_message("user", avatar="👤").write(msg_obj.content)
            elif isinstance(msg_obj, AIMessage):
                # Using '🦉' (Owl) for the assistant, implying wisdom/guidance
                st.chat_message("assistant", avatar="🦉").write(msg_obj.content)

    # Create a form for user input
    with st.form("message_form", clear_on_submit=True):
        # Text area for the user to type their message
        user_input = st.text_area("Type your message here (English/Kiswahili/Sheng)...", height=100, key="user_input")
        # Submit button for the form
        submitted = st.form_submit_button("Send")

    # Process the input only if the form was submitted and the input is not empty/whitespace
    if submitted and user_input.strip():
        # 1. Add the user's message to the session state history as a HumanMessage
        user_message_obj = HumanMessage(content=user_input)
        st.session_state.history.append(user_message_obj)

        # 2. Display the user's message immediately in the chat container
        with chat_container:
            # Using '👤' for user message display
            st.chat_message("user", avatar="👤").write(user_message_obj.content)

        # 3. Process the message and get the assistant's response
        try:
            # Show a "Thinking..." indicator while processing
            with chat_container:
                # Using '🦉' for assistant avatar during thinking
                with st.chat_message("assistant", avatar="🦉"):
                    thinking_placeholder = st.empty()
                    # Using '💭' (Thought Bubble) for the thinking indicator
                    thinking_placeholder.markdown("💭 _Thinking..._")

            # Call the main processing function with the user input text and current history (list of message objects)
            # Pass only the relevant history (e.g., last N messages) if needed to manage context window
            response_text, risk_score = process_message(user_input, st.session_state.history)

            # 4. Add the assistant's generated response to the history as an AIMessage
            assistant_message_obj = AIMessage(content=response_text)
            st.session_state.history.append(assistant_message_obj)

            # 5. Display the assistant's reply, replacing the "Thinking..." indicator
            with chat_container:
                 with st.chat_message("assistant", avatar="🦉"):
                      thinking_placeholder.markdown(response_text) # Update the placeholder

            # 6. Display a crisis banner if the risk score is high
            if risk_score >= 0.15:
                # Keeping '⚠️' (Warning) for the crisis banner for clarity and impact
                st.error("⚠️ If you're feeling overwhelmed or unsafe, please reach out for immediate help. Consider calling **116** (Childline Kenya - free call) or **+254 722 178 177** (Befrienders Kenya). You are not alone!")

        # Handle potential errors during processing
        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            print(f"ERROR: {error_message}") # Log the full error server-side
            st.error("I'm sorry, I encountered a technical issue and couldn't process your message properly. Please try sending it again in a moment.") # Display user-friendly error
            # Add a generic error message to the chat history as an AIMessage
            error_response_text = "Sorry, I ran into a technical problem. Could you please try sending your message again?"
            error_message_obj = AIMessage(content=error_response_text)
            st.session_state.history.append(error_message_obj)
            # Display the error message in the chat interface
            with chat_container:
                # Using '🦉' for error message display
                st.chat_message("assistant", avatar="🦉").write(error_response_text)

# Standard Python entry point
if __name__ == "__main__":
    main()
