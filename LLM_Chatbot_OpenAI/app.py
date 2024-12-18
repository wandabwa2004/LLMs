import openai
import streamlit as st
from utils import call_gpt4, get_system_prompt, generate_feature_prompt

# App Title
st.title("KnowItAll Chatbot 🤖")

# **API Key Input Section**
# Check if an API key exists in the session state; if not, initialize it to None.
if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

# Sidebar for API Key Input
st.sidebar.header("API Key Setup")
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key",  # Prompt text for the API key input field
    type="password",  # Mask the input for security
    placeholder="sk-...",  # Example format for user guidance
    help="Your OpenAI API key will be used to interact with GPT-4.",  # Tooltip help
)

# **Save API Key**
# Button in the sidebar to save the API key
if st.sidebar.button("Save API Key"):
    # Basic validation for API key format
    if api_key_input.startswith("sk-") and len(api_key_input) > 30:
        st.session_state["api_key"] = api_key_input
        st.sidebar.success("API Key saved successfully!")  # Confirmation message
    else:
        st.sidebar.error("Invalid API Key. Please check and try again.")  # Error message for invalid keys

# **Ensure API Key is Set**
# If no API key is found in the session state, display a warning.
if not st.session_state["api_key"]:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
else:
    # **Features Section**
    # Sidebar dropdown for selecting the feature to use
    st.sidebar.header("Features")
    feature = st.sidebar.selectbox(
        "What would you like to do?",  # Dropdown label
        ["Ask a Question", "Generate Content", "Translate Text", "Write Code"],  # Feature options
    )

    # **User Input Area**
    # Text area for the user to input their prompt or question
    user_input = st.text_area("Enter your input:")

    # **Submit Button and GPT-4 Call**
    # Trigger GPT-4 API call when the button is clicked
    if st.button("Submit"):
        if user_input.strip():  # Ensure the input is not empty
            # Generate the system and tailored prompts based on the selected feature
            system_prompt = get_system_prompt(feature)
            tailored_prompt = generate_feature_prompt(feature, user_input)
            
            # Call GPT-4 using the tailored prompt and user's API key
            response = call_gpt4(prompt=tailored_prompt, api_key=st.session_state["api_key"])
            
            # **Output Display**
            # Display the GPT-4 response under a subheader
            st.subheader(f"Output for '{feature}':")
            if feature == "Write Code":
                # Show the output as a code block for the "Write Code" feature
                st.code(response, language="python")
            else:
                # Show the output as plain text for other features
                st.write(response)
        else:
            # Display a warning if the user input is empty
            st.warning("Please enter some input.")
