import streamlit as st
from research_agent import ResearchAgent

# Set up the Streamlit app
def main():
    # Add a title with an icon
    st.title("🔎 Research Agent with LLaMA/Falcon")
    st.markdown("""
    Welcome to the **Research Agent**! 🚀  
    Use state-of-the-art AI models like **Falcon** and **LLaMA** for:
    - 🔍 Web searches
    - 📝 Text summarization
    - 🧠 Intelligent responses
    """)

    # Hugging Face token input section
    st.subheader("🔑 Hugging Face Token")
    st.markdown("Enter your **Hugging Face Token** to access gated models and advanced features.")
    
    # Input for Hugging Face token
    hf_token = st.text_input(
        "Enter your Hugging Face Token:",  # Label
        type="password",  # Hide the input text for security
        placeholder="Your Hugging Face token here..."  # Placeholder text
    )

    # Button to set or update the token
    if st.button("🔒 Set Token"):
        if hf_token.strip():
            # Save the token and create the agent in the session state
            st.session_state["agent"] = ResearchAgent(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                hf_token=hf_token
            )
            st.success("Hugging Face token set successfully! ✅")
        else:
            st.warning("⚠️ Please provide a valid Hugging Face token.")

    # Check if an agent has been created
    if "agent" not in st.session_state:
        # Display a message if no agent is available
        st.info("ℹ️ Please enter your Hugging Face token to start using the Research Agent.")
        return  # Stop execution until the token is provided

    # Query input section
    st.subheader("📝 Ask the Agent")
    query = st.text_input(
        "Enter your query:",
        placeholder="E.g., 'Summarize the latest AI research trends.'"
    )

    # Button to run the research query
    if st.button("🚀 Run Research"):
        # Show a spinner while processing
        with st.spinner("🧠 Thinking..."):
            response = st.session_state["agent"].research(query)

        # Display the results
        if response["success"]:
            st.subheader("💡 Result:")
            st.write(response["result"])
        else:
            st.error(f"❌ Error: {response['error']}")

    # Option to show conversation history
    if st.checkbox("🕒 Show Chat History"):
        st.subheader("📜 Chat History")
        messages = st.session_state["agent"].get_chat_history()
        for msg in messages:
            role = "👤 User" if msg.type == "human" else "🤖 Agent"
            st.markdown(f"**{role}:** {msg.content}")

# Run the Streamlit app
if __name__ == "__main__":
    main()