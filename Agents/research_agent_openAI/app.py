# app.py
import streamlit as st
from research_agent import ResearchAgent

def main():
    st.title("Research Agent Demo")

    # Text input for the user to provide their OpenAI API key
    openai_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password"
    )

    # Button to set or update the key
    if st.button("Set Key"):
        if openai_key.strip():
            # If a key was provided, create/update the agent in session state
            st.session_state["agent"] = ResearchAgent(openai_api_key=openai_key)
            st.success("OpenAI key set successfully!")
        else:
            st.warning("Please provide a valid OpenAI key.")

    # Check if we have an agent in session
    if "agent" not in st.session_state:
        st.info("Please enter your OpenAI key to start using the Research Agent.")
        return  # Stop here, because no agent is available yet

    # Otherwise, proceed with normal functionality
    query = st.text_input("Enter your query:")
    
    if st.button("Run Research"):
        with st.spinner("Running research..."):
            response = st.session_state["agent"].research(query)

        if response["success"]:
            st.subheader("Result:")
            st.write(response["result"])
        else:
            st.error(f"Error: {response['error']}")

    if st.checkbox("Show Chat History"):
        st.write("### Chat History")
        messages = st.session_state["agent"].get_chat_history()
        for msg in messages:
            role = "User" if msg.type == "human" else "Agent"
            st.markdown(f"**{role}:** {msg.content}")

if __name__ == "__main__":
    main()
