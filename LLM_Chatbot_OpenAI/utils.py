import openai
# Function to call GPT-4 API
def call_gpt4(prompt, api_key):
    """
    Sends a prompt to the GPT-4 API and retrieves the response.

    This function interacts with the GPT-4 model using the OpenAI API. 
    It includes a system prompt to establish the assistant's behavior and processes the user prompt to generate a response.

    Parameters:
        prompt (str): The user input or prompt to be processed by GPT-4.
        api_key (str): The API key required to authenticate with OpenAI.

    Returns:
        str: The content of GPT-4's response if successful. 
             In case of an error, an error message is returned as a string.
    """
    try:
        # Set the OpenAI API key
        openai.api_key = api_key

        # Call the GPT-4 API with the provided prompt
        response = openai.chat.completions.create(
            model="gpt-4",  # Specify the GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant called KnowItAll."},  # System message to guide the AI's behavior
                {"role": "user", "content": prompt},  # User's input
            ],
        )

        # Return the content of the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Handle exceptions and return error messages
        return f"Error: {e}"


# Predefined System Prompts
def get_system_prompt(feature):
    """
    Provides a predefined system prompt tailored to the selected feature.

    The system prompt defines the AI's behavior based on the user's selected feature 
    (e.g., answering questions, generating content, translating text, or writing code).

    Parameters:
        feature (str): The feature selected by the user (e.g., "Ask a Question").

    Returns:
        str: A predefined system prompt string describing the behavior of the assistant.
             If the feature is unrecognized, a default prompt is returned.
    """
    system_prompts = {
        "Ask a Question": "You are an expert at answering questions.",
        "Generate Content": "You are a creative writer. Generate well-written content.",
        "Translate Text": "You are a professional translator. Translate the text accurately.",
        "Write Code": "You are an expert coder. Write efficient and readable code."
    }

    # Return the prompt based on the feature or a default behavior
    return system_prompts.get(feature, "You are a helpful assistant.")


# Feature-Specific Prompt Generator
def generate_feature_prompt(feature, user_input):
    """
    Generates a prompt specific to the selected feature and the user's input.

    This function combines the feature type with the user's input to create a 
    specialized prompt for GPT-4. It customizes the input depending on the 
    functionality requested by the user.

    Parameters:
        feature (str): The feature selected by the user (e.g., "Generate Content").
        user_input (str): The user's input related to the selected feature.

    Returns:
        str: A tailored prompt for GPT-4, ready for processing.
    """
    if feature == "Ask a Question":
        # Use the user's input directly for questions
        return user_input
    elif feature == "Generate Content":
        # Generate creative content based on the input
        return f"Generate content for: {user_input}"
    elif feature == "Translate Text":
        # Translate the provided text
        return f"Translate this text: {user_input}"
    elif feature == "Write Code":
        # Generate code based on the input
        return f"Write code for: {user_input}"
    else:
        # Default to the user's input for unknown features
        return user_input
