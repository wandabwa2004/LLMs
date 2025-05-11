
# Teen Therapy Chat (Kenya)

Welcome to the Teen Therapy Chat (Kenya) project! This AI-powered chatbot is designed to provide a supportive, culturally sensitive, and accessible space for Kenyan teenagers to share their feelings, explore challenges, and receive guidance.
 
## Overview
 
Mental well-being is crucial, especially for teenagers navigating the complexities of adolescence in countries like Kenya. This project aims to leverage Large Language Models (LLMs) to create an AI assistant that can:
 
 *   Offer empathetic listening and validation.
 *   Provide culturally relevant support, understanding the specific context of Kenyan youth.
 *   Suggest basic coping mechanisms and CBT-inspired reflections.
 *   Guide users towards setting positive goals.
 *   Connect users with local Kenyan resources in times of need.
 *   Identify and respond appropriately to crisis situations by providing emergency contact information.
 
This repository explores different architectural approaches for building such a chatbot, including:
 *   **Sequential LLM Calls (`app_v2.py`):** Simulating specialized agents through a sequence of targeted LLM interactions.
 *   **Multi-Agent System with `crewAI` (`app_v3.py`):** Utilizing the `crewAI` framework to orchestrate a team of specialized AI agents.
 
## Key Features
 
 * **Culturally Aware Responses:** Tailored to Kenyan youth culture, common languages (English, Kiswahili, Sheng), and social dynamics (e.g., school pressures like KCPE/KCSE, family expectations).
 *  **Empathetic Listening:** Aims to validate feelings and foster a non-judgmental environment for users to express themselves.
 *   **Simulated Therapeutic Modalities:**
     *   **CBT-inspired Reflection:** Helps teens gently identify links between thoughts, feelings, and behaviors.
     *   **Coping Skills Coaching:** Suggests practical and accessible coping strategies for stress and difficult emotions.
     *   **Goal-Setting Guidance:** Assists in clarifying and planning personal goals.
 *   **Crisis Intervention:** Includes a basic risk assessment to identify messages indicating acute distress. In such cases, it prioritizes safety and provides contact information for Kenyan emergency hotlines (e.g., Childline Kenya 116, Befrienders Kenya).
 *   **Resource Connection:** Suggests relevant local Kenyan support resources like helplines and NGOs.
 *   **Interactive Chat Interface:** Built using Streamlit for easy web-based access.
 
 ## Technology Stack
 
 *   **Programming Language:** Python
 *   **LLM Interaction:** OpenAI (GPT-4o-mini) via LangChain
 *   **Agent Framework (for `app_v3.py`):** `crewAI`
 *   **Web Framework:** Streamlit
 *   **Environment Management:** `python-dotenv`
 *   **Core Libraries:** `langchain`, `langchain-openai`
 
 ## Project Structure
 
 A brief overview of the key files:
 
 ```
 ├── app_v2.py         # Chatbot implementation using sequential LLM calls
 ├── app_v3.py         # Chatbot implementation using the crewAI framework
 ├── .env.example      # Example environment file
 ├── requirements.txt  # Python dependencies

 ```
  
 ## Setup and Installation
 
 1.  **Clone the Repository:**
     ```bash
     git clone <your-repository-url>
     cd <repository-name>
     ```
 
 2.  **Create and Activate a Virtual Environment:**
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
 
 3.  **Install Dependencies:**
     ```bash
     pip install -r requirements.txt
     ```
 
 4.  **Set Up Environment Variables:**
     *   Create a `.env` file in the root directory by copying `.env.example` (if you provide one) or creating it manually.
     *   Add your OpenAI API key to the `.env` file:
         ```env
         OPENAI_API_KEY="your_openai_api_key_here"
         ```
 
 ## Running the Application
 
 You can run either version of the chatbot:
 
 *   **To run the version with sequential LLM calls:**
     ```bash
     streamlit run app_v2.py
     ```
 
 *   **To run the version with `crewAI`:**
     ```bash
     streamlit run app_v3.py
     ```
 
 Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
 
 ## How It Works (Conceptual Flow)
 
 1.  **User Input:** The user types a message into the chat interface.
 2.  **Risk Assessment:** The application performs a basic keyword-based analysis to assess if the message indicates a high-risk situation.
 3.  **Agent/Task Orchestration:**
     *   Based on the risk score and the content of the user's message, a sequence of specialized AI agents (or LLM calls simulating these roles) is invoked.
     *   These agents are designed to handle specific aspects of the conversation:
         *   **Cultural Context:** Analyzes cultural nuances.
         *   **Empathy:** Validates feelings.
         *   **Specialized Support:** Depending on the input, agents focused on CBT principles, coping skills, goal-setting, or resource provision might be engaged.
         *   **Crisis Management:** If high risk is detected, the crisis agent takes priority to provide emergency information.
 4.  **Response Generation:** The outputs from the activated agents/tasks are synthesized into a single, coherent, and supportive response.
 5.  **Display:** The AI's response is displayed in the chat interface.
 
 ## Future Enhancements
 
 This project has a lot of potential for growth. Some ideas for future development include:
 
 *   **Personalized Long-Term Memory:** Enabling the chatbot to remember key themes from past (anonymized and opt-in) interactions for more continuous support.
 *   **Interactive Therapeutic Tools:** Integrating simple guided exercises (e.g., mood journaling, CBT thought records, guided mindfulness) and  contacts to therapists directly within the chat.
 *   **Expanded and Verified Local Resource Database:** Continuously updating and verifying the list of Kenyan mental health resources.
 *   **Advanced Risk Assessment:** Implementing more sophisticated NLP techniques for risk detection.
 *   **User Feedback Mechanism:** Allowing users to provide feedback on the helpfulness of responses.
 *   **Multilingual Enhancements:** Improving support for Sheng and Kiswahili beyond basic understanding.
 
 ## Contributing
 
 Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to:
 
 1.  Fork the repository.
 2.  Create a new branch (`git checkout -b feature/your-feature-name`).
 3.  Make your changes.
 4.  Commit your changes (`git commit -m 'Add some feature'`).
 5.  Push to the branch (`git push origin feature/your-feature-name`).
 6.  Open a Pull Request.
 
 Please ensure your code follows good practices and includes relevant comments or documentation.
 
 ## License
 
 Consider adding an open-source license to this project (e.g., MIT License, Apache 2.0). If you do, create a `LICENSE` file and mention it here. For example:
 
 This project is licensed under the MIT License - see the LICENSE file for details.
