import streamlit as st 
from langchain_experimental.agents.agent_toolkits import create_csv_agent
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
load_dotenv()
import os
groq_api_key = os.getenv("")
st.set_page_config(page_title="Talk to your CSV")
st.header("Talk to your CSV")
csv_file = st.file_uploader("Upload a CSV file", type="csv")

if csv_file is not None:
    agent = create_csv_agent(
        ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768"),
        csv_file,
        verbose = True,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    user_question = st.text_input("Ask a question about your CSV:")

    if user_question is not None and user_question != "":
        with st.spinner(text = "In progress..."):
            response = agent.invoke(user_question)
            st.write(response["output"])