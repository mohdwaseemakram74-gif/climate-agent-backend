import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# ---- 1. LLM (free Groq API) ----
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"],
)

# ---- 2. Tools the agent can use ----
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# ---- 3. Agent prompt ----
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an autonomous research agent focused on UN Sustainable "
            "Development Goal 13: Climate Action. Given a topic, you must: "
            "1) search the web for current, factual information, "
            "2) synthesize what you find into a clear structured report with "
            "sections: Overview, Key Facts, Current Challenges, Recommended "
            "Actions. Always cite where facts came from in plain text "
            "(e.g. 'according to [source]'). Keep the report concise but "
            "substantive (around 300-500 words). Do not fabricate statistics.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# ---- 4. Build the agent ----
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_research_agent(topic: str) -> str:
    """Runs the autonomous agent on a given climate-action topic and returns the report text."""
    result = agent_executor.invoke({"input": topic})
    return result["output"]
