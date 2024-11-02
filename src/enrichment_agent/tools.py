"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import json
from typing import Any, Optional, cast

import aiohttp
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State
from enrichment_agent.utils import init_model


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


_INFO_PROMPT = """You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

<Website content>
{content}
</Website content>"""


async def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Scrape and summarize content from a given URL.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    p = _INFO_PROMPT.format(
        info=json.dumps(state.code, indent=2),
        url=url,
        content=content[:40_000],
    )
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    return str(result.content)

####################################  original tools  ##########################################################
from typing import Literal, List, Union
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

async def search_about_nyanta() -> str:
    """data about nyanta.

    this function return a result about nyanta
    """
    return "nyanta sleep at 10pm"

####################################  Define Schema  ##########################################################

# 各行のデータを表現するスキーマ
class Condition(BaseModel):
    """Condition being evaluated sepalatery"""

    condition: str
    value: Union[bool, Literal["-", "N/A"]]

# 各行のデータを表現するスキーマ
class TestCase(BaseModel):
    """Test Case"""

    case_no: int
    conditions: List[Condition]
    output: str

# 複数行を包括するスキーマ
class TestCases(BaseModel):
    """Test Case table to tell user."""

    test_cases: List[TestCase]  # 複数のTestCaseを包括

####################################  Define agent tools  ##########################################################

_TEST_CASE_TABLE_PROMPT = """You are making a test case table. You are trying to make test case table about this code:

<code>
{code}
</code>

You create test cases for branch coverage based on the provided, pre-listed conditions.: {conditions}

Based on the given code and listed up conditions, make a test case table.
"""

# this is the agent function that will be called as tool
# notice that you can pass the state to the tool via InjectedState annotation
def agent_make_test_case_table(
    code: str,
    conditions: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],) -> TestCases:
    """Make test case table from a given code

    Returns:
        TestCases: A Table of the test cases.
    """

    p = _TEST_CASE_TABLE_PROMPT.format(
        code=code,
        conditions=conditions,
    )
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    raw_model = init_model(config)
    structured_model = raw_model.with_structured_output(TestCases)
    response = structured_model.invoke(p)
    # return the LLM response as a string (expected tool response format)
    # this will be automatically turned to ToolMessage
    # by the prebuilt create_react_agent (supervisor)
    return response


_CONDITiONS_LISTUP_PROMPT = """
You are test case maker. Extract all conditional statements from the provided code, including if, else if, or else blocks. For each condition, determine if it is sequential in relation to previous conditions.
Use the following guidelines:
    A condition is sequential if its evaluation depends on the outcome of a previous condition (e.g., part of an if-elif chain or an else block).
    If no sequential relationship exists, the condition can be considered independent by default and doesn't need to be explicitly labeled.
    Label the conditions with their exact expressions as they appear in the code and indicate whether they are sequential by specifying which condition it depends on.
Example:
    Input code:
    def check_conditions(X, Y, Z):
        if X > 5:
            if Y == 10:
                print("Y is 10")  # Sequential condition
            if Z < 0:
                print("Z is negative")  # Sequential condition
        else:
            print("X is not greater than 5")  # Sequential condition

    Expected output:

    Extracted Conditions and Sequential Relationships:
    1 X > 5
    Sequential: None (Evaluated first, no prior condition)

    2 Y == 10
    Sequential: X > 5 (Evaluated only if X > 5 is True)

    3 Z < 0
    Sequential: X > 5, Y == 10 (Evaluated only if both X > 5 and Y == 10 are True)

    4 else (X <= 5)
    Sequential: X > 5 (Executed if X > 5 is False)

Listing up conditions of following code:
<code>
{code}
</code>
"""

def agent_listingup_conditions(
    code: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],) -> str:
    """Listing up conditions from given code. This is useful for the first step to make test case table

    Returns:
        str: List of conditioins and descriptioin about this condition
    """


    p = _CONDITiONS_LISTUP_PROMPT.format(code=code)

    raw_model = init_model(config)
    response = raw_model.invoke(p)
    return response.content
