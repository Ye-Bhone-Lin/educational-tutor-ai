from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import Tool

def ddg_search(query: str):
    ddg = DuckDuckGoSearchResults()
    return ddg.invoke(query)

website_tool = Tool(
    name = "Website Tool",
    func=ddg_search,
    description = "Help the STUDNETS to know better life time"
)

###