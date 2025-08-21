from langchain.schema import SystemMessage
import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper

llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-1106")

alpha_vantage_api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHA_VANTAGE_API_KEY")

if not alpha_vantage_api_key:
    st.error("⚠️ ALPHA_VANTAGE_API_KEY is not set in secrets.toml or environment variables. Please set it to use the financial data features.")
    st.stop()

# Display API key status
if alpha_vantage_api_key == "demo":
    st.warning("⚠️ Using demo API key. This has very limited requests per day. Consider getting a free API key from Alpha Vantage.")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
    )


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    
    """
    args_schema: Type[
        StockMarketSymbolSearchToolArgsSchema
    ] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        try:
            ddg = DuckDuckGoSearchAPIWrapper()
            return ddg.run(query)
        except Exception as e:
            return f"Search Error: {str(e)}"


class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(
        description="Stock symbol of the company.Example: AAPL,TSLA",
    )


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        try:
            r = requests.get(
                f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
            )
            r.raise_for_status()
            data = r.json()
            if "Error Message" in data:
                return f"Error: {data['Error Message']}"
            if "Note" in data:
                return f"API Limit Reached: {data['Note']}"
            return data
        except requests.exceptions.RequestException as e:
            return f"HTTP Error: {str(e)}"


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        try:
            r = requests.get(
                f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
            )
            r.raise_for_status()
            data = r.json()
            if "Error Message" in data:
                return f"Error: {data['Error Message']}"
            if "Note" in data:
                return f"API Limit Reached: {data['Note']}"
            return data.get("annualReports", [])
        except requests.exceptions.RequestException as e:
            return f"HTTP Error: {str(e)}"


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        try:
            r = requests.get(
                f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
            )
            r.raise_for_status()
            data = r.json()
            if "Error Message" in data:
                return f"Error: {data['Error Message']}"
            if "Note" in data:
                return f"API Limit Reached: {data['Note']}"
            if "Weekly Time Series" not in data:
                return "Error: Weekly Time Series data not available"
            return list(data["Weekly Time Series"].items())[:200]
        except requests.exceptions.RequestException as e:
            return f"HTTP Error: {str(e)}"


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.
            
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
            Consider the performance of a stock, the company overview and the income statement.
            
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """
        )
    },
)

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT
            
    Welcome to InvestorGPT.
            
    Write down the name of a company and our Agent will do the research for you.
"""
)

company = st.text_input("Write the name of the company you are interested on.")

if company:
    try:
        with st.spinner(f"Researching {company}..."):
            result = agent.invoke(company)
            st.write(result["output"].replace("$", "\$"))
    except Exception as e:
        st.error(f"An error occurred while researching {company}: {str(e)}")
        st.error("Please try again with a different company name or check if the API services are available.")