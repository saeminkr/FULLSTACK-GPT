"""
Research Assistant using OpenAI API
Streamlit app with conversation history and research tools
"""

import streamlit as st
import re
import json
import openai
from datetime import datetime
from collections import Counter

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader


# --- Page Configuration ---
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)


# --- Tools ---
def search_wikipedia(query: str) -> str:
    try:
        tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return tool.run(query)
    except Exception as e:
        return f"Error searching Wikipedia: {repr(e)}"


def search_duckduckgo(query: str) -> str:
    try:
        api = DuckDuckGoSearchAPIWrapper(
            region="us-en",
            safesearch="moderate",
            time=None,
            max_results=8,
        )
        tool = DuckDuckGoSearchRun(api_wrapper=api)
        return tool.invoke(query)
    except Exception as e:
        return f"Error searching DuckDuckGo: {repr(e)}"


def extract_urls(query: str, ddg_text: str, max_results: int = 8) -> list[str]:
    urls = re.findall(r'https?://[^\s\]]+', ddg_text)
    if not urls:
        api = DuckDuckGoSearchAPIWrapper(
            region="us-en", safesearch="moderate", time=None, max_results=max_results
        )
        results = api.results(query, max_results=max_results)
        urls = [r.get("link") or r.get("href") for r in results if r.get("link") or r.get("href")]
    urls = [u.rstrip(').,]') for u in urls]
    seen, deduped = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def scrape_url(url: str, max_len: int = 3000) -> str:
    try:
        if not url.startswith("http"):
            return f"Invalid URL: {url}"
        docs = WebBaseLoader(url).load()
        text = "\n".join(d.page_content for d in docs)
        text = re.sub(r"\s+", " ", text).strip()
        return (text[:max_len] + "... [Content truncated]") if len(text) > max_len else text
    except Exception as e:
        return f"Error scraping {url}: {repr(e)}"


def summarize_texts(texts: list[str], max_sentences: int = 5) -> str:
    """
    Simple frequency-based extractive summary (English only).
    """
    joined = " ".join([t for t in texts if t and not t.startswith("Error scraping")]).strip()
    if not joined:
        return "(No content to summarize)"
    
    sentences = re.split(r'(?<=[\.\!\?])\s+', joined)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "(No content to summarize)"
    
    tokens = re.findall(r"[A-Za-z]+", joined.lower())
    stop = set("""
        a an the and or for of to in on with from by as is are was were be been being
        this that these those it its into at over under about you your their they we us our
        not no yes if but so than then may might can could should would will
    """.split())
    
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    if not tokens:
        return "\n- " + "\n- ".join(sentences[:max_sentences])
    
    freq = Counter(tokens)
    
    def score_sentence(s: str) -> int:
        ts = re.findall(r"[A-Za-z]+", s.lower())
        ts = [t for t in ts if t not in stop and len(t) > 2]
        return sum(freq.get(t, 0) for t in ts)
    
    scored = [(i, s, score_sentence(s)) for i, s in enumerate(sentences)]
    top = sorted(sorted(scored, key=lambda x: x[2], reverse=True)[:max_sentences], key=lambda x: x[0])
    picks = [s for _, s, _ in top]
    
    return "\n- " + "\n- ".join(picks)


def save_report(content: str, filename: str = None, title: str = "RESEARCH REPORT") -> str:
    try:
        if filename is None:
            filename = f"research_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        report = (
            f"{title}\n"
            f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            + "=" * 47 + "\n\n"
            + content + "\n\n"
            + "=" * 47 + "\nEnd of Report\n"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        return f"Saved: {filename}"
    except Exception as e:
        return f"Error saving file: {repr(e)}"

# --- JUST DO IT GOGOGO ---

def run_research(query: str, scrape_top_k: int = 3) -> str:
    """
    End-to-end research function
    """
    parts = []
    
    # 1) Wikipedia
    wiki = search_wikipedia(query)
    parts.append(f"**WIKIPEDIA RESULTS:**\n{wiki[:2000]}\n")
    
    # 2) DuckDuckGo
    ddg_text = search_duckduckgo(query)
    parts.append(f"\n**DUCKDUCKGO RESULTS:**\n{ddg_text[:2000]}\n")
    
    # 3) Extract URLs and scrape
    urls = extract_urls(query, ddg_text, max_results=8)
    scraped_texts = []
    for i, u in enumerate(urls[:scrape_top_k]):
        body = scrape_url(u)
        scraped_texts.append(body)
        if i == 0:  # Only show first website content preview
            parts.append(f"\n**WEBSITE CONTENT (from {u[:50]}...):**\n{body[:1000]}...\n")
    
    # 4) Auto-summary
    if scraped_texts:
        summary = summarize_texts(scraped_texts, max_sentences=5)
        parts.append(f"\n**KEY FINDINGS:**\n{summary}")
    

    # # 5) Save
    # full = "\n".join(parts)
    # print(save_report(full))  # Print save path
    # return full

    return "\n".join(parts)


# --- API Key Testing ---
def test_openai_key(api_key: str) -> bool:
    """Test if the OpenAI API key is valid"""
    try:
        openai.api_key = api_key
        # Try to list models to test the key
        openai.Model.list()
        return True
    except Exception:
        return False


# --- GPT Functions ---
def get_gpt_response(messages: list, api_key: str) -> str:
    openai.api_key = api_key
    
    functions = [
        {
            "name": "run_research",
            "description": "Research a topic using Wikipedia and web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query or topic to investigate"
                    },
                    "scrape_top_k": {
                        "type": "integer",
                        "description": "Number of top websites to scrape (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=2000
        )
        
        message = response.choices[0].message
        
        # Check if GPT wants to call a function
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            function_args = json.loads(message["function_call"]["arguments"])
            
            if function_name == "run_research":
                # Execute the research function
                function_response = run_research(
                    query=function_args.get("query"),
                    scrape_top_k=function_args.get("scrape_top_k", 3)
                )
                
                # Get final response from GPT with the function result
                messages.append(message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": function_response
                })
                
                final_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                return final_response.choices[0].message["content"]
        
        return message["content"]
        
    except Exception as e:
        return f"Error getting response: {str(e)}"


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gpt_messages" not in st.session_state:
    st.session_state.gpt_messages = [
        {
            "role": "system",
            "content": """You are a helpful research assistant. You can search Wikipedia and the web to find information.
            When users ask research questions, use the run_research function to gather information and provide comprehensive answers.
            Always cite your sources and provide a balanced view of the topic.
            Format your responses with clear sections and bullet points when appropriate."""
        }
    ]


# --- Sidebar ---
with st.sidebar:
    st.markdown("## Settings")
    
    # API Key input
    st.markdown("### OpenAI API Key")
    
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False
    if "stored_api_key" not in st.session_state:
        st.session_state.stored_api_key = ""
    
    api_key_valid = st.session_state.api_key_valid
    stored_api_key = st.session_state.stored_api_key
    
    if not api_key_valid:
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Enter your OpenAI API Key to use the chatbot."
        )
        
        if api_key:
            if st.button("Test API Key"):
                with st.spinner("Testing API Key..."):
                    result = test_openai_key(api_key)
                    if result == True:
                        st.success("API Key is valid! ✓")
                        st.session_state.api_key_valid = True
                        st.session_state.stored_api_key = api_key
                        st.rerun()
                    else:
                        st.error(f"API Key is invalid. Please try again.")
    else:
        st.success("✅ API Key verified!")
        if st.button("Change API Key"):
            st.session_state.api_key_valid = False
            st.session_state.stored_api_key = ""
            st.session_state.messages = []
            st.session_state.gpt_messages = [st.session_state.gpt_messages[0]] 
            st.rerun()
    
    st.markdown("---")
    
    # GitHub link
    st.markdown("### Links")
    st.markdown("[View Code on GitHub](https://github.com/saeminkr/FULLSTACK-GPT)")
    st.markdown("[View Streamlit](https://investorgpt-2025.streamlit.app/)")
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.gpt_messages = [st.session_state.gpt_messages[0]]  # Keep system message
        st.rerun()
    
    # Download report button
    if st.button("💾 Save Last Research"):
        if st.session_state.messages:
            # Find last assistant message with research
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    result = save_report(msg["content"])
                    st.success(result)
                    break


# --- Main Interface ---
st.title("🔍 Research Assistant")
st.markdown("AI-powered research assistant with Wikipedia and web search capabilities")

# Check if API key is valid
if not st.session_state.api_key_valid:
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to start.")
    st.info("You can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
    st.stop()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a research question (e.g., 'Research about the XZ backdoor')"):
    # Add message history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.gpt_messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Researching... This may take a moment while I search and analyze information."):
            response = get_gpt_response(
                st.session_state.gpt_messages.copy(),
                st.session_state.stored_api_key
            )
            
            st.markdown(response)
            
            # Add assistant response to message history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.gpt_messages.append({"role": "assistant", "content": response})