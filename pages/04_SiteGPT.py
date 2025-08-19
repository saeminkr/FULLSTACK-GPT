from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from bs4 import BeautifulSoup

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    api_key = st.session_state.get("stored_api_key", "")
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key,
    )
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    api_key = st.session_state.get("stored_api_key", "")
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key,
    )
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']} \n Source:{answer['source']}"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )




def test_openai_key(api_key):
    # OpenAI API Key 유효성 테스트
    try:
        test_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key,
            max_tokens=10
        )
        test_llm.invoke("test")
        return True
    except Exception as e:
        return False, str(e)


def clean_cloudflare_content(soup):
    # 불필요한 요소들 제거
    unwanted_elements = ["header", "footer", "nav", "aside"]
    for tag in unwanted_elements:
        elements = soup.find_all(tag)
        for element in elements:
            element.decompose()
    
    # Cloudflare 특화 불필요 요소 제거
    breadcrumbs = soup.find_all(class_=lambda x: x and "breadcrumb" in str(x).lower())
    for bc in breadcrumbs:
        bc.decompose()
        
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal", "")
    )


@st.cache_data(show_spinner="Loading documents from cloudflare...")
def load_cloudflare_docs_sitemap(api_key):
    sitemap_url = "https://developers.cloudflare.com/sitemap-0.xml"
    
    # URL 패턴으로 특정 섹션만 필터링
    target_patterns = [
        r"^(.*\/ai-gateway\/).*",
        r"^(.*\/vectorize\/).*", 
        r"^(.*\/workers-ai\/).*"
    ]
    
    # 제외할 URL 패턴 (토큰 사용량 최적화)
    exclude_patterns = [
        r".*\/workers-ai\/models\/[^\/]+\/?$",  # 개별 모델 페이지
        r".*\/ai-gateway\/usage\/providers\/[^\/]+\/?$",  # 개별 프로바이더 페이지
        r".*\/tutorials\/[^\/]+\/?$",  # 상세 튜토리얼
        r".*\/examples\/[^\/]+\/?$",  # 개별 예제 페이지
        r".*\/reference\/[^\/]+\/?$",  # 상세 레퍼런스 (메인 레퍼런스는 유지)
    ]
    
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800,
            chunk_overlap=150
        )
        
        loader = SitemapLoader(
            sitemap_url,
            filter_urls=target_patterns,
            parsing_function=clean_cloudflare_content
        )
        
        # 리퀘스트 속도 제한
        loader.requests_per_second = 3
        loader.headers = {"User-Agent": "Mozilla/5.0 (compatible; SiteGPT/1.0)"}
        
        documents = loader.load_and_split(text_splitter=splitter)
            
        import re
        filtered_documents = []
        for doc in documents:
            url = doc.metadata.get('source', '')
            exclude = any(re.match(pattern, url) for pattern in exclude_patterns)
            if not exclude:
                filtered_documents.append(doc)
        
        documents = filtered_documents
        
        st.success(f"✅ Cloudflare documentation loaded successfully! ({len(documents)} chunks)")
        
        # 벡터 스토어 생성
        vector_store = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=api_key))
        return vector_store.as_retriever()
        
    except Exception as e:
        st.error(f"Error loading sitemap: {str(e)}")
        return None


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about Cloudflare documentation.
            
    This chatbot can answer questions about:
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
            
    Enter your OpenAI API key in the sidebar to get started. The documentation will be automatically loaded.
"""
)


with st.sidebar:
    st.markdown("## Settings")
    
    # API Key 입력
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
            st.rerun()
    
    st.markdown("---")
    
    # GitHub 링크
    st.markdown("### Links")
    st.markdown("[View Code on GitHub](https://github.com/saeminkr/FULLSTACK-GPT)")
    st.markdown("[View Streamlit](https://sitegpt-2025.streamlit.app/)")


# 메인 콘텐츠
if not st.session_state.api_key_valid:
    st.markdown(
        """
    Welcome to SiteGPT for Cloudflare Documentation.
                
    I can answer questions about Cloudflare's AI Gateway, Vectorize, and Workers AI documentation.
                
    Please enter your OpenAI API Key in the sidebar to get started.
    """
    )
else:
    # API 키가 유효하면 자동으로 Cloudflare 문서 로드
    retriever = load_cloudflare_docs_sitemap(st.session_state.stored_api_key)
    
    if retriever:
        query = st.text_input("Ask a question about Cloudflare documentation.")
        if query:
            with st.spinner("Searching documentation..."):
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                result = chain.invoke(query)
                st.markdown(result.content.replace("$", "\$"))
    else:
        st.error("Failed to load Cloudflare documentation. Please try refreshing the page.")