from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader, WebBaseLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# OpenAI API Key 설정 함수
def create_llm(api_key):
    return ChatOpenAI(
        temperature=0.1,
        model_name="gpt-3.5-turbo",
        streaming=True,
        openai_api_key=api_key,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs(f"./.cache/embeddings/{file.name}", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


# 저번 과제의 웹 콘텐츠 사용
@st.cache_data(show_spinner="Loading web content...")
def embed_web_content(api_key):
    cache_name = "nineteen_eighty_four"
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs(f"./.cache/embeddings/{cache_name}", exist_ok=True)
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{cache_name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = WebBaseLoader("https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
    
    # 메모리에 저장
    if "memory" in st.session_state and st.session_state["memory"]:
        if role == "human":
            st.session_state["memory_input"] = message
        elif role == "ai" and "memory_input" in st.session_state:
            st.session_state["memory"].save_context(
                {"input": st.session_state["memory_input"]},
                {"output": message}
            )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(_):
    if "memory" in st.session_state and st.session_state["memory"]:
        return st.session_state["memory"].load_memory_variables({})["chat_history"]
    return []

def test_openai_key(api_key):
    """OpenAI API Key 유효성 테스트"""
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


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


st.title("📃 DocumentGPT")

st.markdown(
    """
Hello!
            
This is a chatbot that can ask questions about the uploaded file to AI.

**How to use:**
1. Enter and test your OpenAI API Key in the sidebar
2. Choose to upload your own file or use the pre-loaded novel
3. Start chatting with AI about the content!
"""
)

with st.sidebar:
    # API Key 상태 체크
    api_key_valid = st.session_state.get("api_key_valid", False)
    stored_api_key = st.session_state.get("stored_api_key", "")
    
    # API Key 입력
    if not api_key_valid:
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Enter your OpenAI API Key to use the chatbot."
        )
        
        # API Key 테스트 버튼
        if api_key:
            if st.button("Test API Key"):
                with st.spinner("Testing API Key..."):
                    result = test_openai_key(api_key)
                    if result == True:
                        st.success("API Key is valid! ✓")
                        st.session_state["api_key_valid"] = True
                        st.session_state["stored_api_key"] = api_key
                        st.rerun()
                    else:
                        st.error(f"API Key is invalid. Please try again.")
                        st.session_state["api_key_valid"] = False
    else:
        # API Key 인증 완료
        st.success("✅ API Key verified!")
        if st.button("Change API Key"):
            st.session_state["api_key_valid"] = False
            st.session_state["stored_api_key"] = ""
            st.session_state["messages"] = []
            if "memory" in st.session_state:
                st.session_state["memory"].clear()
            st.rerun()
                
        # 콘텐츠 선택
        content_source = st.radio(
            "Choose content source:",
            ["📁 Upload File", "🌐 Load Novel (Nineteen Eighty-Four)"],
            help="Choose whether to upload your own file or use the pre-loaded novel."
        )
        
        # 세션 상태 저장
        st.session_state["content_source"] = content_source
        
        if content_source == "📁 Upload File":
            # 파일 업로드
            file = st.file_uploader(
                "Upload a file (.txt, .pdf, .docx)",
                type=["pdf", "txt", "docx"],
                help="Upload a file to analyze."
            )
            st.session_state["uploaded_file"] = file
        else:
            # 저번 과제의 웹 콘텐츠 사용
            st.info("📖 Using novel: **Nineteen Eighty-Four** by George Orwell")
            st.session_state["uploaded_file"] = None
        
        st.markdown("---")
        
        # 채팅 기록 삭제 버튼
        if st.button("🗑️ Clear Chat History"):
            st.session_state["messages"] = []
            if "memory" in st.session_state:
                st.session_state["memory"].clear()
            st.success("Chat history cleared.")
            st.rerun()
    
    # API key 설정
    api_key = stored_api_key if api_key_valid else st.session_state.get("temp_api_key", "")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memory" not in st.session_state:
    st.session_state["memory"] = None
if "api_key_valid" not in st.session_state:
    st.session_state["api_key_valid"] = False
if "stored_api_key" not in st.session_state:
    st.session_state["stored_api_key"] = ""

# 사이드바에서 설정한 변수 가져오기
api_key_valid = st.session_state.get("api_key_valid", False)
stored_api_key = st.session_state.get("stored_api_key", "")

# 콘텐츠 정보 가져오기
content_source = st.session_state.get("content_source", "📁 Upload File")
file = st.session_state.get("uploaded_file", None)
use_web_content = (content_source == "🌐 Load Novel (Nineteen Eighty-Four)")

if not api_key_valid:
    st.warning("⚠️ Please enter and test your OpenAI API Key in the sidebar.")  # API Key 경고
elif not file and not use_web_content:
    st.info("📚 Please choose a content source in the sidebar.")  # 콘텐츠 선택 안내
else:
    # 콘텐츠에 따라 키 설정
    content_key = "web_novel" if use_web_content else file.name
    
    # 콘텐츠 변경시 메모리 초기화
    if "current_content" not in st.session_state or st.session_state["current_content"] != content_key:
        st.session_state["memory"] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        st.session_state["current_content"] = content_key
        st.session_state["messages"] = []  # 새로운 콘텐츠 -> 채팅 기록 초기화
    
    # LLM 초기화
    llm = create_llm(stored_api_key)
    
    try:
        # Retriever 선택
        if use_web_content:
            retriever = embed_web_content(stored_api_key)
            content_name = "Nineteen Eighty-Four"
        else:
            retriever = embed_file(file, stored_api_key)
            content_name = file.name
        
        # 초기 메시지
        if len(st.session_state["messages"]) == 0:
            welcome_msg = f"Ready to chat about **{content_name}**! Ask me anything! 🚀"
            send_message(welcome_msg, "ai", save=False)
        
        paint_history()
        message = st.chat_input(f"Ask anything about {content_name}...")
        
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "chat_history": RunnableLambda(load_memory),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your API Key and try again.")