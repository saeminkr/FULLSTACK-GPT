import json
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import os

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


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


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def create_questions_prompt(difficulty, num_questions):
    # 난이도별 지시
    difficulty_instructions = {
        "Easy": "Make simple, straightforward questions focusing on basic facts and definitions.",
        "Medium": "Make moderate questions that require understanding of concepts and relationships.",
        "Hard": "Make challenging questions that require deep understanding, analysis, and critical thinking."
    }
    
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make {num_questions} questions to test the user's knowledge about the text.
    
    Difficulty Level: {difficulty}
    {difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])}
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {{context}}
""",
            )
        ]
    )

# 퀴즈 포멧팅
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    os.makedirs("./.cache/quiz_files", exist_ok=True)
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def run_quiz_chain(_docs, topic, difficulty, num_questions, api_key):
    # 함수 호출을 사용한 퀴즈 생성. 동영상과 같은 GPT모델지정
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        openai_api_key=api_key,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    questions_prompt = create_questions_prompt(difficulty, num_questions)
    questions_chain = {"context": format_docs} | questions_prompt | llm
    formatting_chain = formatting_prompt | llm
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


# 세션 상태 초기화
if "quiz_generated" not in st.session_state:
    st.session_state.quiz_generated = False
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "stored_api_key" not in st.session_state:
    st.session_state.stored_api_key = ""


with st.sidebar:
    st.markdown("## Settings")
    
    # API Key 입력
    st.markdown("### OpenAI API Key")
    api_key_valid = st.session_state.get("api_key_valid", False)
    stored_api_key = st.session_state.get("stored_api_key", "")
    
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
            st.session_state.quiz_generated = False
            st.session_state.quiz_data = None
            st.rerun()
    
    st.markdown("---")
    
    # 난이도 선택
    difficulty = st.selectbox(
        "Select Difficulty Level",
        ["Easy", "Medium", "Hard"],
        index=1,
        help="Choose the difficulty level of quiz questions"
    )
    
    # 문제 수 선택
    num_questions = st.selectbox(
        "Number of Questions",
        list(range(3, 11)),
        index=7,  # 기본값 10
        help="Select the number of questions (3-10)"
    )
    
    st.markdown("---")
    
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    
    st.markdown("---")
    
    # GitHub 링크
    st.markdown("### Links")
    st.markdown("[View Code on GitHub](https://github.com/saeminkr/FULLSTACK-GPT)")
    st.markdown("[View Streamlit](https://quizgpt-07.streamlit.app/)")


# 메인 콘텐츠
# api 키 없을 때 초기화면
if not st.session_state.api_key_valid:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Please enter your OpenAI API Key in the sidebar to get started.
    """
    )
# 파일 없을 때 초기화면
elif not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    # 퀴즈 생성
    if not st.session_state.quiz_generated:
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Making quiz..."):
                try:
                    response = run_quiz_chain(
                        docs, 
                        topic if topic else file.name,
                        difficulty,
                        num_questions,
                        st.session_state.stored_api_key
                    )
                    st.session_state.quiz_data = response
                    st.session_state.quiz_generated = True
                    st.session_state.quiz_submitted = False
                    st.session_state.score = 0
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")
    
    # 퀴즈 표시 및 채점
    if st.session_state.quiz_generated and st.session_state.quiz_data:
        if not st.session_state.quiz_submitted:
            # 퀴즈 폼
            with st.form("questions_form"):
                for i, question in enumerate(st.session_state.quiz_data["questions"]):
                    st.write(f"**Question {i+1}:** {question['question']}")
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                        key=f"q_{i}"
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                    elif value is not None:
                        st.error("Wrong!")
                button = st.form_submit_button("Submit Quiz")
                
                if button:
                    # 점수 계산
                    score = 0
                    for i, question in enumerate(st.session_state.quiz_data["questions"]):
                        selected = st.session_state.get(f"q_{i}")
                        if {"answer": selected, "correct": True} in question["answers"]:
                            score += 1
                    
                    st.session_state.score = score
                    st.session_state.quiz_submitted = True
                    st.rerun()
        
        else:
            # 결과 표시
            total = len(st.session_state.quiz_data["questions"])
            score = st.session_state.score
            
            st.markdown(f"## Quiz Results")
            st.markdown(f"### Score: {score}/{total}")
            
            # 각 문제별 정답/오답 표시
            for i, question in enumerate(st.session_state.quiz_data["questions"]):
                st.markdown(f"**Question {i+1}:** {question['question']}")
                
                user_answer = st.session_state.get(f"q_{i}")
                correct_answer = None
                
                # 정답 찾기
                for answer in question["answers"]:
                    if answer["correct"]:
                        correct_answer = answer["answer"]
                        break
                
                # 결과 표시
                if user_answer == correct_answer:
                    st.success("Correct!")
                    st.markdown(f"✅ {correct_answer}")
                elif user_answer is None:
                    st.warning("Not answered")
                    st.markdown(f"✅ {correct_answer}")
                else:
                    st.error("Incorrect choice")
                    st.markdown(f"❌ {user_answer}")
                    st.markdown(f"✅ {correct_answer}")
                
                st.markdown("")
            
            st.markdown("---")
            
            percentage = (score / total) * 100
            if score == total:
                st.success("🎉 Perfect Score! Congratulations!")
                st.balloons()
            else:
                st.info(f"You got {percentage:.1f}% correct!")
                
                # 만점이 아닌 경우만 재시험 버튼
                if st.button("Retake Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.score = 0
                    # 폼 답변 초기화
                    for i in range(len(st.session_state.quiz_data["questions"])):
                        if f"q_{i}" in st.session_state:
                            del st.session_state[f"q_{i}"]
                    st.rerun()
            
            # 새 퀴즈 생성
            if st.button("Go to New Quiz"):
                st.session_state.quiz_generated = False
                st.session_state.quiz_data = None
                st.session_state.quiz_submitted = False
                st.session_state.score = 0
                # 폼 답변 초기화
                for key in list(st.session_state.keys()):
                    if key.startswith("q_"):
                        del st.session_state[key]
                st.rerun()