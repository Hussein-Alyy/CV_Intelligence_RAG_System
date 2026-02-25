import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()


# ==============================
# 1. Processing Logic
# ==============================

def get_docs_with_metadata(folder_path="CVS"):
    # كل CV بيتقرأ لوحده وكل chunk بتاخد metadata باسم الـ CV
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if len(pdf_files) != 5:
        return None, len(pdf_files)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_docs = []
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        pdf_reader = PdfReader(filepath)

        # استخراج نص الـ CV كامل
        cv_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                cv_text += extracted

        # تقسيم الـ CV لـ chunks
        chunks = text_splitter.split_text(cv_text)

        # كل chunk بتاخد اسم الـ CV كـ metadata
        candidate_name = filename.replace(".pdf", "")
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": candidate_name}
            )
            all_docs.append(doc)

    return all_docs, len(pdf_files)


def get_vector_store(docs):
    # تحويل الـ chunks لـ embeddings وحفظها في FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local("faiss_cv_index")
    return vector_store


def get_all_candidate_context(vector_store, user_question, folder_path="CVS"):
    """
    - For comparison questions: fetch chunks from each CV separately so no candidate is missed
    - For questions about a specific person: use regular similarity search
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    candidate_names = [f.replace(".pdf", "") for f in pdf_files]

    # كشف لو السؤال مقارنة أو عن كل الـ candidates
    comparison_keywords = ["highest", "most", "best", "compare", "all", "who has", "which candidate", "rank", "top"]
    is_comparison = any(kw in user_question.lower() for kw in comparison_keywords)

    if is_comparison:
        # جيب أحسن chunks من كل CV على حدة
        all_docs = []
        for candidate in candidate_names:
            candidate_docs = vector_store.similarity_search(
                user_question,
                k=4,
                filter={"source": candidate}
            )
            all_docs.extend(candidate_docs)
        return all_docs
    else:
        # similarity search عادي
        return vector_store.similarity_search(user_question, k=8)


def ask_question(user_question, folder_path="CVS"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(
        "faiss_cv_index", embeddings, allow_dangerous_deserialization=True
    )

    docs = get_all_candidate_context(vector_store, user_question, folder_path)

    context_text = ""
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        context_text += f"\n[CV: {source}]\n{doc.page_content}\n"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    template = """You are a world-class HR expert and talent analyst with deep expertise in evaluating candidates.

Your job is to provide EXCEPTIONAL, accurate, and insightful answers.

Critical Instructions:
- Every chunk is labeled with [CV: candidate_name] — use this to correctly attribute info to the right person.
- For project-counting questions: go through EVERY CV in the context, list ALL projects per candidate, then compare.
- Never skip a candidate — if their CV is in the context, include them.
- Be specific: mention full names, project titles, technologies, dates.
- For comparison questions: structure your answer candidate by candidate, then give a final conclusion.
- Infer skills from projects and experience, not just the skills section.

Context:
{context}

Question:
{question}

Expert Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # بنبني الـ chain يدوياً عشان نبعت الـ context اللي عملناه
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(user_question)
    return response, docs


# ==============================
# 2. Streamlit UI
# ==============================

st.set_page_config(page_title="CV RAG Challenge", layout="wide")
st.title("🧠 CV Analysis System")
st.markdown("---")

# تهيئة الـ Session State
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("⚙️ Control Panel")
    folder_path = "CVS"

    if os.path.exists(folder_path):
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        st.markdown(f"**📁 CVs found in folder:** `{len(pdf_files)}`")
        for f in pdf_files:
            st.markdown(f"- 📄 {f}")
    else:
        st.error("❌ Folder 'CVS' not found!")
        pdf_files = []

    st.markdown("---")

    if st.button("🚀 Process CVs"):
        # هنا عشان نتاكد من عدد ال cv
        if not os.path.exists(folder_path):
            st.error("❌ Folder 'CVS' does not exist!")
        elif len(pdf_files) != 5:
            st.error(f"❌ You must have exactly 5 CVs. Found: {len(pdf_files)}")
        else:
            with st.spinner("⏳ Processing CVs... Please wait"):
                docs, count = get_docs_with_metadata(folder_path)
                if docs is None:
                    st.error(f"❌ Expected 5 CVs, found {count}.")
                else:
                    get_vector_store(docs)
                    st.session_state.processed = True
                    st.success(f"✅ Done! {len(docs)} chunks created. Ready to chat!")

    if st.session_state.processed:
        st.info("✅ CVs are processed and ready!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ==============================
# 3. Chat Section
# ==============================

st.subheader("💬 Ask about the Candidates")

# عشان الشات يفضل محفوظ
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        with st.expander("🔍 Source Chunks"):
            for j, doc in enumerate(chat["chunks"]):
                source = doc.metadata.get("source", "Unknown")
                st.info(f"**Chunk {j+1} | 📄 {source}:**\n\n{doc.page_content}")

# Input السؤال الجديد
user_question = st.chat_input("Ask a question about the candidates...")

if user_question:
    if not os.path.exists("faiss_cv_index"):
        st.warning("⚠️ Please process the CVs first using the sidebar button.")
    else:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching through CVs..."):
                answer, docs = ask_question(user_question, folder_path="CVS")
            st.write(answer)
            with st.expander("🔍 Source Chunks"):
                for j, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown")
                    st.info(f"**Chunk {j+1} | 📄 {source}:**\n\n{doc.page_content}")

        # حفظ السؤال والإجابة في الـ history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer,
            "chunks": docs
        })

                ##  streamlit run Application.py