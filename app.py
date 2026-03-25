import streamlit as st
import tempfile
import os
from vector_store import create_vector_store
from rag_pipeline import get_rag_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="")
st.title(" RAG Chatbot — PDF se baat karo!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.header(" PDF Upload Karo")
    uploaded_file = st.file_uploader("PDF choose karo", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("PDF process ho raha hai..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name

            vs = create_vector_store(tmp_path)
            st.session_state.chain = get_rag_chain(vs)
            os.unlink(tmp_path)
            st.success(" PDF ready! Ab question pucho.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("PDF ke baare mein kuch pucho..."):
    if not st.session_state.chain:
        st.warning(" Pehle PDF upload karo!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
             answer = st.session_state.chain(question)
        st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})