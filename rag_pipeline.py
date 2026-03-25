from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def get_rag_chain(docs):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    full_text = "\n\n".join([doc.page_content for doc in docs[:5]])

    prompt = PromptTemplate.from_template("""
    Answer the question based on the context below.
    Give a detailed and helpful answer in simple English.
    
    Context: {context}
    Question: {question}
    Answer:
    """)

    chain = prompt | llm | StrOutputParser()

    def answer(question):
        return chain.invoke({"context": full_text, "question": question})

    return answer