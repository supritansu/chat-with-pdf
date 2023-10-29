import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

os.environ["OPENAI_API_KEY"] = ""

#sidebar contents

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM powered chatbot  using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model                
                
                
    ''')
    add_vertical_space(5)
    st.write("Made by [Suprit] (https://github.com/supritansu)")  


def main():
    st.header("Chat with PDF")

    load_dotenv()

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    

    if pdf is not None:
        
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)
        strin=""
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size of 1000 tokens
            chunk_overlap=200,  # overlap size of 200 tokens between consecutive chunks
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pk1", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        
        query = st.text_input("Ask questions about your PDF file:")
        st.write(query)
        
        with open("output.txt", "r") as file:
                strin = file.read()
        strin=strin+ " User: "+query+","
        print(strin)
        if query:
            
            docs = VectorStore.similarity_search(query=query)

            # Create an instance of OpenAI
            llm = OpenAI()

            # Specify the chain type without an extra field
            chain = load_qa_chain(llm=llm)

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=strin)
                strin=strin+ " AI: "+ response+","
                print(strin)
                
                print(cb)
            with open("output.txt", "w") as file:
                    file.write(strin)
            st.write(response)

if __name__ == '__main__':
    with open("output.txt", "w") as file:
        file.truncate(0)
    main()