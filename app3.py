import streamlit as st 
from transformers import AutoTokenizer ,AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma 
# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
#-----------------------------------------------------------------------------------------------
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message
#-----------------------------------------------------------------------------------------------
#model and tokenizer loading
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
offload="D:/2/Search-Your-PDF-App-main - 2/offload"
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', offload_folder="offload", torch_dtype=torch.float32)
#-----------------------------------------------------------------------------------------------
persist_directory = "db"
#-----------------------------------------------------------------------------------------------
def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db=None 
    #-----------------------------------------------------------------------------------
# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    # db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    # metadata = generated_text['metadata']
    # for text in generated_text:
        
    #     print(answer)

    # wrapped_text = textwrap.fill(response, 100)
    # return wrapped_text
    return answer,generated_text
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def main2():
    st.title("Search Your PDF 🐦📄")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            """
        )
    question = st.text_area("Enter your Question")
    if st.button("Ask"):
        st.info("Your Question: " + question)

        st.info("Your Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)

def main():  
    st.markdown("<h1 style='text-align: center; color: blue;'>จัดการข้อมูลสมรรถนะ 🦜📄 </h1>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>นำเข้าข้อมูล pdf 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])
    print (f"file path:{uploaded_file}")
    
    if uploaded_file is not None:
        #-----------------------------------------------------------------------------------
        file_details = {
            "ชื่อไฟล์": uploaded_file.name,
            "ขนาดไฟล์": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
        #-----------------------------------------------------------------------------------
        col1, col2= st.columns([1,2])
        with col1:
            st.markdown("<h4 style color:black;'>รายละเอียดไฟล์</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>การแสดงตัวอย่างไฟล์</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            answer="" 
            metadata=""
            with st.spinner('อยู่ระหว่างการฝังข้อมูล...'):
                ingested_data = data_ingestion()
            st.success('สร้างการฝังข้อมูลสำเร็จแล้ว!')
            st.markdown("<h4 style color:black;'>เริ่มพูดคุยได้ที่นี่</h4>", unsafe_allow_html=True)
        #-----------------------------------------------------------------------------------
            question = st.text_input("", key="input")
            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]
        #-----------------------------------------------------------------------------------                
            # Search the database for a response based on user input and update session state
            if question:
                answer, metadata = process_answer(question)
                # answer = process_answer({'query': question})
                st.session_state["past"].append(question)
                st.session_state["generated"].append(answer)
        #-----------------------------------------------------------------------------------
            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)
                # st.write(answer)
                st.write(metadata)

# python -m streamlit run app.py
if __name__ == '__main__':
    main()

