import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# st.image("./Data/Capture.PNG")

#This function will go through pdf and extract and return list of page texts.
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]

# centered page layout
st.set_page_config(layout="centered", page_title="Cooee - Document QA")
st.header("Cooee - Public Extension")
st.write("---")

#file uploader
# uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
# st.write("---")
# if uploaded_files  is None:
#     st.info(f"""Upload files to analyse""")

# #Use vectorDB to QnA
# elif uploaded_files:
# st.write(str(len(uploaded_files)) + " document(s) loaded..")
#get text from documents
# textify_output = read_and_textify(uploaded_files)
# documents = textify_output[0]
# sources = textify_output[1]
#manual chunking based on pages.
# docs = documents

persist_directory = './Vector_store/'

#extract embeddings
embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
#vstore with metadata. Here we will store page numbers.

vStore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
st.write(vStore.get())
#     vStore = Chroma.from_texts(docs, embeddings, metadatas=[{"source": s} for s in sources], persist_directory=persist_directory)
#deciding model
model_name = "gpt-3.5-turbo"
# # model_name = "gpt-4"

# retriever = vStore.as_retriever()
# retriever.search_kwargs = {'k':2}

# #initiate model
llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
# model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
model = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vStore)

st.header("Ask your data")
user_q = st.text_area("Enter your question here")
if st.button("Get Response"):
        try:
            # create gpt prompt
            # result = model.run(user_q)
            with st.spinner("Cooee is working on it..."):
                result = model({"query": user_q}, return_only_outputs=True)
                st.subheader('Your response:')
                st.write(result['answer'])
                st.subheader('Source pages:')
                st.write(result['sources'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
