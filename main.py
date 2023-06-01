import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# centered page layout
st.set_page_config(layout="centered", page_title="Cooee - Document QA")
st.header("Cooee - Public Extension")
st.write("---")


embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
index_name = "external-facing-cooee"

index = Pinecone.from_existing_index(index_name, embeddings)

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs


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

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  
  answer =  chain.run(input_documents=similar_docs, question=query)
  # print('Answer >>>>>>>>>>')
  # print(answer)
  # print("Relevant Documents >>>>>>>>>>")
  sources = []
  for d in similar_docs:
    # print(d.metadata)
    sources.append(d.metadata['source'])
  return  {'Answer':answer,'Sources':sources}

st.header("Ask your data")
user_q = st.text_area("Enter your question here")
if st.button("Get Response"):
        try:
            # create gpt prompt
            # result = model.run(user_q)
            with st.spinner("Cooee is working on it..."):
                result = get_answer(user_q)
                st.subheader('Your response:')
                st.write(result['Answer'])
                st.subheader('Source pages:')
                st.write(result['Sources'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
