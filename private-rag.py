import gradio as gr
import lmstudio as lms
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from LMStudioEmbeddings import LMStudioEmbeddings  # Import the custom embeddings class
from langchain_community.chat_models import ChatOpenAI

# 1. Initialization (Only do this once) - Moved outside Gradio for persistence
print("Loading document")
document = "2025-USA-Pickleball-Rulebook.pdf"
loader = PyPDFLoader(document)
documents = loader.load()
print("Document loaded, splitting into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print("Document split into chunks, creating embeddings and vector store")
embedding_model_name = "text-embedding-nomic-embed-text-v1.5"
embedding_model = lms.embedding_model(embedding_model_name)
print(embedding_model)
embeddings_object = LMStudioEmbeddings(embedding_model)
db = Chroma.from_documents(chunks, embeddings_object, persist_directory="./chroma_db")
print("Getting llm and creating retriever and QA chain")
llm_model_name = "gemma-3-12B-it-qat"
retriever = db.as_retriever()
llm = ChatOpenAI(base_url = "http://localhost:1234/v1",
                 api_key = "lm-studio", # dummy key
                 model = llm_model_name)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# 3. Gradio Interface
def query_knowledge_base(query):
    """Handles the user's query and returns the agent's response."""
    try:
        result = qa_chain.invoke(query)
        return result['result']  # Return only the answer from the agent
    except Exception as e:
        return f"Error processing query: {e}"


iface = gr.Interface(
    fn=query_knowledge_base,
    inputs=gr.Textbox(lines=5, placeholder="Enter your question here..."),
    outputs=gr.Textbox(lines=10),
    title="Knowledge Base Query Interface",
    description="Ask questions about the document loaded into the knowledge base.",
)

iface.launch()
