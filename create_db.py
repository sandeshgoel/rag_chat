from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

db_dir = 'chromadb'

urls = [
    "https://www.cnbc.com/guide/personal-finance-101-the-complete-guide-to-managing-your-money/#create-a-budget",
    "https://www.nerdwallet.com/article/investing/what-is-a-financial-plan",
]

pdfs = [
    "pdfs/personal-finance-guide.pdf",
]

docs_list = []

if len(pdfs) > 0:
    for file in pdfs:
        loader = PyPDFLoader(file)
        docs_list_pdf = loader.load()
        print(f'PDF docs_list length for file {file}: {len(docs_list_pdf)}')
        docs_list += docs_list_pdf

if len(urls) > 0:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list_url = [item for sublist in docs for item in sublist]
    print(f'URL docs_list length: {len(docs_list_url)}')
    docs_list += docs_list_url

print(f'TOTAL docs_list length: {len(docs_list)}')

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
print(f'doc_splits length: {len(doc_splits)}')

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    persist_directory=db_dir
)
#vectorstore.persist()

retriever = vectorstore.as_retriever()

print('retriever generated')

question = "financial plan"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content

print(f'\n{question}\n')
print(doc_txt)
