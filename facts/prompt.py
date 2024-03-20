from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriver
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

retriever = RedundantFilterRetriver(
    embeddings=embeddings,
    chroma=db
)

# retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)

result = chain.run("what is an interesting fact about english?")

print(result)
