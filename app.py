from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from prompts import qa_system_prompt, contextualize_q_system_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
import os

from flask import Flask, render_template, request, jsonify

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Chat history and conversation store
chat_history = []
conversation_store = {}
FAISS_PATH = "faiss"

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    temperature=0.3,
    max_tokens=2000,
    default_headers={
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Jio AI Assistant"
    }
)

app = Flask(__name__)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in conversation_store:
        print(f"Creating store for session: {session_id}")
        conversation_store[session_id] = ChatMessageHistory()
    return conversation_store[session_id]


def get_document_loader():
    """Load all PDF documents from the static folder"""
    loader = DirectoryLoader('static', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs


def get_text_chunks(documents):
    """Split documents into chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def get_embeddings():
    """Create or load FAISS embeddings"""
    path = os.path.join(os.getcwd(), FAISS_PATH)
    if os.path.exists(path):
        print(f"Index exists. Loading from {path}")
        db = FAISS.load_local(path, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", model_kwargs={'device':'cpu'}), allow_dangerous_deserialization=True)
    else:
        print(f"Index does not exist. Creating now...")
        documents = get_document_loader()
        chunks = get_text_chunks(documents)
        db = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", model_kwargs={'device':'cpu'}))
        print(f"Index created. Storing at {path}")
        db.save_local(path)
    return db


def get_retriever():
    """Get the retriever for querying documents"""
    db = get_embeddings()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever


def process_llm_response(chain, question):
    """Process the LLM response and extract source information"""
    llm_response = chain(question)
    
    result = llm_response['result']
    source_document = None
    page_number = None
    
    if 'source_documents' in llm_response and len(llm_response['source_documents']) > 0:
        source = llm_response['source_documents'][0]
        source_document = source.metadata.get('source', '')
        page_number = source.metadata.get('page', 0)
        
        # Remove 'static/' prefix from source path
        if source_document.startswith('static/'):
            source_document = source_document[7:]
        
        print(f"Source: {source_document}, Page: {page_number}")
    
    return result, source_document, page_number


def get_chain():
    """Create the RetrievalQA chain"""
    retriever = get_retriever()
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route - handles both GET (display page) and POST (process question)"""
    
    if request.method == 'GET':
        # Display the main page with empty chat history
        return render_template('index.html', chat_history=chat_history)
    
    # POST request - process the question
    question = request.form.get('question', '').strip()
    
    if not question:
        return render_template('index.html', chat_history=chat_history)
    
    try:
        # Get retriever and create conversational chain
        retriever = get_retriever()
        
        # Create contextualize prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Create QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational RAG chain with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Get response
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "abc123"}}  # Hardcoded session
        )
        
        # Add to chat history
        chat_history.append(question)
        chat_history.append(response['answer'])
        
        print(f"Question: {question}")
        print(f"Answer: {response['answer']}")
        
    except Exception as e:
        print(f"Error processing question: {e}")
        error_message = f"Sorry, I encountered an error: {str(e)}"
        chat_history.append(question)
        chat_history.append(error_message)
    
    return render_template('index.html', chat_history=chat_history)


@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    conversation_store.clear()
    return jsonify({"status": "success", "message": "Chat history cleared"})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "embeddings_loaded": os.path.exists(os.path.join(os.getcwd(), FAISS_PATH)),
        "chat_history_size": len(chat_history)
    })


if __name__ == "__main__":
    print("Starting Jio AI Assistant...")
    print("Loading embeddings...")
    
    try:
        get_embeddings()
        print("✅ Embeddings loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading embeddings: {e}")
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)  # Disable reloader