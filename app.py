from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import google.generativeai as genai
import os
import sys
import uuid
from dotenv import load_dotenv
import logging
from session_rag import ChatRAG
from langchain_ollama import OllamaEmbeddings

# Path to the rag-tutorial-v2 chroma database
rag_tutorial_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag-tutorial-v2")

# Import Chroma for document retrieval
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

app = Flask(__name__)
rag = ChatRAG()
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API (lazy init in case env loads later)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables at startup. Ensure it is set in .env")

# Path to the document RAG database
# After running migrate_db.py, you can change this to use LOCAL_DOCUMENT_PATH instead
DOCUMENT_CHROMA_PATH = os.path.join(rag_tutorial_path, "chroma")
LOCAL_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "document_store")

# Use local path if it exists, otherwise fall back to rag-tutorial-v2 path
if os.path.exists(LOCAL_DOCUMENT_PATH):
    DOCUMENT_CHROMA_PATH = LOCAL_DOCUMENT_PATH

def get_document_context(query_text, max_results=5):
    """Get relevant document context based on query."""
    try:
        # Log which database path we're using
        logger.info(f"Using document database at: {DOCUMENT_CHROMA_PATH}")
        
        # Prepare the document DB with Ollama embeddings (same as rag-tutorial-v2)
        embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
        doc_db = Chroma(persist_directory=DOCUMENT_CHROMA_PATH, embedding_function=embedding_function)
        
        # Search the DB
        logger.info(f"Searching for documents related to: '{query_text}'")
        results = doc_db.similarity_search_with_score(query_text, k=max_results)
        logger.info(f"Found {len(results)} relevant documents")
        
        # Format the context
        if not results:
            logger.warning("No document results found for query")
            return ""
            
        context_parts = []
        for doc, score in results:
            # Add document content with source info if available
            source = doc.metadata.get("id", "Unknown source")
            logger.info(f"Found relevant document: {source} with score: {score}")
            context_parts.append(f"Document: {source}\n{doc.page_content}")
        
        document_context = "\n\n---\n\n".join(context_parts)
        return document_context
    except Exception as e:
        logger.exception(f"Error retrieving document context: {e}")
        return ""

def get_model():
    """Lazily configure and return the Gemini model, with safe fallback."""
    global GEMINI_API_KEY
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please configure it in your environment or .env file.")

    # Configure SDK each call to be safe in reloads
    genai.configure(api_key=GEMINI_API_KEY)

    # Prefer a current model if available, fall back to gemini-pro for compatibility
    preferred_models = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro'
    ]
    last_error = None
    for m in preferred_models:
        try:
            return genai.GenerativeModel(m)
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to initialize model '{m}': {e}")
    # If all fail, raise the last error
    raise RuntimeError(f"Failed to initialize any Gemini model. Last error: {last_error}")

# Store conversation history (in production, use a database)
conversation_history = {}

@app.route('/')
def index():
    """Serve the main chatbot interface with session management"""
    # Get session ID from cookie or create a new one
    session_id = request.cookies.get('chat_session_id')
    if not session_id:
        session_id = rag.create_new_session()
    
    # Render template and set session cookie
    resp = make_response(render_template('index.html'))
    resp.set_cookie('chat_session_id', session_id, max_age=60*60*24*30)  # 30 day expiration
    return resp

@app.route('/api/ping')
def ping():
    """Quick connectivity test endpoint."""
    return jsonify({"ok": True, "message": "pong"})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and get responses from Gemini"""
    try:
        data = request.get_json(silent=True) or {}
        user_message = str(data.get('message', '')).strip()
        
        # Get session ID prioritizing request JSON payload over cookies
        # This ensures Postman and API clients can maintain sessions
        session_id = str(data.get('session_id', ''))
        if not session_id:
            session_id = request.cookies.get('chat_session_id')
            if not session_id:
                # Create new session if none exists
                session_id = rag.create_new_session()

        logger.info(f"/api/chat received - session_id={session_id}, bytes={len(user_message)}")
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
            
        # Store user message in RAG
        rag.add_message(session_id, user_message, "user")
        
        # Get conversation history context
        conversation_context = rag.get_conversation_context(session_id, user_message)
        
        # Get relevant document context
        document_context = get_document_context(user_message)
        
        # Create prompt with both contexts
        prompt = f"""You are a helpful AI assistant. Please respond based on the following information:
         
CONVERSATION HISTORY:
{conversation_context}

RELEVANT DOCUMENT INFORMATION:
{document_context}

Current User Message: {user_message}

Please provide a helpful, informative response based on both the conversation history and relevant document information.

Please provide a helpful, informative, and engaging response based on both the conversation history and relevant document information."""
        
        # Generate response using Gemini
        model = get_model()
        response = model.generate_content(prompt)
        bot_response = response.text
        
        # Store bot response in RAG
        rag.add_message(session_id, bot_response, "assistant")
        
        logger.info(f"Chat exchange - Session: {session_id}, User: {user_message[:50]}...")
        
        # Create response with session cookie
        resp = jsonify({
            'response': bot_response,
            'session_id': session_id
        })
        resp.set_cookie('chat_session_id', session_id, max_age=60*60*24*30)  # 30 day expiration
        return resp
        
    except Exception as e:
        logger.exception("Error in chat endpoint")
        return jsonify({'error': 'Sorry, I encountered an error processing your message.'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Create a new session and return its ID"""
    try:
        data = request.get_json(silent=True) or {}
        
        # Create a new session
        new_session_id = rag.create_new_session()
        
        # Return response with new session cookie
        resp = jsonify({
            'message': 'New conversation started',
            'session_id': new_session_id
        })
        resp.set_cookie('chat_session_id', new_session_id, max_age=60*60*24*30)
        return resp
        
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({'error': 'Failed to clear chat history'}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history using RAG"""
    # Get session ID prioritizing request parameters over cookies
    session_id = request.args.get('session_id', '')
    if not session_id:
        session_id = request.cookies.get('chat_session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
    
    query = request.args.get('query', '')
    
    context = rag.get_conversation_context(
        session_id=session_id,
        current_query=query,
        max_results=50
    )
    
    # Get session info
    session_info = rag.get_session_info(session_id)
    
    return jsonify({
        'history': context,
        'session_info': session_info
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Chatbot server is running'})

if __name__ == '__main__':
    # Run the development server
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)