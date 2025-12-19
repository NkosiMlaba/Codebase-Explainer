import os
import re
import json
import logging
import sqlite3
import hashlib
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, g
from dotenv import load_dotenv
import nest_asyncio
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader, GithubClient

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Rate Limiting ===
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# === Database Setup for Logging ===
DB_PATH = "logs.db"
INDEX_STORAGE_DIR = "index_storage"

def init_db():
    """Initialize SQLite database for logging."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            category TEXT,
            message TEXT,
            details TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS index_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            source_path TEXT,
            hash TEXT UNIQUE,
            indexed_files TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_to_db(level, category, message, details=None):
    """Log an event to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO logs (timestamp, level, category, message, details) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), level, category, message, json.dumps(details) if details else None)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log to DB: {e}")

def get_index_hash(source_type, source_path):
    """Generate a unique hash for an index based on source."""
    key = f"{source_type}:{source_path}"
    return hashlib.md5(key.encode()).hexdigest()

# === Global State ===
index_engine = None
indexing_status = {
    "is_indexing": False,
    "progress": 0,
    "message": "",
    "error": None
}
indexing_lock = threading.Lock()

# --- Configuration ---
def init_settings():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
        log_to_db("ERROR", "config", "OPENROUTER_API_KEY not found")
        return False

    llm = OpenRouter(
        api_key=api_key,
        model="meta-llama/llama-3.3-70b-instruct:free",
        max_tokens=512,
        temperature=0.1,
        system_prompt=(
            "You are an expert technical assistant that answers ONLY using the provided codebase context. "
            "Never hallucinate. If the answer is not in the context, say so. "
            "Provide short, clear, factual responses with code snippets where relevant."
        ),
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    log_to_db("INFO", "config", "Settings initialized successfully")
    return True

init_settings()

# === Input Validation ===
def validate_local_path(path):
    """Validate a local directory path."""
    if not path:
        return False, "Path is required"
    if not os.path.isabs(path):
        return False, "Path must be absolute (e.g., /home/user/project)"
    if not os.path.exists(path):
        return False, f"Directory not found: {path}"
    if not os.path.isdir(path):
        return False, "Path must be a directory, not a file"
    return True, None

def validate_github_url(url):
    """Validate a GitHub repository URL."""
    if not url:
        return False, "GitHub URL is required"
    pattern = r'^https?://github\.com/[\w.-]+/[\w.-]+/?$'
    if not re.match(pattern, url):
        return False, "Invalid GitHub URL format. Use: https://github.com/owner/repo"
    return True, None

# === Index Persistence ===
def get_index_storage_path(index_hash):
    """Get the storage path for an index."""
    return os.path.join(INDEX_STORAGE_DIR, index_hash)

def save_index(index, index_hash, source_type, source_path, indexed_files):
    """Persist index to disk."""
    try:
        storage_path = get_index_storage_path(index_hash)
        os.makedirs(storage_path, exist_ok=True)
        index.storage_context.persist(persist_dir=storage_path)
        
        # Save metadata to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO index_metadata (source_type, source_path, hash, indexed_files, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (source_type, source_path, index_hash, json.dumps(indexed_files), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        log_to_db("INFO", "index", f"Index saved to disk: {index_hash}")
        return True
    except Exception as e:
        log_to_db("ERROR", "index", f"Failed to save index: {e}")
        return False

def load_cached_index(index_hash):
    """Load a cached index from disk if available."""
    storage_path = get_index_storage_path(index_hash)
    if os.path.exists(storage_path):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            index = load_index_from_storage(storage_context)
            log_to_db("INFO", "index", f"Index loaded from cache: {index_hash}")
            return index
        except Exception as e:
            log_to_db("ERROR", "index", f"Failed to load cached index: {e}")
    return None

def get_cached_metadata(index_hash):
    """Get metadata for a cached index."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT indexed_files, created_at FROM index_metadata WHERE hash = ?", (index_hash,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"indexed_files": json.loads(row[0]), "created_at": row[1]}
    except Exception:
        pass
    return None

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index_codebase():
    global index_engine, indexing_status
    data = request.json
    source_type = data.get('type')
    source_path = data.get('path')
    force_reindex = data.get('force', False)

    # Input validation
    if source_type == 'local':
        valid, error = validate_local_path(source_path)
        if not valid:
            log_to_db("WARNING", "validation", error, {"path": source_path})
            return jsonify({"error": error}), 400
    elif source_type == 'github':
        valid, error = validate_github_url(source_path)
        if not valid:
            log_to_db("WARNING", "validation", error, {"url": source_path})
            return jsonify({"error": error}), 400
    else:
        return jsonify({"error": "Invalid source type. Use 'local' or 'github'."}), 400

    # Check for cached index
    index_hash = get_index_hash(source_type, source_path)
    if not force_reindex:
        cached_index = load_cached_index(index_hash)
        if cached_index:
            index_engine = cached_index
            metadata = get_cached_metadata(index_hash)
            return jsonify({
                "message": f"Loaded cached index (created: {metadata['created_at'][:10] if metadata else 'unknown'})",
                "indexed_files": metadata['indexed_files'] if metadata else [],
                "cached": True
            })

    log_to_db("INFO", "index", f"Starting indexing: {source_type} - {source_path}")

    try:
        documents = []
        if source_type == 'local':
            import pathspec
            from pathlib import Path

            gitignore_path = os.path.join(source_path, ".gitignore")
            ignore_patterns = []
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    ignore_patterns = f.read().splitlines()
            
            spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)

            all_files = []
            for root, dirs, files in os.walk(source_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in [".git", ".venv", "__pycache__"]]
                
                for file in files:
                    if file.startswith("."):
                        continue
                    
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, source_path)
                    
                    if not spec.match_file(rel_path):
                        all_files.append(full_path)

            if not all_files:
                return jsonify({"error": "No files found (check .gitignore settings)"}), 400

            documents = SimpleDirectoryReader(input_files=all_files).load_data()
        
        elif source_type == 'github':
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                error_msg = "GITHUB_TOKEN not configured. Add it to your .env file to index GitHub repos."
                log_to_db("ERROR", "config", error_msg)
                return jsonify({"error": error_msg}), 400

            parts = source_path.replace("https://github.com/", "").rstrip("/").split("/")
            owner, repo = parts[0], parts[1]

            github_client = GithubClient(github_token=github_token)
            loader = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo,
                use_parser=False,
                verbose=False,
                filter_file_extensions=(
                    [".py", ".js", ".ts", ".md", ".html", ".css", ".cpp", ".c", ".h", ".txt", ".json", ".ipynb", ".cs", ".tsx"],
                    GithubRepositoryReader.FilterType.INCLUDE
                ),
            )
            raw_documents = loader.load_data(branch="main")
            
            filtered_documents = []
            for doc in raw_documents:
                file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name')
                if not file_path: 
                    filtered_documents.append(doc)
                    continue
                if any(part.startswith('.') for part in file_path.split('/')):
                    continue
                filtered_documents.append(doc)
            
            documents = filtered_documents

        if not documents:
            return jsonify({"error": "No documents found to index"}), 400

        indexed_files = []
        for doc in documents:
            file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name', 'unknown')
            indexed_files.append(file_path)
  
        index_engine = VectorStoreIndex.from_documents(documents)
        
        # Save to disk
        save_index(index_engine, index_hash, source_type, source_path, indexed_files)
        
        log_to_db("INFO", "index", f"Indexing complete: {len(documents)} documents", {"files": indexed_files[:10]})
        
        return jsonify({
            "message": f"Successfully indexed {len(documents)} documents.",
            "indexed_files": indexed_files,
            "cached": False
        })

    except Exception as e:
        error_msg = str(e)
        # Improve error messages
        if "rate limit" in error_msg.lower():
            error_msg = "GitHub API rate limit exceeded. Try again later or add a GITHUB_TOKEN."
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            error_msg = "Authentication failed. Check your API keys in .env file."
        elif "404" in error_msg:
            error_msg = "Repository not found. Check the URL and ensure it's public (or you have access)."
        
        log_to_db("ERROR", "index", f"Indexing error: {error_msg}")
        logger.error(f"Indexing error: {e}")
        return jsonify({"error": error_msg}), 500

@app.route('/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat():
    global index_engine
    if not index_engine:
        return jsonify({"error": "Codebase not indexed yet. Please index a codebase first."}), 400

    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    if len(user_message) > 2000:
        return jsonify({"error": "Message too long. Maximum 2000 characters."}), 400

    log_to_db("INFO", "chat", f"Chat query: {user_message[:100]}...")

    def generate():
        try:
            query_engine = index_engine.as_query_engine(streaming=True)
            streaming_response = query_engine.query(user_message)
            
            for text in streaming_response.response_gen:
                yield text
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                yield "Error: API rate limit reached. Please wait a moment and try again."
            else:
                yield f"Error: {error_msg}"
            log_to_db("ERROR", "chat", f"Chat error: {error_msg}")

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# === Admin / Logging Dashboard ===
@app.route('/admin/logs')
def admin_logs():
    """Return recent logs as JSON."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "timestamp": row[1],
                "level": row[2],
                "category": row[3],
                "message": row[4],
                "details": json.loads(row[5]) if row[5] else None
            })
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin')
def admin_dashboard():
    """Simple admin dashboard."""
    return render_template('admin.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    api_key = bool(os.getenv("OPENROUTER_API_KEY"))
    github_token = bool(os.getenv("GITHUB_TOKEN"))
    return jsonify({
        "status": "ok",
        "index_loaded": index_engine is not None,
        "openrouter_configured": api_key,
        "github_configured": github_token
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    log_to_db("WARNING", "rate_limit", "Rate limit exceeded", {"ip": get_remote_address()})
    return jsonify({"error": "Rate limit exceeded. Please slow down your requests."}), 429

if __name__ == '__main__':
    os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)
    is_render = os.environ.get("RENDER", "False").lower() == "true"
    
    if is_render:
        port = int(os.environ.get("PORT", 10000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(host='127.0.0.1', port=5000, debug=True)
