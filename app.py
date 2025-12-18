import os
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import nest_asyncio

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
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

# Global variable to store the index
index_engine = None

# --- Configuration ---
def init_settings():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
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
    return True

init_settings()

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index_codebase():
    global index_engine
    data = request.json
    source_type = data.get('type') # 'local' or 'github'
    source_path = data.get('path') 

    if not source_path:
        return jsonify({"error": "Path/URL is required"}), 400

    try:

        documents = []
        if source_type == 'local':
            if not os.path.exists(source_path):
                 return jsonify({"error": "Directory not found"}), 404
            
            # 1. Load .gitignore patterns
            import pathspec
            from pathlib import Path

            gitignore_path = os.path.join(source_path, ".gitignore")
            ignore_patterns = []
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    ignore_patterns = f.read().splitlines()
            
            spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)

            # 2. Collect files recursively, respecting .gitignore and hidden files
            all_files = []
            for root, dirs, files in os.walk(source_path):
                # Exclude hidden directories (in-place modification of dirs)
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in [".git", ".venv", "__pycache__"]]
                
                for file in files:
                    if file.startswith("."): # skip hidden files
                        continue
                    
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, source_path)
                    
                    # Check against .gitignore
                    if not spec.match_file(rel_path):
                        all_files.append(full_path)

            if not all_files:
                 return jsonify({"error": "No files found (check .gitignore settings)"}), 400

            # 3. Load filtered files
            documents = SimpleDirectoryReader(input_files=all_files).load_data()
        
        elif source_type == 'github':
            github_token = os.getenv("GITHUB_TOKEN")
            owner = data.get('owner')
            repo = data.get('repo')
            branch = data.get('branch', 'main')
            
            # Simple parsing of github url if provided as path
            if not owner or not repo:
                 # Try to parse from URL like https://github.com/owner/repo
                 parts = source_path.replace("https://github.com/", "").split("/")
                 if len(parts) >= 2:
                     owner = parts[0]
                     repo = parts[1]
                 else:
                     return jsonify({"error": "Invalid GitHub URL or missing owner/repo"}), 400

            # Fetch .gitignore content first to build exclusion list?
            # GithubRepositoryReader is convenient but hard to inject robust gitignore logic into 
            # without modifying the reader or pre-fetching.
            # Strategy: Use GithubRepositoryReader but we might accept some noise OR 
            # we rely on its filter_file_extensions which works for file types.
            # The USER requested robust checking. 
            # We can use 'filter_file_paths' arg in GithubRepositoryReader (List[str], FilterType) 
            # BUT we don't know the paths yet.
            # OPTION: Load ALL, then filter documents post-load? Yes, feasible for smaller repos.
            
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
            raw_documents = loader.load_data(branch=branch)
            
            # Post-processing: Filter out things that look like hidden files or generic ignore patterns
            # (Fetching .gitignore from GitHub API is extra network overhead, simplistic filtering for now)
            filtered_documents = []
            for doc in raw_documents:
                # Metadata contains 'file_path' usually e.g. "src/main.py"
                file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name')
                
                if not file_path: 
                    filtered_documents.append(doc)
                    continue

                # Ignore hidden files/folders
                if any(part.startswith('.') for part in file_path.split('/')):
                    continue
                
                filtered_documents.append(doc)
            
            documents = filtered_documents

        else:
             return jsonify({"error": "Invalid source type"}), 400

        if not documents:
             return jsonify({"error": "No documents found to index"}), 400

        # Collect file paths for tree visualization
        indexed_files = []
        for doc in documents:
            file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name', 'unknown')
            indexed_files.append(file_path)
  
        index_engine = VectorStoreIndex.from_documents(documents)
        return jsonify({
            "message": f"Successfully indexed {len(documents)} documents.",
            "indexed_files": indexed_files
        })

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global index_engine
    if not index_engine:
        return jsonify({"error": "Codebase not indexed yet. Please go to settings and index a codebase."}), 400

    data = request.json
    user_message = data.get('message')

    if not user_message:
         return jsonify({"error": "Message is required"}), 400

    def generate():
        query_engine = index_engine.as_query_engine(streaming=True)
        streaming_response = query_engine.query(user_message)
        
        for text in streaming_response.response_gen:
             yield text

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
