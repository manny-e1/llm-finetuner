import os, json
import threading
import queue
import hashlib
from datetime import datetime
import re
from flask import Flask, request, jsonify, Response
import requests
from src import db, Run, User, Tenant, DocumentSource, DocumentChunk, create_app
from src.config import Config
import runpod
from flask_cors import CORS
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

app = create_app()
CORS(app)

runpod.api_key = Config.RUNPOD_KEY

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')
MILVUS_URI = os.getenv('MILVUS_URI')
MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')
MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'tenant_vectors')
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1024'))
SHARED_DOCKER_APP_URL = os.getenv('SHARED_DOCKER_APP_URL')

ingest_queue = queue.Queue()

def init_milvus():
    if MILVUS_TOKEN:
        connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    else:
        connections.connect(uri=MILVUS_URI)
    if not utility.has_collection(MILVUS_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="version", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields)
        Collection(name=MILVUS_COLLECTION, schema=schema)
    collection = Collection(MILVUS_COLLECTION)
    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
    collection.load()
    return collection

milvus_collection = init_milvus()

def get_or_create_tenant(tenant_key: str):
    tenant = Tenant.query.filter_by(key=tenant_key).first()
    if tenant:
        return tenant
    tenant = Tenant(key=tenant_key, name=tenant_key)
    db.session.add(tenant)
    db.session.commit()
    return tenant

def embed_texts(texts):
    resp = requests.post(EMBEDDING_API_URL, json={"texts": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]

def get_model_base_url(model_id: str):
    if SHARED_DOCKER_APP_URL:
        return SHARED_DOCKER_APP_URL.rstrip("/")
    return f"https://{model_id}.proxy.runpod.net"

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def enqueue_ingest(payload):
    ingest_queue.put(payload)

def ingestion_worker():
    with app.app_context():
        while True:
            payload = ingest_queue.get()
            if payload is None:
                break
            tenant_id = payload["tenant_id"]
            source_id = payload["source_id"]
            source_type = payload["source_type"]
            content = payload["content"]
            doc_source = DocumentSource.query.filter_by(tenant_id=tenant_id, source_id=source_id).first()
            if not doc_source:
                doc_source = DocumentSource(tenant_id=tenant_id, source_id=source_id, source_type=source_type)
                db.session.add(doc_source)
                db.session.commit()
            chunks = chunk_text(content)
            embeddings = embed_texts(chunks)
            insert_records = []
            for idx, chunk in enumerate(chunks):
                content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                existing = DocumentChunk.query.filter_by(
                    tenant_id=tenant_id,
                    source_id=source_id,
                    chunk_id=idx,
                    content_hash=content_hash
                ).first()
                if existing:
                    continue
                prev = DocumentChunk.query.filter_by(
                    tenant_id=tenant_id,
                    source_id=source_id,
                    chunk_id=idx
                ).order_by(DocumentChunk.version.desc()).first()
                next_version = 1 if not prev else prev.version + 1
                db.session.add(DocumentChunk(
                    tenant_id=tenant_id,
                    source_id=source_id,
                    chunk_id=idx,
                    version=next_version,
                    content_hash=content_hash
                ))
                insert_records.append({
                    "tenant_id": str(tenant_id),
                    "source_id": source_id,
                    "chunk_id": idx,
                    "version": next_version,
                    "embedding": embeddings[idx],
                    "text": chunk
                })
            db.session.commit()
            if insert_records:
                milvus_collection.insert(insert_records)
            ingest_queue.task_done()

threading.Thread(target=ingestion_worker, daemon=True).start()

@app.route('/finetune', methods=['POST'])
def finetune_route():
    """
    Initiate a finetuning run for a user with additional parameters.
    """
    data = request.get_json()

    # Validate required parameters
    email = data.get('email')
    model_name = data.get('model_name')
    model_type = data.get('model_type')
    is_llm = data.get('is_llm', False)
    is_agent = data.get('is_agent', False)
    description = data.get('description')
    peft_r = data.get('peft_r', 16)
    peft_alpha = data.get('peft_alpha', 16)
    peft_dropout = data.get('peft_dropout', 0.0)

    if not email or not model_name or not model_type or not description:
        return jsonify({"error": "email, model_name, model_type, and description are required"}), 400

    # Retrieve user by email
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check if the user already has an active run
    if user.run_id is not None:
        active_run = Run.query.get(user.run_id)
        if active_run and active_run.status in ["pending", "running"]:
            return jsonify({"error": "User already has an active simulation"}), 400

    new_run = Run(
        user_id=user.id,
        tenant_id=user.tenant_id,
        status="pending",
        podcast_id="",
        model_name=model_name,
        model_type=model_type,
        is_llm=is_llm,
        is_agent=is_agent,
        description=description,
        peft_alpha=peft_alpha,
        peft_dropout=peft_dropout,
        peft_r=peft_r,
    )
    db.session.add(new_run)
    db.session.commit()

    run_id = new_run.id
    new_run.podcast_id = str(run_id)
    user.run_id = run_id
    db.session.commit()

    return jsonify({
        "message": "Finetuning initiated successfully.",
        "run_id": run_id,
        "podcast_id": new_run.podcast_id or "",
        "model_name": new_run.model_name,
        "model_type": new_run.model_type,
        "description": new_run.description,
        "status": new_run.status,
    }), 200

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.get_json()
    tenant_key = data.get("tenant_id")
    source_id = data.get("source_id")
    source_type = data.get("source_type")
    content = data.get("content")
    if not tenant_key or not source_id or not source_type or not content:
        return jsonify({"error": "tenant_id, source_id, source_type, content are required"}), 400
    tenant = get_or_create_tenant(tenant_key)
    enqueue_ingest({
        "tenant_id": tenant.id,
        "source_id": source_id,
        "source_type": source_type,
        "content": content
    })
    return jsonify({"message": "Ingestion queued"}), 202

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    tenant_key = data.get("tenant_id")
    query = data.get("query")
    if not tenant_key or not query:
        return jsonify({"error": "tenant_id and query are required"}), 400
    tenant = get_or_create_tenant(tenant_key)
    embedding = embed_texts([query])[0]
    milvus_collection.load()
    results = milvus_collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=int(data.get("top_k", 5)),
        expr=f'tenant_id == "{tenant.id}"',
        output_fields=["source_id", "chunk_id", "version", "text"]
    )
    return jsonify({"results": results}), 200

@app.route('/docker_app/run_model_llm', methods=['POST'])
def docker_app_run_model_llm():
    if not SHARED_DOCKER_APP_URL:
        return jsonify({"error": "SHARED_DOCKER_APP_URL is not configured"}), 400
    metadata = request.get_json()
    if not metadata:
        return jsonify({"error": "No metadata provided"}), 400
    model_id = metadata.get("model_id")
    if model_id:
        run = Run.query.filter_by(podcast_id=str(model_id)).first()
        if run:
            run.status = "running"
            db.session.commit()
    try:
        resp = requests.post(f"{SHARED_DOCKER_APP_URL.rstrip('/')}/run_model_llm", json=metadata)
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream run_model_llm failed: {str(e)}"}), 502

@app.route('/docker_app/run_model', methods=['POST'])
def docker_app_run_model():
    if not SHARED_DOCKER_APP_URL:
        return jsonify({"error": "SHARED_DOCKER_APP_URL is not configured"}), 400
    model_id = request.form.get("model_id")
    if model_id:
        run = Run.query.filter_by(podcast_id=str(model_id)).first()
        if run:
            run.status = "running"
            db.session.commit()
    files = {k: (v.filename, v.stream, v.mimetype) for k, v in request.files.items()}
    try:
        resp = requests.post(f"{SHARED_DOCKER_APP_URL.rstrip('/')}/run_model", data=request.form, files=files)
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream run_model failed: {str(e)}"}), 502

@app.route('/docker_app/current_logs', methods=['GET'])
def docker_app_current_logs():
    if not SHARED_DOCKER_APP_URL:
        return jsonify({"error": "SHARED_DOCKER_APP_URL is not configured"}), 400
    try:
        resp = requests.get(f"{SHARED_DOCKER_APP_URL.rstrip('/')}/current_logs")
        return Response(resp.text, status=resp.status_code, mimetype="text/plain")
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream current_logs failed: {str(e)}"}), 502

@app.route('/docker_app/logs', methods=['GET'])
def docker_app_logs():
    if not SHARED_DOCKER_APP_URL:
        return jsonify({"error": "SHARED_DOCKER_APP_URL is not configured"}), 400
    try:
        resp = requests.get(f"{SHARED_DOCKER_APP_URL.rstrip('/')}/logs")
        return Response(resp.text, status=resp.status_code, mimetype="text/plain")
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream logs failed: {str(e)}"}), 502

@app.route('/docker_app/clear_logs', methods=['POST'])
def docker_app_clear_logs():
    if not SHARED_DOCKER_APP_URL:
        return jsonify({"error": "SHARED_DOCKER_APP_URL is not configured"}), 400
    try:
        resp = requests.post(f"{SHARED_DOCKER_APP_URL.rstrip('/')}/clear_logs")
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream clear_logs failed: {str(e)}"}), 502

@app.route('/finished_finetuning', methods=['GET'])
def finished_finetuning():
    """
    Removes the run_id from the associated user and marks the run as completed or removed.
    Accepts podcast_id as a parameter.
    """
    params = request.args
    podcast_id = params.get('podcast_id')
    is_llm_str = request.args.get('is_llm', False)

    if is_llm_str is not None:
        if is_llm_str.lower() in ['true', '1', 'yes']:
            is_llm = True
        elif is_llm_str.lower() in ['false', '0', 'no']:
            is_llm = False
        else:
            return jsonify({"error": "Invalid value for is_llm. Use true or false."}), 400
    else:
        is_llm = False

    if not podcast_id:
        return jsonify({"error": "Podcast ID is required"}), 400

    # Find the run associated with the podcast_id
    run = Run.query.filter_by(podcast_id=podcast_id).first()
    if not run:
        return jsonify({"error": "Run with the provided podcast ID not found"}), 404

    try:
        run.status = "finished"  # or "removed" based on your logic
        run.is_llm = is_llm

        user = User.query.filter_by(run_id=run.id).first()
        if user:
            user.run_id = None
        db.session.commit()

        return jsonify({"message": "Finetuning finished successfully", "run_id": run.id}), 200

    except Exception as e:
        db.session.rollback()  # Rollback in case of errors
        return jsonify({"error": f"Failed to finish finetuning: {str(e)}"}), 500

@app.route('/delete', methods=['GET'])
def delete_run():
    """
    Deletes a run based on the provided podcast_id.
    If the associated pod does not exist, it still proceeds to delete the run instance.
    """
    podcast_id = request.args.get('podcast_id')
    if not podcast_id:
        return jsonify({"error": "Podcast ID is required"}), 400

    pod_id = podcast_id.replace("-5000", "")

    run = Run.query.filter_by(podcast_id=podcast_id).first()
    if not run:
        return jsonify({"error": "Run not found"}), 404

    if not SHARED_DOCKER_APP_URL:
        try:
            runpod.stop_pod(pod_id)
            runpod.terminate_pod(pod_id)
        except Exception as e:
            error_message = str(e)
            if "Attempted to stop pod that does not exist" in error_message:
                print(f"Pod {pod_id} does not exist, proceeding with run deletion...")
            else:
                return jsonify({"error": f"Failed to delete run: {error_message}"}), 500
    try:
        # Remove the run from the database
        db.session.delete(run)
        db.session.commit()
        return jsonify({"message": f"Run with podcast_id {podcast_id} deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route('/run_list', methods=['GET'])
def run_list():
    user_email = request.args.get('email')
    if not user_email:
        return jsonify({"error": "Email is required"}), 400

    try:
        # Find the user by email
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get the runs associated with the user
        user_runs = Run.query.filter_by(user_id=user.id).order_by(Run.id.desc()).all()

        # Format the runs as a list of dictionaries
        run_data = [
            {
                "run_id": run.id,
                "podcast_id": run.podcast_id,
                "status": run.status,
                "model_name": run.model_name,
                "model_type": run.model_type,
                "description": run.description,
                "is_llm": run.is_llm,
                "is_agent": run.is_agent,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            }
            for run in user_runs
        ]

        return jsonify(run_data), 200

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve run list: {str(e)}"}), 500


@app.route('/get_podcast', methods=['GET'])
def get_podcast():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user.run_id is None:
        return jsonify({
            "message": "User has no active simulation",
            "run_id": "",
            "status": "",
            "podcast_id": "",
        }), 200

    active_run = Run.query.get(user.run_id)
    if not active_run:
        return jsonify({
            "message": "User has no active simulation",
            "run_id": "",
            "status": "",
            "podcast_id": ""
        }), 200

    return jsonify({
        "message": "User has an active simulation",
        "run_id": user.run_id,
        "status": active_run.status,
        "podcast_id": active_run.podcast_id
    }), 200


@app.route("/oauth/callback")
def oauth_callback():
    code = request.args.get("code")
    if not code:
        return "Missing code", 400

    # Exchange authorization code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    r = requests.post(token_url, data=data)
    tokens = r.json()

    return f"Tokens: {tokens}"


@app.route('/api/login/google', methods=['POST'])
def google_login():
    data = request.get_json()
    token = data.get('credential')
    tenant_key = data.get('tenant_id', 'default')
    if not token:
        return jsonify({"error": "No credential token provided"}), 400
    
    try:
        # Verify the ID token using Google's libraries
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID  # Replace with your actual client ID
        )
        # Extract user info from the verified token
        user_email = idinfo.get('email')
        user_name = idinfo.get('name')
        user_picture = idinfo.get('picture')

        # Upsert user into your database
        tenant = get_or_create_tenant(tenant_key)
        existing_user = User.query.filter_by(email=user_email).first()
        if not existing_user:
            new_user = User(email=user_email, name=user_name, picture=user_picture, tenant_id=tenant.id)
            db.session.add(new_user)
            db.session.commit()
            user_id = new_user.id
        else:
            existing_user.name = user_name
            existing_user.picture = user_picture
            existing_user.tenant_id = tenant.id
            db.session.commit()
            user_id = existing_user.id

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user_id,
                "email": user_email,
                "name": user_name,
                "picture": user_picture
            }
        }), 200
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/inference/<model_id>', methods=['POST'])
def inference(model_id):
    model_endpoint = f"{get_model_base_url(model_id)}/inference"

    input_text = request.form.get("input")
    temperature = request.form.get("temperature", 0.5)  # Default: 0.0
    max_tokens = request.form.get("max_tokens", 500)    # Default: 500
    image = request.files.get("image")

    if not input_text or not image:
        return jsonify({"error": "Missing required parameters: input and/or image"}), 400

    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type 
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    # Prepare request payload
    files = {"image": (image.filename, image.stream, image.mimetype)}
    data = {
        "input": input_text,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type
    }
    try:
        response = requests.post(model_endpoint, files=files, data=data)
        return jsonify(response.json()), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


@app.route('/inference_b64/<model_id>', methods=['POST'])
def inference_b64(model_id):
    model_endpoint = f"{get_model_base_url(model_id)}/inference_b64"

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON or empty request body."}), 400

    input_text = data.get("input", "").strip()
    image_b64 = data.get("image", "")
    
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type 
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    temperature = float(data.get("temperature", 0.5))
    max_tokens = int(data.get("max_tokens", 500))

    if not input_text or not image_b64:
        return jsonify({"error": "Missing required parameters: input and/or image"}), 400

    payload = {
        "input": input_text,
        "image": image_b64,  # Forward base64 as a string
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type
    }

    try:
        response = requests.post(model_endpoint, json=payload, timeout=120)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to model API failed: {str(e)}"}), 500


@app.route('/inference-video/<model_id>', methods=['POST'])
def inference_video_front(model_id):
    model_endpoint = f"{get_model_base_url(model_id)}/inference-video"

    input_text = request.form.get("input")
    temperature = request.form.get("temperature", 0.5)
    max_tokens = request.form.get("max_tokens", 500)
    fps = request.form.get("fps", 1.0)

    if not input_text:
        return jsonify({"error": "Missing 'input' parameter"}), 400

    if 'video' not in request.files:
        return jsonify({"error": "Missing 'video' file in form data."}), 400
    
    video_file = request.files['video']

    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id or model not found."}), 404

    model_type = run.model_type
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    data = {
        "input": input_text,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type,
        "fps": fps
    }
    files = {
        "video": (video_file.filename, video_file.stream, video_file.mimetype)
    }

    try:
        response = requests.post(model_endpoint, data=data, files=files)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/inference-video/result/<model_id>', methods=['GET'])
def get_video_inference_logs_front(model_id):
    model_endpoint = f"{get_model_base_url(model_id)}/video_inference_logs"
    try:
        # Forward the GET request to the Docker app
        response = requests.get(model_endpoint)

        # If the response is successful, return its text content directly
        if response.status_code == 200:
            return Response(response.content, mimetype='text/plain')
        else:
            # Return any JSON error from the Docker app
            return jsonify(response.json()), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Log request failed: {str(e)}"}), 500


@app.route('/inference-llm/<model_id>', methods=['POST'])
def inference_llm(model_id):
    # Retrieve JSON payload from the request body.
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    input_text = data.get("input")
    temperature = data.get("temperature", 0.5)  # Default: 0.0
    max_tokens = data.get("max_tokens", 1000)    # Default: 500
    session_id = data.get("session_id", "default")

    if not input_text or not model_id:
        return jsonify({"error": "Missing required parameters: input and/or model_id"}), 400

    # Build the model endpoint using the provided model_id.
    model_endpoint = f"{get_model_base_url(model_id)}/inference-llm"

    # Look up the model's record.
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type 
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    # Prepare JSON payload for the inference request.
    payload = {
        "input": input_text,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type,
        "is_agent": run.is_agent,
        "session_id": session_id,
    }
    
    try:
        # Post JSON payload to the model endpoint.
        response = requests.post(model_endpoint, json=payload)
        return jsonify(response.json()), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


@app.route('/inference-llm/stream/<model_id>', methods=['POST'])
def inference_llm_stream_proxy(model_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    input_text = data.get("input")
    temperature = data.get("temperature", 0.5)
    max_tokens = data.get("max_tokens", 1000)
    session_id = data.get("session_id", "default")

    if not input_text or not model_id:
        return jsonify({"error": "Missing required parameters: input and/or model_id"}), 400

    container_stream_url = f"{get_model_base_url(model_id)}/inference-llm/stream"

    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    payload = {
        "input": input_text,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "model_type": model_type,
        "is_agent": run.is_agent,
        "session_id": session_id,
    }

    try:
        r = requests.post(container_stream_url, json=payload, stream=True)
        def plain_chunked_proxy():
            for raw_line in r.iter_lines(decode_unicode=True):
                if raw_line:
                    # 1) Remove SSE prefix like "data: "
                    line = raw_line.replace("data: ", "")

                    # 2) Collapse multiple spaces into one
                    line = re.sub(r'\s+', ' ', line)

                    # 3) Remove space before punctuation: e.g. "word  ," -> "word,"
                    line = re.sub(r'\s+([.,!?:;])', r'\1', line)

                    yield line

        return Response(plain_chunked_proxy(), mimetype='text/plain')

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to container streaming endpoint failed: {str(e)}"}), 500


@app.route("/create-session/<model_id>", methods=["POST"])
def create_session_proxy(model_id):
    url = f"{get_model_base_url(model_id)}/create-session"
    try:
        r = requests.post(url)
        return jsonify(r.json()), r.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream create-session failed: {e}"}), 502


@app.route('/update_status', methods=['GET'])
def update_status():
    model_id = request.args.get('model_id')
    status = request.args.get('status')
    is_llm_str = request.args.get('is_llm', False)

    if is_llm_str is not None:
        if is_llm_str.lower() in ['true', '1', 'yes']:
            is_llm = True
        elif is_llm_str.lower() in ['false', '0', 'no']:
            is_llm = False
        else:
            return jsonify({"error": "Invalid value for is_llm. Use true or false."}), 400
    else:
        is_llm = False

    if not model_id or not status:
        return jsonify({"error": "model_id and status are required"}), 400

    # Find the corresponding run
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Run not found for the given model_id"}), 404

    try:
        # Update run status
        if status.lower() not in ["finished", "failed"]:
            return jsonify({"error": "Invalid status. Must be 'finished' or 'failed'"}), 400

        run.status = status.lower()
        run.is_llm = is_llm

        # Find the user associated with this run
        user = User.query.filter_by(run_id=run.id).first()
        if user:
            user.run_id = None  # Nullify user's run_id since the process is completed

        # Commit changes to database
        db.session.commit()
        
        if status.lower() == "failed" and not SHARED_DOCKER_APP_URL:
            pod_id = model_id.replace("-5000", "")
            runpod.stop_pod(pod_id)
            runpod.terminate_pod(pod_id)

        return jsonify({
            "message": f"Run {run.id} status updated to '{status}' and user run_id nullified.",
            "run_id": run.id,
            "new_status": run.status
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to update status: {str(e)}"}), 500


@app.route('/get_retrain_info', methods=['GET'])
def get_retrain_info():
    """
    Returns retraining info (model name, model type, etc.) for the given model_id (podcast_id).
    Example usage: /get_retrain_info?model_id=12345
    """
    model_id = request.args.get('model_id')
    if not model_id:
        return jsonify({"error": "model_id query parameter is required"}), 400

    # Find the run in your database
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": f"No run found for model_id: {model_id}"}), 404

    # Return the relevant fields from the Run table
    return jsonify({
        "model_name": run.model_name or "",
        "model_type": run.model_type or "",
        "description": run.description or "",
        "peft_r": run.peft_r if run.peft_r is not None else 16,
        "peft_alpha": run.peft_alpha if run.peft_alpha is not None else 16,
        "peft_dropout": run.peft_dropout if run.peft_dropout is not None else 0.0
    }), 200

@app.route('/filter_livetalking_image', methods=['GET'])
def filter_livetalking_image():
    try:
        runpod.api_key = Config.RUNPOD_KEY  # or hardcode it if needed

        all_pods = runpod.get_pods()

        for pod in all_pods:
            if pod.get("imageName", "").lower() == "brianarfeto/livetalking:latest":
                return jsonify({
                    "result": pod["id"]
                }), 200

        return jsonify({"result": None}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
