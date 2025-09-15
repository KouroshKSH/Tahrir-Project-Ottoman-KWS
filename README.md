# Ottoman Keyword Search (KWS) Backend

A complete end-to-end system for keyword spotting in historical Ottoman documents using neural embeddings and vector similarity search. The system consists of three main components: vector database upload utilities, a backend API service, and a React frontend.

## Architecture Overview

```
┌─────────────────┐  1. Image Upload   ┌──────────────────┐
│   Frontend      │ ───────────────────>│     Backend      │
│ (Firebase Host) │                     │  (Cloud Run)     │
│                 │ <───────────────────│                  │
└─────────────────┘  4. Search Results  └──────────────────┘
                                                 │        │
                                        2. Model │        │ 3. Vector
                                        Request  │        │ Search
                                                 ▼        ▼
                                       ┌─────────────────┐ ┌─────────────────┐
                                       │   Amazon S3     │ │    Pinecone     │
                                       │ (PyTorch Model) │ │   Vector DB     │
                                       └─────────────────┘ └─────────────────┘

Data Flow:
1. Frontend sends keyword image to backend
2. Backend downloads model from S3, generates embeddings
3. Backend queries Pinecone with embeddings for similar vectors
4. Backend returns matching documents with coordinates to frontend
```

## Project Structure

```
ottoman-kws-backend/
├── vector-db-upload/          # Vector database population utilities
│   ├── create_collection.py   # Pinecone collection creation & upload
│   ├── preprocess.py          # Image preprocessing transforms
│   ├── makejit.py            # PyTorch model JIT compilation
│   ├── inspect_*.py          # Debugging utilities
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile           # Containerization
├── backend/                  # FastAPI backend service
│   ├── main.py              # Main API service (Google Cloud Functions)
│   └── requirements.txt     # Backend dependencies
└── frontend/                # React frontend application
    ├── src/
    │   ├── App.jsx         # Main application component
    │   └── main.jsx        # Entry point
    ├── package.json        # Node.js dependencies
    ├── vite.config.js     # Vite configuration
    └── firebase.json      # Firebase hosting config
```

## Components

### 1. Vector Database Upload (`vector-db-upload/`)

Processes pre-computed FAISS embeddings and associated metadata, uploading them to Pinecone for production use.

**Technical Implementation:**
- Reads FAISS index files containing document patch embeddings
- Loads metadata from pickle files with document coordinates and identifiers
- Converts embeddings to float32 format required by Pinecone
- Uploads in configurable batches (default 1000 vectors) with progress checkpointing
- Implements exponential backoff retry logic for failed uploads
- Handles problematic vectors (zero vectors, malformed metadata) gracefully

**Files:**
- `create_collection.py`: Main upload script with batch processing
- `makejit.py`: Converts PyTorch models to TorchScript format
- `preprocess.py`: Standard image preprocessing pipeline
- `inspect_embeddings.py` & `inspect_metadata.py`: Debugging utilities

**Usage:**
```bash
cd vector-db-upload
python create_collection.py --recreate --datatype float32
```

### 2. Backend API (`backend/`)

A serverless API service built with Google Cloud Functions Framework that performs real-time similarity search on uploaded images.

**Technical Implementation:**
- Downloads JIT-compiled PyTorch models from S3 on cold start with SHA256 verification
- Preprocesses uploaded images using ImageNet normalization (224x224 input)
- Generates embeddings using the loaded model and converts to float32
- Queries Pinecone index with generated embeddings, retrieving up to 10,000 candidates
- Applies distance-based filtering using configurable thresholds
- Returns structured results with document metadata and bounding box coordinates

**Environment Variables Required:**
```bash
# AWS S3 Model Storage
S3_BUCKET=your-model-bucket
S3_KEY=path/to/embedder.pth
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=ottoman-kws
PINECONE_ENVIRONMENT=us-east-1-aws

# Model Configuration
MODEL_SHA256=optional-checksum
MODEL_FORMAT=jit
INPUT_SIZE=224

# Performance Settings
TORCH_THREADS=1
MAX_CANDIDATES=10000
DEFAULT_THRESHOLD=0.5
```

**API Endpoints:**
- `POST /match` - Main search endpoint
- `GET /match?inspect=1` - Index statistics
- `GET /match?probe_random=1` - Random search probe
- `OPTIONS /match` - CORS preflight

### 3. Frontend (`frontend/`)

A React single-page application that provides the user interface for keyword search functionality.

**Technical Implementation:**
- Presents predefined keyword images (11 Ottoman terms) for search selection
- Implements adaptive threshold calculation based on distance distribution of results
- Makes POST requests to backend with multipart/form-data image uploads
- Processes returned match coordinates to overlay highlights on original documents
- Uses HTML5 Canvas API to render yellow bounding boxes over document images
- Groups results by document name and sorts by similarity distance

**Keywords Supported:**
- arak, bağ, boza, bozahane
- duhan, hamr, mahzen, meyhane
- şarap, şıra, üzüm

**Development:**
```bash
cd frontend
npm install
npm run dev
```

## Deployment Guide

### Prerequisites

1. **PyTorch Model**: JIT-compiled model compatible with PyTorch 2.2.x
2. **Amazon S3**: Bucket for model storage
3. **Pinecone**: Vector database account
4. **Google Cloud**: Account for Cloud Run deployment
5. **Firebase**: Account for frontend hosting

### Step 1: Prepare Vector Database

```bash
cd vector-db-upload

# Install dependencies
pip install -r requirements.txt

# Convert your model to JIT format
python makejit.py

# Upload vectors and metadata to Pinecone
python create_collection.py --recreate
```

### Step 2: Deploy Backend

```bash
# Set environment variables (see backend section above)
export PINECONE_API_KEY=your-key
export S3_BUCKET=your-bucket
# ... other variables

# Deploy to Google Cloud Run
gcloud run deploy ottoman-kws-backend \
  --source=./backend \
  --platform=managed \
  --region=europe-west4 \
  --allow-unauthenticated \
  --set-env-vars="PINECONE_API_KEY=${PINECONE_API_KEY},S3_BUCKET=${S3_BUCKET}"
```

### Step 3: Deploy Frontend

```bash
cd frontend

# Install dependencies
npm install

# Build for production
npm run build

# Deploy to Firebase
firebase deploy
```

## Model Requirements

The system requires a PyTorch model compatible with:
- **PyTorch Version**: 2.2.x+cpu
- **Format**: TorchScript (.pth file)
- **Input**: 224x224 RGB images
- **Output**: Normalized embeddings (float32)
- **Preprocessing**: ImageNet normalization

### Model Conversion Example

```python
# In makejit.py
model = Embedder(emb_dim=128).eval()
# ... load your trained weights ...
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, "embedder.pth")
```

## Configuration

### Vector Database Settings

- **Index Type**: Pinecone Serverless
- **Metric**: Cosine similarity
- **Dimension**: Matches model output (typically 128 or 512)
- **Batch Size**: 1000 vectors per upload batch

### Backend Performance

- **Concurrency**: Single-threaded PyTorch inference
- **Memory**: Optimized for serverless environments
- **Timeout**: 300s for complex searches
- **Max Candidates**: 10,000 vectors per search

### Frontend Configuration

- **Build Tool**: Vite
- **Framework**: React 19+
- **Hosting**: Firebase Hosting
- **API**: RESTful calls to Cloud Run backend

## Monitoring & Debugging

### Health Checks

```bash
# Backend health
curl https://your-backend-url/match?inspect=1

# Vector database stats
curl -X POST https://your-backend-url/match?probe_random=1
```

### Logging

- Backend logs are available in Google Cloud Console
- Frontend logs in browser developer tools
- Vector upload progress saved to checkpoints

### Common Issues

1. **Model Loading Errors**: Verify PyTorch version compatibility
2. **Vector Dimension Mismatch**: Ensure model output matches Pinecone index
3. **Memory Issues**: Reduce batch size in upload script
4. **CORS Errors**: Check `ALLOWED_ORIGINS` environment variable

## Development

### Local Development

```bash
# Backend (requires environment variables)
cd backend
python main.py

# Frontend
cd frontend
npm run dev
```

### Testing

```bash
# Test vector upload
cd vector-db-upload
python inspect_embeddings.py

# Test backend endpoints
curl -X POST localhost:8080/match?debug=1 -F "image=@test.jpg"
```