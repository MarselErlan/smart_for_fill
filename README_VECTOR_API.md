# Vector Database API Documentation

## Overview

The Smart Form Fill API now includes vector database management endpoints that allow you to re-embed and update your resume and personal information vector databases via HTTP API calls.

## Features

- **Re-embed Resume**: Update the resume vector database from `docs/resume/ERIC_ABRAM33.docx`
- **Re-embed Personal Info**: Update the personal info vector database from `docs/info/personal_information.txt`
- **Batch Re-embedding**: Update both databases in a single API call
- **Database Status**: Check the current status of vector databases
- **Search Functionality**: Search through vector databases with similarity scoring

## Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### 📊 Status Endpoints

#### Get Resume Database Status

```http
GET /api/v1/resume/status
```

**Response:**

```json
{
  "database_type": "resume",
  "status": "ready",
  "last_updated": "2025-06-12T22:28:11.582561",
  "total_entries": 2,
  "latest_entry": {
    "timestamp": "20250612_222810",
    "total_chunks": 7,
    "embedding_dimension": 1536,
    "model": "text-embedding-3-small"
  }
}
```

#### Get Personal Info Database Status

```http
GET /api/v1/personal-info/status
```

### 🔄 Re-embedding Endpoints

#### Re-embed Resume

```http
POST /api/v1/resume/reembed
```

**Response:**

```json
{
  "status": "success",
  "message": "Resume vector database updated successfully",
  "timestamp": "20250612_222810",
  "processing_time": 1.23,
  "details": {
    "documents_loaded": 1,
    "chunks_created": 7,
    "embedding_dimension": 1536,
    "model": "text-embedding-3-small",
    "vectorstore_created": true
  }
}
```

#### Re-embed Personal Info

```http
POST /api/v1/personal-info/reembed
```

#### Batch Re-embed (Both Databases)

```http
POST /api/v1/reembed-all
```

**Response:**

```json
{
  "status": "success",
  "message": "Batch re-embedding completed in 2.00 seconds",
  "processing_time": 2.0,
  "results": {
    "resume": {
      "status": "success",
      "timestamp": "20250612_222810"
    },
    "personal_info": {
      "status": "success",
      "timestamp": "20250612_222811"
    }
  }
}
```

### 🔍 Search Endpoints

#### Search Resume

```http
POST /api/v1/resume/search?query=python%20experience&k=3
```

**Parameters:**

- `query` (string): Search query
- `k` (int): Number of results to return (default: 5)

**Response:**

```json
{
  "status": "success",
  "query": "python experience",
  "results": [
    {
      "content": "Programming Languages: Java, Python, JavaScript...",
      "metadata": {
        "source": "docs/resume/ERIC _ABRAM33.docx"
      },
      "similarity_score": 1.4906
    }
  ],
  "total_results": 1
}
```

#### Search Personal Info

```http
POST /api/v1/personal-info/search?query=work%20authorization&k=2
```

## Usage Examples

### Using cURL

```bash
# Check resume status
curl -X GET "http://localhost:8000/api/v1/resume/status"

# Re-embed personal info
curl -X POST "http://localhost:8000/api/v1/personal-info/reembed"

# Search resume
curl -X POST "http://localhost:8000/api/v1/resume/search?query=python&k=3"

# Batch re-embed both databases
curl -X POST "http://localhost:8000/api/v1/reembed-all"
```

### Using Python

```python
import requests

# Re-embed personal info
response = requests.post("http://localhost:8000/api/v1/personal-info/reembed")
result = response.json()
print(f"Status: {result['status']}")
print(f"Processing time: {result['processing_time']:.2f}s")

# Search resume
params = {"query": "python experience", "k": 3}
response = requests.post("http://localhost:8000/api/v1/resume/search", params=params)
results = response.json()

for i, result in enumerate(results["results"], 1):
    print(f"{i}. Score: {result['similarity_score']:.4f}")
    print(f"   Content: {result['content'][:100]}...")
```

### Using the Test Script

Run the comprehensive test script:

```bash
python test_vector_api.py
```

This script will:

- Test API connectivity
- Check database status
- Perform search tests
- Test re-embedding functionality
- Run batch operations

## File Structure

```
smart_for_fill/
├── docs/
│   ├── resume/
│   │   └── ERIC_ABRAM33.docx          # Source resume file
│   └── info/
│       └── personal_information.txt    # Source personal info file
├── resume/
│   └── vectordb/                       # Resume vector database
│       ├── embeddings_*.json
│       ├── embeddings_*.pkl
│       ├── faiss_store_*/
│       ├── metadata_*.json
│       └── index.json
├── info/
│   └── vectordb/                       # Personal info vector database
│       ├── personal_info_*.json
│       ├── personal_info_*.pkl
│       ├── faiss_store_*/
│       ├── metadata_*.json
│       └── index.json
├── main.py                             # FastAPI server
├── resume_extractor.py                 # Resume processing
├── personal_info_extractor.py          # Personal info processing
└── test_vector_api.py                  # API test script
```

## When to Re-embed

Re-embed your vector databases when:

1. **Resume Updated**: You've modified `docs/resume/ERIC_ABRAM33.docx`
2. **Personal Info Changed**: You've updated `docs/info/personal_information.txt`
3. **Model Upgrade**: You want to use a different embedding model
4. **Performance Issues**: Vector search is returning poor results
5. **Regular Maintenance**: Periodic updates to ensure data freshness

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `404`: Vector database not found (need to run re-embed first)
- `500`: Internal server error (check logs for details)

## Performance Notes

- **Resume re-embedding**: ~1-2 seconds (7 chunks, 1536 dimensions)
- **Personal info re-embedding**: ~1-2 seconds (4 chunks, 1536 dimensions)
- **Batch re-embedding**: ~2-3 seconds (both databases)
- **Search queries**: ~100-500ms per query

## Interactive API Documentation

When the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly in your browser.
