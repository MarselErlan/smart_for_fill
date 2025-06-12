# Smart Form Fill

AI-powered form analyzer and auto-filler.

## API Usage

The API provides endpoints to analyze forms and saves URLs to Supabase:

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables (copy from .env-example):

```bash
cp .env-example .env
# Edit .env to add your API keys
export OPENAI_API_KEY=your_api_key_here
export SUPABASE_URL=your_supabase_url_here
export SUPABASE_KEY=your_supabase_key_here
```

3. Set up Supabase:

   - Create a new table called `forms` with these columns:
     - `id` (default primary key)
     - `url` (text, unique)
     - `created_at` (timestamp with timezone, default: now())
     - `analyzed` (boolean, default: false)

4. Run the API:

```bash
python main.py
```

The API will be available at http://localhost:8000

### Project Structure

The API uses a clean separation between models and schemas:

- **Models** (`app/models/models.py`): Database entities and internal data structures
- **Schemas** (`app/models/schemas.py`): API request/response schemas used for validation

### Endpoints

#### Analyze Form

```
POST /api/analyze-form
```

Request body:

```json
{
  "url": "https://example.com/form"
}
```

Response:

```json
{
  "status": "success",
  "field_map": "...",
  "timestamp": "...",
  "database_id": "...",
  "url": "https://example.com/form"
}
```

#### Analyze Form with Detailed Fields

```
POST /api/analyze-form-detailed
```

Request body:

```json
{
  "url": "https://example.com/form"
}
```

Response:

```json
{
  "status": "success",
  "field_map": "...",
  "timestamp": "...",
  "database_id": "...",
  "url": "https://example.com/form",
  "fields": [
    {
      "field_type": "text",
      "purpose": "full_name",
      "selector": "#fullName",
      "validation": "required"
    }
  ]
}
```

#### Get All Forms

```
GET /api/forms
```

Response:

```json
[
  {
    "id": "...",
    "url": "https://example.com/form",
    "created_at": "2023-...",
    "analyzed": true
  }
]
```

#### Health Check

```
GET /api/health
```

Response:

```json
{
  "status": "healthy"
}
```
# smart_for_fill
