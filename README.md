# LiveKit Documentation Assistant Agent

An AI-powered voice assistant built with LiveKit Agents framework that helps users learn about LiveKit documentation through natural conversation.

## Features

- **Semantic Search**: Vector-based similarity search across LiveKit documentation using Chroma and HuggingFace embeddings
- **Web Search Integration**: Live web search capability using SerpAPI for the latest documentation
- **Conversation Persistence**: Save and retrieve previous conversations via FastAPI backend
- **Audio Enhancement**: Built-in noise cancellation and voice activity detection (VAD)

## Architecture

The project consists of three main components:

### 1. Voice Agent ([agent.py](agent.py))
The core LiveKit agent that handles:
- Voice-to-text transcription (Deepgram STT)
- Text-to-speech synthesis (Cartesia TTS)
- Knowledge base search using vector embeddings
- Web search via SerpAPI
- Conversation management

### 2. API Server ([api/server.py](api/server.py))
FastAPI backend providing:
- `POST /conversation` - Save conversation messages
- `GET /conversation` - Retrieve conversation history
- `GET /conversation/stats` - Get conversation statistics
- `DELETE /conversation` - Clear conversation history

### 3. Documentation Indexer ([scripts/md_to_vectordb.py](scripts/md_to_vectordb.py))
Utility script to convert Markdown documentation into a searchable vector database.

## Prerequisites

- Python 3.11+
- LiveKit account and API credentials
- OpenAI API key
- SerpAPI key (for web search)
- HuggingFace access token (for embeddings model)
- Cartesia API key (for text-to-speech)
- Deepgram API key (for speech-to-text)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:joshiayush/knowledge-base-agent.git
cd knowledge-base-agent
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```env
LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>
OPENAI_API_KEY=<your-openai-api-key>
SERP_API_KEY=<your-serpapi-key>
HUGGINGFACEHUB_ACCESS_TOKEN=<your-huggingface-token>
CARTESIA_API_KEY=<your-cartesia-api-key>
DEEPGRAM_API_KEY=<your-deepgram-api-key>
```

## Setup

### 1. Index Documentation
Place your Markdown documentation files in the `docs/` directory, then run:
```bash
python scripts/md_to_vectordb.py
```

This creates a Chroma vector database in `livekit_agents_db/` for semantic search.

### 2. Start the API Server
```bash
cd api
uvicorn server:app --reload --port 8000
```

### 3. Run the Agent
```bash
python agent.py
```

### 4. Run the Web Client

The web interface is built using the [LiveKit Agent Starter React](https://github.com/livekit-examples/agent-starter-react) template.

#### Installation

1. **Clone the agent-starter-react repository:**
   ```bash
   git clone https://github.com/livekit-examples/agent-starter-react.git
   cd agent-starter-react
   ```

   Alternatively, use the LiveKit CLI:
   ```bash
   lk app create --template agent-starter-react
   ```

2. **Install dependencies:**
   ```bash
   pnpm install
   ```

   If you don't have pnpm installed:
   ```bash
   npm install -g pnpm
   ```

3. **Configure environment variables:**

   Create a `.env.local` file in the root directory:
   ```env
   LIVEKIT_API_KEY=<your-livekit-api-key>
   LIVEKIT_API_SECRET=<your-livekit-api-secret>
   LIVEKIT_URL=<your-livekit-url>
   ```

   Use the same LiveKit credentials from your main agent `.env` file.

4. **Start the development server:**
   ```bash
   pnpm dev
   ```

5. **Access the web interface:**

   Open [http://localhost:3000](http://localhost:3000) in your browser.


## Usage

**Note:** This is a demo project. Logical routing is not yet implemented, so you need to be explicit with your queries to help the agent understand your intent. Examples:
- For knowledge base search: "Search the knowledge base for information about LiveKit agents"
- For web search: "Search the web for recent LiveKit updates"
- Instead of vague queries like "Tell me about agents", be specific about which tool you want to use

The agent provides several function tools accessible through natural conversation:

### Search Knowledge Base
Ask the agent about LiveKit concepts:
> "How do I create a new agent in LiveKit?"

The agent searches the indexed documentation using semantic search.

### Web Search
Request the latest information:
> "Search the web for recent LiveKit updates"

### Save Conversation
> "Save our conversation"

### View Previous Conversations
> "Show me our previous conversations"

## Technical Details

### Vector Database
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Vector Store**: Chroma

### Voice Processing
- **STT**: Deepgram
- **TTS**: Cartesia (Sonic-3 model)
- **VAD**: Silero
- **Turn Detection**: English language model
- **Noise Cancellation**: BVC (Background Voice Cancellation)

### LLM Configuration
- **Model**: GPT-4o-mini

## API Endpoints

### Save Message
```bash
curl -X POST http://localhost:8000/conversation \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Hello"}'
```

### Get Conversations
```bash
curl http://localhost:8000/conversation
```

### Get Statistics
```bash
curl http://localhost:8000/conversation/stats
```

### Clear Conversations
```bash
curl -X DELETE http://localhost:8000/conversation
```
