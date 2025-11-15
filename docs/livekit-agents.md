# LiveKit Python Agents Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Installation and Setup](#installation-and-setup)
4. [Building Your First Agent](#building-your-first-agent)
5. [Agent Architecture](#agent-architecture)
6. [Voice Pipeline Components](#voice-pipeline-components)
7. [Tools and Function Calling](#tools-and-function-calling)
8. [Advanced Features](#advanced-features)
9. [Testing and Evaluation](#testing-and-evaluation)
10. [Deployment](#deployment)
11. [Best Practices](#best-practices)

---

## Introduction

LiveKit Agents is a powerful framework for building production-grade, multimodal AI agents in Python. The framework enables you to create AI agents that can join LiveKit rooms as full participants, processing realtime audio, video, and data streams.

### Key Features

- **Multimodal Support**: Build agents that handle voice, video, and text
- **Production Ready**: Built-in worker orchestration, load balancing, and Kubernetes compatibility
- **Extensive Plugin Ecosystem**: Integrations with OpenAI, Deepgram, Google, ElevenLabs, and more
- **WebRTC Transport**: Reliable, low-latency communication even over unstable connections
- **Open Source**: Fully open-source under Apache 2.0 license
- **Stateful**: Natural state management for conversational AI
- **Tool Use**: LLM-powered function calling to extend agent capabilities

### Use Cases

- AI voice assistants and chatbots
- Call center automation (inbound/outbound)
- Telehealth applications
- Realtime translation
- Video avatars and NPCs
- Robotics control

---

## Core Concepts

### Worker

A **Worker** is the main process that coordinates job scheduling and launches agents for user sessions. When you start your agent application, it registers as a worker with the LiveKit server and waits for job requests.

**Key Characteristics:**
- Registers via persistent WebSocket connection
- Can run multiple jobs simultaneously
- Each job runs in isolated process
- Exchanges availability and capacity info with server
- Supports graceful shutdown

### Job

A **Job** is a single agent instance that handles a user session. When a user connects to a room, the LiveKit server dispatches a job to an available worker.

**Job Lifecycle:**
1. Worker receives job request from LiveKit server
2. Worker accepts request and spawns isolated process
3. Job executes the entrypoint function
4. Agent joins room and interacts with participants
5. Session continues until all participants leave or manual shutdown

### Entrypoint Function

The **entrypoint** is the main function called when a new job starts. It's where you define your agent's logic and behavior.

```python
async def entrypoint(ctx: agents.JobContext):
    # Your agent logic here
    await ctx.connect()
    
    # Create and start your agent
    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent)
```

### JobContext

The **JobContext** provides access to the room, agent API, and job information.

**Key Properties:**
- `ctx.room`: The LiveKit room instance
- `ctx.agent`: The local participant (your agent)
- `ctx.api`: LiveKit API for server operations
- `ctx.job`: Job metadata and information
- `ctx.proc`: Process information

**Key Methods:**
- `ctx.connect()`: Connect agent to the room
- `ctx.add_shutdown_callback()`: Register cleanup functions
- `ctx.add_participant_entrypoint()`: Handle per-participant logic

---

## Installation and Setup

### Prerequisites

- Python >= 3.9
- LiveKit Cloud account or self-hosted LiveKit server
- API keys for AI providers (OpenAI, Deepgram, etc.)

### Installation

Using `uv` (recommended):

```bash
# Create new project
uv init my-agent
cd my-agent

# Install LiveKit Agents with plugins
uv add "livekit-agents[openai,deepgram,silero]"
uv add python-dotenv
```

Using `pip`:

```bash
# Install base SDK
pip install livekit-agents

# Install with plugins
pip install "livekit-agents[openai,deepgram,elevenlabs,silero]"
pip install python-dotenv
```

### Environment Configuration

Create a `.env` file:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# AI Provider Keys
OPENAI_API_KEY=your_openai_key
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### LiveKit CLI Setup

```bash
# Install CLI
pip install livekit-cli

# Authenticate with LiveKit Cloud
lk cloud auth

# Link your project
lk app env -w
```

---

## Building Your First Agent

### Simple Voice Agent

Here's a complete example of a basic voice agent:

```python
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, deepgram, silero, noise_cancellation

load_dotenv()

class MyAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice AI assistant."
        )

async def entrypoint(ctx: agents.JobContext):
    # Connect to the room
    await ctx.connect()
    
    # Create agent session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
    )
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=MyAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )
    
    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

### Using OpenAI Realtime API

For simpler speech-to-speech:

```python
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="coral")
    )
    
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful assistant.")
    )
    
    await session.generate_reply(
        instructions="Greet the user warmly."
    )
```

### Running Your Agent

**Development Mode:**
```bash
uv run agent.py dev
```

**Console Mode (Terminal only):**
```bash
lk agent dev --console
```

**Production Mode:**
```bash
uv run agent.py start
```

---

## Agent Architecture

### AgentSession

`AgentSession` is the main orchestrator for voice AI applications. It manages the voice pipeline, collects user input, invokes the LLM, and sends output back to users.

**Key Components:**
- **VAD (Voice Activity Detection)**: Detects when user is speaking
- **STT (Speech-to-Text)**: Converts speech to text
- **LLM (Large Language Model)**: Generates responses
- **TTS (Text-to-Speech)**: Converts text to speech
- **Turn Detection**: Determines when user has finished speaking

**Creation:**

```python
session = AgentSession(
    vad=silero.VAD.load(),
    stt=deepgram.STT(model="nova-3"),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=elevenlabs.TTS(voice="Rachel"),
    # Optional parameters
    allow_interruptions=True,
    min_endpointing_delay=0.5,
    max_endpointing_delay=6.0,
)
```

**Using LiveKit Inference (String Descriptors):**

```python
session = AgentSession(
    stt="assemblyai/universal-streaming:en",
    llm="openai/gpt-4o-mini",
    tts="cartesia/sonic-3:voice-id",
    vad=silero.VAD.load(),
)
```

### Agent Class

The `Agent` class defines your agent's behavior, instructions, and capabilities.

**Basic Agent:**

```python
class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant specialized in weather.",
            # Optional: Override session defaults
            llm=openai.LLM(model="gpt-4"),
            temperature=0.7,
        )
```

**Agent with Lifecycle Hooks:**

```python
class MyAgent(Agent):
    async def on_enter(self):
        """Called when agent becomes active"""
        await self.say("Hello! How can I help you today?")
    
    async def on_user_turn_completed(
        self, 
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage
    ):
        """Called when user finishes speaking"""
        # You can modify the chat context here
        pass
    
    async def on_agent_turn_started(self):
        """Called when agent starts responding"""
        pass
    
    async def on_exit(self):
        """Called when agent is deactivated"""
        await self.say("Goodbye!")
```

### WorkerOptions

Configure worker behavior:

```python
agents.cli.run_app(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=custom_request_handler,  # Optional
        prewarm_fnc=prewarm_models,  # Optional
        num_idle_processes=3,  # Keep warm processes
        shutdown_process_timeout=60.0,
        load_threshold=0.75,
    )
)
```

---

## Voice Pipeline Components

### Voice Activity Detection (VAD)

VAD detects when a user is speaking.

**Silero VAD (Recommended):**

```python
from livekit.plugins import silero

vad = silero.VAD.load()
```

### Speech-to-Text (STT)

Converts user speech to text.

**Deepgram:**

```python
from livekit.plugins import deepgram

stt = deepgram.STT(
    model="nova-3",
    language="en",
    smart_format=True,
)
```

**AssemblyAI:**

```python
from livekit.plugins import assemblyai

stt = assemblyai.STT(
    model="universal-streaming",
    language="en"
)
```

**OpenAI Whisper:**

```python
from livekit.plugins import openai

stt = openai.STT(model="whisper-1")
```

### Large Language Models (LLM)

Generates intelligent responses.

**OpenAI:**

```python
from livekit.plugins import openai

llm = openai.LLM(
    model="gpt-4o-mini",
    temperature=0.7,
)
```

**Anthropic Claude:**

```python
from livekit.plugins import anthropic

llm = anthropic.LLM(
    model="claude-sonnet-4-20250514",
    temperature=0.8,
)
```

**Google Gemini:**

```python
from livekit.plugins import google

llm = google.LLM(model="gemini-2.0-flash-exp")
```

### Text-to-Speech (TTS)

Converts agent responses to speech.

**OpenAI TTS:**

```python
from livekit.plugins import openai

tts = openai.TTS(
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    speed=1.0,
)
```

**ElevenLabs:**

```python
from livekit.plugins import elevenlabs

tts = elevenlabs.TTS(
    voice="Rachel",
    model="eleven_turbo_v2_5",
)
```

**Cartesia:**

```python
from livekit.plugins import cartesia

tts = cartesia.TTS(
    voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    model="sonic-3",
)
```

### Turn Detection

Advanced turn detection using transformer models:

```python
from livekit.plugins import turn_detector

session = AgentSession(
    vad=silero.VAD.load(),
    turn_detection=turn_detector.MultilingualModel(),
    # ... other components
)
```

**Features:**
- 85% true positive rate
- 97% true negative rate
- Prevents early interruptions
- Multilingual support

---

## Tools and Function Calling

LiveKit Agents supports LLM tool/function calling to extend agent capabilities.

### Defining Tools with @function_tool

**Basic Tool:**

```python
from livekit.agents import function_tool, RunContext

@function_tool
async def get_weather(
    context: RunContext,
    location: str,
) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    # Your weather API call here
    return f"The weather in {location} is sunny, 72°F"
```

**Adding Tools to Agent:**

```python
class WeatherAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a weather assistant.",
            tools=[get_weather],
        )
```

### Advanced Tool Features

**Accessing Agent State:**

```python
@function_tool
async def book_appointment(
    context: RunContext,
    date: str,
    time: str,
) -> str:
    """Book an appointment.
    
    Args:
        date: Date in YYYY-MM-DD format
        time: Time in HH:MM format
    """
    # Access session
    session = context.agent.session
    
    # Store in user data
    context.userdata.appointments.append({
        "date": date,
        "time": time
    })
    
    return f"Appointment booked for {date} at {time}"
```

**Generating Agent Speech from Tools:**

```python
@function_tool
async def tell_joke(context: RunContext) -> str:
    """Tell a random joke"""
    
    # Make agent speak directly
    await context.agent.say("Here's a joke for you!")
    
    return "Why did the programmer quit? They didn't get arrays!"
```

**Tool with Type Validation:**

```python
from typing import Literal

@function_tool
async def set_temperature(
    context: RunContext,
    temperature: int,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    """Set the room temperature.
    
    Args:
        temperature: Temperature value
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Temperature set to {temperature}°{unit[0].upper()}"
```

### Raw Function Schemas

For advanced use cases:

```python
from livekit.agents import function_tool

weather_tool = function_tool(
    name="get_weather",
    description="Get weather for a location",
    raw_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    }
)

@weather_tool
async def handler(raw_arguments: dict):
    location = raw_arguments["location"]
    return f"Weather in {location}"
```

### Error Handling in Tools

```python
from livekit.agents import function_tool, ToolError

@function_tool
async def database_lookup(context: RunContext, user_id: str) -> str:
    """Look up user information"""
    
    try:
        result = await db.get_user(user_id)
        return result
    except UserNotFound:
        raise ToolError("User not found. Please check the ID.")
```

---

## Advanced Features

### Multi-Agent Handoff

Transfer control between specialized agents:

```python
from livekit.agents import Agent, function_tool, RunContext

class RouterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="Route users to the right specialist.",
            tools=[transfer_to_sales, transfer_to_support]
        )

@function_tool
async def transfer_to_sales(context: RunContext) -> tuple[Agent, str]:
    """Transfer to sales team"""
    return SalesAgent(), "Transferring you to sales..."

@function_tool
async def transfer_to_support(context: RunContext) -> tuple[Agent, str]:
    """Transfer to support team"""
    return SupportAgent(), "Connecting you with support..."

class SalesAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a sales specialist.",
            llm=openai.realtime.RealtimeModel(voice="echo")
        )
```

### Custom Pipeline Nodes

Override default behavior:

```python
class CustomAgent(Agent):
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        """Custom LLM processing"""
        # Modify chat context
        custom_ctx = self.preprocess_context(chat_ctx)
        
        # Call LLM with custom logic
        stream = await self.llm.chat(
            chat_ctx=custom_ctx,
            tools=tools,
            temperature=0.9
        )
        
        async for chunk in stream:
            yield chunk
    
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ):
        """Custom TTS processing"""
        async for text_chunk in text:
            # Apply custom text transformations
            processed = self.postprocess_text(text_chunk)
            
            # Generate audio
            async for audio in self.tts.synthesize(processed):
                yield audio
```

### Background Audio

Add ambient sounds and thinking indicators:

```python
from livekit.agents import BackgroundAudioPlayer, AudioConfig

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    # Create background audio player
    player = BackgroundAudioPlayer(
        room=ctx.room,
        ambient_sound=AudioConfig(
            source="https://example.com/office-ambience.mp3",
            volume=0.3,
        ),
        thinking_sound=AudioConfig(
            source="https://example.com/thinking.mp3",
            volume=0.5,
        ),
    )
    
    await player.start()
    
    # Create session
    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent)
```

### Transcription Control

Configure transcription forwarding:

```python
from livekit.agents import AgentSession, TranscriptionOptions

session = AgentSession(
    # ... components
    transcription=TranscriptionOptions(
        user_transcription=True,  # Forward user speech
        agent_transcription=True,  # Forward agent speech
        agent_transcription_speed=1.0,
    )
)
```

### Telephony Integration

**Inbound Calls:**

```python
# Configure dispatch rules in LiveKit Cloud dashboard
# Agent automatically joins when calls arrive

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    
    session = AgentSession(
        # Use BVCTelephony for phone calls
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony()
        )
    )
    
    await session.start(room=ctx.room, agent=agent)
    
    # Answer the call
    await session.generate_reply(
        instructions="Answer the phone professionally."
    )
```

**Outbound Calls:**

```python
from livekit import api

async def entrypoint(ctx: agents.JobContext):
    # Get phone number from metadata
    phone_number = json.loads(ctx.job.metadata)["phone_number"]
    
    if phone_number:
        # Place outbound call
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id="your_trunk_id",
                sip_call_to=phone_number,
                participant_identity=phone_number,
            )
        )
    
    # Rest of agent logic
    session = AgentSession(...)
    await session.start(room=ctx.room, agent=agent)
```

---

## Testing and Evaluation

### Test Setup

Install testing dependencies:

```bash
pip install pytest pytest-asyncio
```

### Basic Test Example

```python
import pytest
from livekit import agents
from livekit.agents.testing import run

@pytest.mark.asyncio
async def test_greeting():
    """Test that agent greets user"""
    
    result = await run(
        agent_cls=MyAssistant,
        agent_session=session,
        input="Hello",
    )
    
    # Assert agent responded
    result.expect.next_event().is_message(role="assistant")
    
    # Check message content contains greeting
    result.expect.next_event().contains_text("hello", case_sensitive=False)
```

### Testing Tool Calls

```python
@pytest.mark.asyncio
async def test_weather_tool():
    """Test weather tool is called correctly"""
    
    result = await run(
        agent_cls=WeatherAgent,
        agent_session=session,
        input="What's the weather in San Francisco?",
    )
    
    # Assert tool was called
    result.expect.next_event().is_tool_call(name="get_weather")
    
    # Assert correct arguments
    tool_call = result.expect.next_event().event()
    assert tool_call.arguments["location"] == "San Francisco"
```

### Using LLM Judges

```python
from livekit.plugins import openai

@pytest.mark.asyncio
async def test_response_quality():
    """Test response meets quality standards"""
    
    result = await run(
        agent_cls=MyAgent,
        agent_session=session,
        input="Explain quantum computing",
    )
    
    # Use LLM to judge response quality
    result.expect.next_event().is_message().judge(
        intent="The response should explain quantum computing "
               "in simple terms suitable for beginners",
        llm=openai.LLM(model="gpt-4o"),
    )
```

### Testing Error Handling

```python
@pytest.mark.asyncio
async def test_invalid_input():
    """Test agent handles invalid input gracefully"""
    
    result = await run(
        agent_cls=MyAgent,
        agent_session=session,
        input="!@#$%^&*()",
    )
    
    # Assert agent asks for clarification
    result.expect.next_event().is_message().contains_text(
        "understand", case_sensitive=False
    )
```

### Mock Tools for Testing

```python
from livekit.agents import mock_tools

async def mock_weather(location: str) -> str:
    return "Sunny, 75°F"

def test_with_mocked_tools():
    with mock_tools(WeatherAgent, {"get_weather": mock_weather}):
        result = await run(
            agent_cls=WeatherAgent,
            agent_session=session,
            input="What's the weather?",
        )
        
        # Test uses mock instead of real API
        result.expect.next_event().is_message()
```

---

## Deployment

### Deploy to LiveKit Cloud

**Initial Deployment:**

```bash
# Ensure you have livekit.toml configured
lk agent deploy

# Or specify agent name
lk agent deploy --name my-agent
```

**Update Deployment:**

```bash
lk agent deploy
```

**Rollback:**

```bash
lk agent rollback
```

### Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Download model files
RUN python agent.py download-files

# Run agent
CMD ["python", "agent.py", "start"]
```

### Configuration (livekit.toml)

```toml
[agent]
name = "my-voice-agent"
version = "1.0.0"
region = "us-west-2"

[build]
command = "docker build -t my-agent ."
dockerfile = "Dockerfile"

[start]
command = "python agent.py start"
```

### Environment Variables

For production, use environment variables:

```python
import os
from dotenv import load_dotenv

# Only load .env in development
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

# Access variables
OPENAI_KEY = os.environ["OPENAI_API_KEY"]
```

### Custom Deployment

**Kubernetes Example:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: livekit-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: livekit-agent
  template:
    metadata:
      labels:
        app: livekit-agent
    spec:
      containers:
      - name: agent
        image: your-registry/livekit-agent:latest
        env:
        - name: LIVEKIT_URL
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: url
        - name: LIVEKIT_API_KEY
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: api-key
        - name: LIVEKIT_API_SECRET
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: api-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Resource Requirements

**Recommended per worker:**
- CPU: 4 cores
- Memory: 8GB
- Storage: 10GB ephemeral
- Handles: 10-25 concurrent jobs

**Graceful Shutdown:**

Configure adequate grace period (10+ minutes for voice agents):

```python
agents.cli.run_app(
    agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        shutdown_process_timeout=600.0,  # 10 minutes
    )
)
```

---

## Best Practices

### Security

1. **Never commit secrets**: Use environment variables
2. **Separate environments**: Different API keys for dev/staging/prod
3. **Validate inputs**: Sanitize user inputs before processing
4. **Rate limiting**: Implement appropriate rate limits

### Performance

1. **Use VAD effectively**: Silero VAD is recommended
2. **Enable preemptive synthesis**: Reduce latency
3. **Warm processes**: Keep idle processes warm
4. **Monitor metrics**: Track TTFT, latency, success rates

### Reliability

1. **Error handling**: Always handle exceptions in tools
2. **Graceful degradation**: Fallback for service failures
3. **Testing**: Write comprehensive tests
4. **Monitoring**: Log important events and metrics

### Development Workflow

1. **Local development**: Use `dev` mode for fast iteration
2. **Testing**: Run tests before deployment
3. **Staging**: Test in staging environment
4. **Deployment**: Use rolling deployments
5. **Monitoring**: Watch logs and metrics post-deployment

### Agent Instructions

1. **Be specific**: Clear, detailed instructions
2. **Set boundaries**: Define what agent can/cannot do
3. **Provide examples**: Include example interactions
4. **Iterate**: Refine based on user feedback

### Code Organization

```
my-agent/
├── agent.py              # Main agent code
├── tools/
│   ├── __init__.py
│   ├── weather.py        # Weather tools
│   └── calendar.py       # Calendar tools
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   └── conftest.py
├── .env.example
├── requirements.txt
├── Dockerfile
├── livekit.toml
└── README.md
```

---

## Summary

LiveKit Agents for Python provides a powerful, production-ready framework for building voice AI applications. Key takeaways:

1. **Core Architecture**: Worker → Job → Entrypoint → AgentSession
2. **Components**: VAD, STT, LLM, TTS work together in a pipeline
3. **Tools**: Extend capabilities with function calling
4. **Testing**: Built-in testing framework for reliability
5. **Deployment**: Simple deployment to LiveKit Cloud or custom environments
6. **Flexibility**: Support for multiple AI providers and custom logic

For more information, visit:
- Documentation: https://docs.livekit.io/agents/
- GitHub: https://github.com/livekit/agents
- Examples: https://github.com/livekit-examples/
- Community: LiveKit Slack

Happy building! 🚀
