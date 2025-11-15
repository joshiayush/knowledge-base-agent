import httpx
import os
import aiohttp
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    RoomInputOptions,
    JobProcess,
)
from livekit.plugins import openai, deepgram, silero, cartesia, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


system_prompt = """<identity>
You are a LiveKit documentation expert assistant.
</identity>

<task>
Help users learn about LiveKit Agents framework by:
- Searching the vector knowledge base for relevant documentation using semantic search
- Checking the cache for similar questions
- Searching the web for the latest LiveKit documentation if needed
- Caching your responses for future queries
</task>

<rules>
- Important: Speak naturally as if explaining to a colleague. Do NOT use bullet points, numbered lists, or formatting markers like 1., 2., -, *, etc. Instead, use connecting words like 'first', 'additionally', 'also', 'furthermore' to flow between ideas. Avoid any code or special formatting characters.
- When asked to get previous conversation, only say a recap of the previous conversation and DO NOT call save_conversation() after the call to get_previous_conversation()
</rules>
"""


class LiveKitLearningAgent(Agent):
    def __init__(self, hf_embeddings: HuggingFaceEmbeddings):
        super().__init__(instructions=system_prompt)
        self.api_base = "http://127.0.0.1:8000"

        self.embeddings = hf_embeddings
        self.vectorstore = Chroma(
            persist_directory=str(Path(__file__).parent / "livekit_agents_db"),
            collection_name="livekit_agents_docs",
            embedding_function=self.embeddings,
        )

    @function_tool
    async def search_knowledge_base(
        self,
        ctx: RunContext,
        query: Annotated[
            str,
            Field(
                description="The search query to find relevant LiveKit documentation"
            ),
        ],
    ) -> str:
        """Search the LiveKit documentation knowledge base using semantic search."""
        try:
            docs = self.vectorstore.similarity_search(query, k=3)

            if docs:
                results = []
                for i, doc in enumerate(docs, 1):
                    results.append(f"Result {i}:")
                    results.append(doc.page_content[:500])
                    results.append("")

                return "\n".join(results)
            else:
                return "I couldn't find any relevant information for that query."

        except Exception as exc:
            return (
                f"I encountered an error while searching the knowledge base: {str(exc)}"
            )

    @function_tool
    async def search_web(
        self,
        ctx: RunContext,
        query: Annotated[
            str, Field(description="The search query to look up on the web")
        ],
        num_results: Annotated[
            int, Field(description="Number of results to return (default: 5)")
        ] = 5,
    ) -> str:
        """Search the web using Google Search via SerpAPI.

        Returns formatted search results with titles, URLs, and snippets.
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "api_key": os.getenv("SERP_API_KEY"),
                    "q": query,
                    "num": num_results,
                    "engine": "google",
                }

                async with session.get(
                    "https://serpapi.com/search", params=params
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return f"Error performing search: {error_text}"

                    data = await response.json()

                    results = []

                    if data.get("organic_results"):
                        results.append("Search Results:\n")
                        for result in data["organic_results"][:num_results]:
                            title = result.get("title", "No title")
                            snippet = result.get("snippet", "No description available")

                            results.append(f"-  {title}")
                            results.append(f"   {snippet}\n")
                    else:
                        results.append("No results found.")

                    formatted_results = "\n".join(results)
                    return formatted_results

        except Exception as e:
            return f"I encountered an error while running a web search: {str(e)}"

    @function_tool
    async def save_conversation(self, ctx: RunContext) -> None:
        """Save the current conversation history to the database.

        Called when the user explicitly asks to save the conversation.
        Saves all messages from the current session to the backend API.
        """
        try:
            chat_items = self.chat_ctx.items

            async with httpx.AsyncClient() as client:
                saved_count = 0
                for item in chat_items:
                    if item.type == "message":
                        # Save each message to the conversation API
                        await client.post(
                            f"{self.api_base}/conversation",
                            json={
                                "role": item.role,
                                "content": item.text_content or "",
                            },
                        )
                        saved_count += 1

            await ctx.session.say(
                f"I've saved {saved_count} messages from our conversation to the database."
            )
        except Exception as e:
            await ctx.session.say(
                "I encountered an error while saving the conversation. Please try again."
            )

    @function_tool
    async def get_previous_conversations(self, ctx: RunContext) -> None:
        """Retrieve and summarize previous conversations from the database.

        Called when the user asks to see or hear about previous conversations.
        Fetches all saved messages and generates a natural language summary.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base}/conversation")
                data = response.json()

                messages = data.get("messages", [])

                if not messages:
                    await ctx.session.say(
                        "There are no previous conversations saved in the database."
                    )
                    return

                # Build a summary of the conversation
                conversation_text = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in messages
                        if msg["role"] != "system"
                    ]
                )

            return f"Previous conversation:\n{conversation_text}"
        except Exception as e:
            await ctx.session.say(
                "I encountered an error while retrieving previous conversations. Please try again."
            )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(
            voice="c1c65fc2-528a-4dde-a2c4-f822785c2704",
            model="sonic-3",
            language="en",
        ),
        turn_detection=EnglishModel(),
    )
    agent = LiveKitLearningAgent(hf_embeddings=ctx.proc.userdata["hf_embeddings"])

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["hf_embeddings"] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
