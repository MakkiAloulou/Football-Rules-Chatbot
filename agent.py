import asyncio
import websockets
import groq
import os
from dotenv import load_dotenv
import nest_asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.retriever import retrieve_chunks  

# Load environment variables from .env file
load_dotenv()
nest_asyncio.apply()

class Agent:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Agent, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.system_prompt = None
        self.retriever = retrieve_chunks 

    def getSystemPrompt(self, top_k_chunks, user_question):
        self.system_prompt = f"""
        You are a domain-specific chatbot assistant trained to help users understand
        official football rules and related information. Your job is to provide clear, accurate, and easy-to-understand answers using only the provided context.

        Follow these rules:
        1. Only answer using the information in the context below.
        2. If the context does not contain enough information to answer the question, respond with:
        "I’m sorry, the context provided does not contain enough detail about this topic."
        3. Explain football terms and rules in simple language, as if speaking to someone who enjoys football but is not an expert.
        4. When explaining a rule, include details such as conditions, penalties, and examples if available.
        5. Avoid vague summaries. Be specific and informative.
        6. Format your response using Markdown. Use proper headings, bullet points, and examples for clarity.
        7. When you mention the source (e.g., a rule section or page number), make it **bold** using Markdown.
        8. Make the response concise but informative. Avoid unnecessary repetition.

        ---

        Example:
        **Context Chunks (Extracted Passages):**
        [Source: Laws of the Game - Page 34]
        The offside position occurs when a player is nearer to the opponent's goal line than both the ball and the second-last opponent.

        **User Question:**
        What is the offside rule?

        **Your Answer (Based on Context Only):**
        The **offside rule** states that a player is in an offside position if they are nearer to the opponent's goal line than both the ball and the second-last opponent when the ball is played, except when they are in their own half or receiving the ball directly from a goal kick, throw-in, or corner kick.

        ---

        **Context Chunks (Extracted Passages):**
        {top_k_chunks}

        **User Question:**
        {user_question}

        **Your Answer (Based on Context Only):**
        """
        return self.system_prompt

    async def handle_connection(self, websocket, path=None):
        print("🤖 Agent connected and ready for queries.")

        messages = [{"role": "system", "content": "You are a helpful football rules assistant."}]

        while True:
            try:
                message = await websocket.recv()
                print(f"📨 Received from client: {message}")

                # Retrieve context
                relevant_chunks = self.retriever(message, top_k=5)
                if not relevant_chunks:
                    await websocket.send("I couldn't find any related rule in the documents.")
                    continue

                # Prepare context
                context = "\n\n".join([
                    f"[Source: {os.path.basename(chunk['source'])}]\n{chunk['text']}"
                    for chunk in relevant_chunks
                ])

                system_prompt = self.getSystemPrompt(context, message)

                # Add messages
                messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": message})

                # Get LLM response
                response = self.client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL"),
                    messages=messages
                )

                llm_reply = response.choices[0].message.content
                await websocket.send(llm_reply)
                print(f"🤖 Sent: {llm_reply[:80]}...")

                messages.append({"role": "assistant", "content": llm_reply})

            except websockets.ConnectionClosed:
                print("❌ Client disconnected from Agent.")
                break

    # Start the WebSocket server
    async def start_server(self):
        async with websockets.serve(
            self.handle_connection, 
            "localhost", 
            8765,
            ping_interval=3600, 
            ping_timeout=3600
        ):
            print("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # Keep the server running

# Run the WebSocket server
if __name__ == "__main__":
    agent = Agent()
    asyncio.run(agent.start_server())