import streamlit as st
import asyncio
import websockets

st.set_page_config(page_title="Chatbot", layout="centered", page_icon="🤖")
st.title("🤖 Football Rules Assistant")

# WebSocket URL (FastAPI app)
CLIENT_WS_URL = "ws://localhost:8080/ws"

# Persistent chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
question = st.chat_input("Ask a question about a Football rule...")

# Function to interact with FastAPI WebSocket server
async def ask_ai(query):
    try:
        async with websockets.connect(CLIENT_WS_URL, ping_interval=3600, ping_timeout=3600) as websocket:
            await websocket.send(query)  # send the user question
            response = await websocket.recv()  # get the AI's reply
            return response
    except Exception as e:
        return f"❌ Error: {e}"

# When user submits a question
if question:
    st.session_state.chat_history.append(("user", question))

    with st.spinner("Thinking..."):
        answer = asyncio.run(ask_ai(question))
        st.session_state.chat_history.append(("ai", answer))

# Render conversation history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
