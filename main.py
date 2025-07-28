from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets
import asyncio

app = FastAPI()

AGENT_WS_URL = "ws://localhost:8765"  # Agent WebSocket
agent_ws = None  # Global persistent connection

@app.on_event("startup")
async def connect_to_agent():
    """
    Connect to the agent WebSocket when the FastAPI server starts.
    """
    global agent_ws
    print("🔗 Connecting to Agent WebSocket...")
    agent_ws = await websockets.connect(
        AGENT_WS_URL,
        ping_interval=3600,
        ping_timeout=3600
    )
    print("✅ Connected to Agent")

@app.on_event("shutdown")
async def disconnect_agent():
    """
    Close the Agent WebSocket when the server shuts down.
    """
    global agent_ws
    if agent_ws:
        await agent_ws.close()
        print("🔌 Agent connection closed")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handle incoming WebSocket connections from clients.
    """
    global agent_ws
    print("🔌 Incoming WebSocket connection from client...")
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            user_message = await websocket.receive_text()
            print(f"📨 Client says: {user_message}")

            # Forward to Agent
            await agent_ws.send(user_message)

            # Receive response from Agent
            ai_reply = await agent_ws.recv()
            print(f"🤖 Agent replies: {ai_reply}")

            # Send response back to client
            await websocket.send_text(ai_reply)

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
