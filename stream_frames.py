import redis
import asyncio
import websockets
import cv2
import numpy as np
import base64


async def send_frame(websocket):
    while True:
        message = redis_client.xread({'Frame': '$'}, None, 0)
        final_frame = cv2.imdecode(np.frombuffer(message[0][1][0][1][b'Final_Frame'], np.uint8), cv2.IMREAD_COLOR)
        _, final_frame = cv2.imencode('.JPEG', final_frame)
        final_frame = "data:image/  jpg;base64," + str(base64.b64encode(final_frame).decode())
        await websocket.send(final_frame)
        print("Frame Sent!")

# Initialize redis client
redis_client = redis.Redis(host='127.0.0.1')
print("Redis Client Initialized!\n")

# Initialize websockets server
start_server = websockets.serve(send_frame, "0.0.0.0", 8080)
print("Websocket Server Initialized!\n")

# Run websocket server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()