import argparse
import asyncio
import json
import sys

import websockets


DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/answers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print streamed InterviewCopilot answers.")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--session-id", default="default")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    ws_url = f"{args.ws_url}?session_id={args.session_id}"

    async with websockets.connect(ws_url, max_size=None) as websocket:
        print(f"Listening for answers on {ws_url}. Press Ctrl+C to stop.")
        async for message in websocket:
            payload = json.loads(message)
            message_type = payload.get("type")

            if message_type == "ready":
                print(f"Answer stream ready for session '{payload.get('session_id')}'.")
            elif message_type == "transcript":
                print(f"\n[heard] {payload.get('text')}")
            elif message_type == "question_gate":
                if not payload.get("should_answer"):
                    print(f"[ignored: {payload.get('reason')}] {payload.get('text')}")
            elif message_type == "answer_start":
                print(
                    "\n[question] "
                    f"{payload.get('question')}\n"
                    f"[answer:{payload.get('provider')}/{payload.get('model')}] ",
                    end="",
                    flush=True,
                )
            elif message_type == "answer_token":
                print(payload.get("token", ""), end="", flush=True)
            elif message_type == "answer_done":
                print("\n[done]")
            elif message_type == "answer_cancelled":
                print("\n[cancelled]")
            elif message_type == "error":
                print(f"\n[error] {payload.get('message')}", file=sys.stderr)
            else:
                print(payload)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
