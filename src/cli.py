"""Terminal chat interface for the Aster & Row support agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .llm import GroqLLM
from .retrieval import Retriever
from .ingest import load_chunks
from .tools import OrderStore


def build_agent() -> Agent:
    project_root = Path(__file__).resolve().parent.parent

    chunks = load_chunks(str(project_root / "knowledge-base"))
    retriever = Retriever(chunks)
    order_store = OrderStore(str(project_root / "data" / "orders.json"))
    llm = GroqLLM()

    return Agent(retriever, order_store, llm)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is missing from your .env file.")
        return

    try:
        agent = build_agent()
    except Exception as error:
        print(f"Unable to start the agent: {error}")
        return

    session_id = "terminal-session"

    print("\nAster & Row Support Agent")
    print("Type /reset to start a new conversation or /quit to exit.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_message:
            continue

        command = user_message.lower()

        if command in {"/quit", "/exit"}:
            print("Goodbye!")
            break

        if command == "/reset":
            agent.sessions.reset(session_id)
            print("Assistant: Conversation reset.\n")
            continue

        try:
            result = agent.handle_turn(session_id, user_message)
            print(f"\nAssistant: {result.text}\n")

            if result.handoff:
                print("Status: This conversation has been flagged for human support.\n")

        except Exception as error:
            print(f"\nAssistant: I couldn't complete that request right now. ({error})\n")


if __name__ == "__main__":
    main()