import argparse
import uvicorn
import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    # Read model + optional card overrides from env
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    default_card_name = f"openai-{model_name}"
    default_card_desc = f"OpenAI {model_name} LLM-only Agent"

    card_name = os.environ.get("AGENT_CARD_NAME", default_card_name).strip() or default_card_name
    card_desc = os.environ.get("AGENT_CARD_DESC", default_card_desc).strip() or default_card_desc
    card_version = os.environ.get("AGENT_CARD_VERSION", "1.0.0").strip() or "1.0.0"

    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="qa_answerer",
        name="QA Answerer",
        description="Answer binary QA questions in the format Final Answer: Yes/No",
        tags=[],
        examples=[]
    )

    agent_card = AgentCard(
        name=card_name,
        description=card_desc,
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version=card_version,
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
