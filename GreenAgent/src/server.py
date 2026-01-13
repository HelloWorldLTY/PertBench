import argparse
import uvicorn

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
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="general_multi_template_qa_evaluator",
        name="General Multi-Template QA Evaluation",
        description="Evaluates QA agents using a benchmark spec (JSON) and ground-truth data (CSV). Computes majority-vote prediction, validity under a minimum-valid threshold, and template robustness metrics (consistency, ambiguous/tie rate, invalid rate, etc.). Outputs detailed per-unit results and a summary artifact for downstream analysis.",
        tags=["qa", "evaluation", "benchmark", "robustness", "templates", "majority-vote", "a2a"],
        examples=["Evaluate participant agents on a binary QA benchmark with several paraphrase templates, majority vote, and consistency.", 
                  "Evaluate participant agents on a CSV of question-answer pairs with strict output parsing."]
    )

    agent_card = AgentCard(
        name="Multi-Template QA Evaluator",
        description="A general-purpose Green agent for QA benchmarking. It reads a JSON benchmark specification and a ground-truth dataset, queries participant agents, and produces reproducible evaluation artifacts.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
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
