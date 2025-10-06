## Purpose

This file gives concise, actionable guidance for an AI coding agent (or human reviewer) to be productive in this repository. It focuses on the big-picture architecture, critical run/debug workflows, project-specific conventions, and integration points you should know before making changes.

## Quick summary

- Two primary flavors of workflows live here:
  - Message-based graph (main.py) — uses `MessageGraph` for LLM-driven prompt/tool orchestration.
  - State-based workflow (pod_workflow.py / trend_agent.py) — uses `StateGraph` for autonomous, side-effectful pipelines (creates Printify products, generates images).
- Tool integrations are implemented via `langgraph` ToolNode/StructuredTool and the `schemas.py` Pydantic models (e.g. `AnswerQuestion`, `ReviseAnswer`).
- Network calls (OpenAI, Printify, Shopify) are done directly via `openai.OpenAI` client and `requests`.

## How to run locally (developer notes)

- This project uses Poetry. Recommended quick setup:

```pwsh
poetry install
poetry run python main.py         # runs the message-based example graph
poetry run python pod_workflow.py # runs the state-based Printify workflow example
```

- Environment variables are required (can be provided in a `.env` file). Key vars discovered:
  - OPENAI_API_KEY (required)
  - PRINTIFY_API_KEY (required for Printify calls in `pod_workflow.py`)
  - SHOPIFY_STORE_URL (required for publishing links)
  - PRINTIFY_SHOP_ID (used by `pod_workflow.py`)
  - OPENAI_TREND_MODEL, TREND_TEMPERATURE, STORE_NICHE, PRINTIFY_DEFAULT_PRICE_CENTS (optional overrides)

If these are missing, `pod_workflow.py` will raise ValueError early. `langsmith.traceable` is optional — code provides a no-op fallback.

## Key files and what to look at

- `main.py` — small example that builds a `MessageGraph` and wires: `first_responder` -> `execute_tools` -> `revisor`. Use it to understand message-driven orchestration and how tools are invoked.
- `chains.py` — defines LLM prompts, `ChatOpenAI` usage, output parsers (`JsonOutputToolsParser`, `PydanticToolsParser`) and the `first_responder`/`revisor` actors. Look here for prompt templates and parser contracts.
- `tool_executor.py` — shows how tools are wrapped: `TavilySearch` and `ToolNode` usage. Concrete example: `StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__)`.
- `schemas.py` — Pydantic models for tool inputs/outputs (AnswerQuestion, ReviseAnswer). Follow these models when adding or modifying tools.
- `pod_workflow.py` — the stateful, side-effecting workflow that:
  - builds a `StateGraph` of nodes (identify_trend -> generate_design -> select_products -> create_single_product)
  - integrates with OpenAI image generation (`_openai_client.images.generate`) and with Printify/Shopify via `requests`
  - uses `PodState` TypedDict to carry state between nodes and a `log(state, msg)` helper that appends messages to `state['logs']`.

## Project-specific patterns & gotchas

- Dual graph styles: MessageGraph vs StateGraph — don't conflate the two. `main.py` is message-first; `pod_workflow.py` is state-first and performs network side effects.
- Errors are handled by logging and safe fallbacks. Many network calls wrap requests in try/except and append human-readable messages to `state['logs']`.
- `traceable` decorator: code attempts to import `langsmith.traceable`. If LangSmith isn't installed/configured the repo defines a no-op replacement. Preserve decorator usage when editing: it's used to mark nodes for optional tracing.
- Image upload approaches differ across files: `trend_agent.py` uploads via an image URL payload, while `pod_workflow.py` shows a base64 upload flow and also contains the more robust variant that checks Content-Type and uses `raise_for_status()`.
  - When changing Printify logic, update both `create_single_product` and `get_print_provider_variants` and keep content-type and status checks.
- Tool contracts: `schemas.py` defines expected fields (e.g. `AnswerQuestion.answer`, `search_queries`). When changes are made to tool outputs, update the `PydanticToolsParser` usage in `chains.py` accordingly.

## Integration & dependencies

- Dependencies declared in `pyproject.toml`. Notable libs: `langchain`, `langgraph`, `langchain-openai`, `langsmith`, `langchain-tavily`, `requests`, `pillow`.
- Network endpoints: OpenAI (via `openai.OpenAI`), Printify API (`api.printify.com`), Shopify admin URLs (constructed using `SHOPIFY_STORE_URL`). Tests or dry-run flags are not implemented — be conservative when running code that posts to Printify/Shopify.

## Editing guidance for AI agents

- When making changes to workflows, update the corresponding graph builder (`MessageGraph` in `main.py` or `StateGraph` in `pod_workflow.py`) so the runtime reflects the new nodes/edges.
- Preserve and follow `PodState` shape (see `pod_workflow.py`) and ensure nodes return the (possibly mutated) state dict.
- Keep HTTP calls robust: use `response.raise_for_status()` and check `Content-Type` where images are downloaded; include timeouts on requests.post/get.
- When adding tools:
  - add a Pydantic model in `schemas.py` to define the contract
  - expose the function via `StructuredTool.from_function(...)` in `tool_executor.py` (or similar)
  - wire the tool into `MessageGraph`/`ToolNode` using the tool's class name (see `AnswerQuestion` example)

## Quick checklist for change PRs

1. Confirm lint/format per `pyproject.toml` (Black/isort are declared).
2. Run the focused example (`main.py` or `pod_workflow.py`) with environment variables set; verify no unexpected network calls when tests are not intended.
3. If modifying Printify/Shopify logic, test using a dummy or staging shop credentials.
4. Update `schemas.py` when tool inputs/outputs change and ensure `chains.py` parsers match.

---

If anything in this file is unclear or you'd like extra details (e.g. example inputs for `PodState`, minimal `.env` template, or a suggested dry-run flag for Printify calls) tell me which part and I will expand or iterate.
