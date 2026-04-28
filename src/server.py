from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import argparse
import asyncio
import json
import uuid

from src.core.llm import ToolEvent, RouteEvent
from src.core.registry import load_domain
from src.core.events_registry import parse_slash_command, list_commands
from src.shared.config import Config

app = FastAPI(title="AndrewCLI API")

# Async task bookkeeping for /chat/async. Each entry holds the running
# asyncio.Task so DELETE can actually cancel it.
tasks: dict[str, dict] = {}


def _resolve_domain_name(requested: Optional[str]) -> str:
    return requested or Config().domain


def _new_domain(requested_domain: Optional[str] = None):
    """Create a fresh Domain per request.

    The previous implementation cached a single Domain module-wide and
    mutated its LLM memory from every request, which interleaved
    concurrent conversations (and could generate malformed tool_call /
    tool_response sequences). A fresh domain per request is the simplest
    correct fix.
    """
    name = _resolve_domain_name(requested_domain)
    return load_domain(name), name


class ChatRequest(BaseModel):
    message: str
    domain: Optional[str] = None


class ChatResponse(BaseModel):
    domain: str
    response: str
    tool_calls: list


@app.get("/events")
async def get_events():
    """List all available slash-command events."""
    from src.core.events_registry import available_events
    import inspect
    registry = available_events()
    result = []
    for name, cls in sorted(registry.items()):
        sig = inspect.signature(cls.__init__)
        params = [n for n in sig.parameters if n != "self"]
        result.append({"name": name, "args": params})
    return {"events": result, "usage": "/name [arg1] [arg2] ..."}


async def _run_slash_command(req: ChatRequest) -> ChatResponse:
    """Handle a /name [args] message: fire the event once via generate_event."""
    text = req.message.strip()
    domain_name = _resolve_domain_name(req.domain)
    if text == "/events":
        return ChatResponse(domain=domain_name, response=list_commands(), tool_calls=[])
    if text.startswith("/stop"):
        return ChatResponse(
            domain=domain_name,
            response="Event stopping is stateful and only supported in CLI/tray mode.",
            tool_calls=[],
        )
    event = parse_slash_command(text)
    if event is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown slash command. {list_commands()}",
        )
    if not event.message:
        return ChatResponse(
            domain=_resolve_domain_name(req.domain),
            response=f"Event '{event.name}' registered (no agent message).",
            tool_calls=[],
        )
    domain, domain_name = _new_domain(req.domain)
    response_text = ""
    tool_calls = []
    async for token in domain.generate_event(event.message):
        if isinstance(token, RouteEvent):
            continue
        if isinstance(token, ToolEvent):
            if token.tool_name:
                tool_calls.append({"tool": token.tool_name, "args": token.tool_args or {}})
            continue
        response_text += token
    return ChatResponse(domain=domain_name, response=response_text, tool_calls=tool_calls)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.message.startswith("/"):
        return await _run_slash_command(req)
    try:
        domain, domain_name = _new_domain(req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_text = ""
    tool_calls = []

    async for token in domain.generate(req.message):
        if isinstance(token, RouteEvent):
            # Router info is internal; don't fold it into the response body
            # (and concatenating the object with str would raise TypeError).
            continue
        if isinstance(token, ToolEvent):
            if token.tool_name:
                tool_calls.append({
                    "tool": token.tool_name,
                    "args": token.tool_args or {},
                })
            continue
        response_text += token

    return ChatResponse(
        domain=domain_name,
        response=response_text,
        tool_calls=tool_calls,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if req.message.startswith("/"):
        result = await _run_slash_command(req)
        async def _once():
            yield f"data: {json.dumps({'type': 'session', 'domain': result.domain})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': result.response})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(_once(), media_type="text/event-stream")
    try:
        domain, domain_name = _new_domain(req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'domain': domain_name})}\n\n"

        async for token in domain.generate(req.message):
            if isinstance(token, RouteEvent):
                yield f"data: {json.dumps({'type': 'route', 'tools': token.tool_names})}\n\n"
                continue
            if isinstance(token, ToolEvent):
                if token.tool_name:
                    yield f"data: {json.dumps({'type': 'tool', 'tool': token.tool_name, 'args': token.tool_args or {}})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'tool_done'})}\n\n"
                continue
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/async")
async def chat_async(req: ChatRequest):
    try:
        domain, domain_name = _new_domain(req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_id = str(uuid.uuid4())
    entry: dict = {
        "status": "processing",
        "domain": domain_name,
        "response": "",
        "tool_calls": [],
    }
    tasks[task_id] = entry

    async def _process():
        try:
            async for token in domain.generate(req.message):
                if isinstance(token, RouteEvent):
                    continue
                if isinstance(token, ToolEvent):
                    if token.tool_name:
                        entry["tool_calls"].append({
                            "tool": token.tool_name,
                            "args": token.tool_args or {},
                        })
                    continue
                entry["response"] += token
            entry["status"] = "done"
        except asyncio.CancelledError:
            entry["status"] = "cancelled"
            raise
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)

    entry["task"] = asyncio.create_task(_process())
    return {"task_id": task_id, "status": "processing"}


@app.get("/chat/status/{task_id}")
async def chat_status(task_id: str):
    entry = tasks.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # Hide the internal Task handle from the wire response.
    return {k: v for k, v in entry.items() if k != "task"}


@app.delete("/chat/status/{task_id}")
async def delete_task(task_id: str):
    entry = tasks.pop(task_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task = entry.get("task")
    if task is not None and not task.done():
        task.cancel()
    return {"status": "deleted", "task_id": task_id}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="AndrewCLI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
