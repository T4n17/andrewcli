from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import argparse
import asyncio
import importlib
import json
import uuid

from src.shared.config import Config
from src.core.llm import ToolEvent

app = FastAPI(title="AndrewCLI API")

tasks: dict = {}

_domain = None
_domain_name = None


def _load_domain(domain_name: str):
    """Load a domain by name."""
    try:
        module = importlib.import_module(f"src.domains.{domain_name}")
        class_name = f"{domain_name.capitalize()}Domain"
        domain_class = getattr(module, class_name)
        return domain_class()
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Could not load domain '{domain_name}': {e}")


def _get_domain(requested_domain: Optional[str] = None):
    global _domain, _domain_name
    if _domain is None:
        config = Config()
        _domain_name = requested_domain or config.domain
        _domain = _load_domain(_domain_name)
    elif requested_domain and requested_domain != _domain_name:
        _domain = _load_domain(requested_domain)
        _domain_name = requested_domain
    return _domain, _domain_name


class ChatRequest(BaseModel):
    message: str
    domain: Optional[str] = None


class ChatResponse(BaseModel):
    domain: str
    response: str
    tool_calls: list


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        domain, domain_name = _get_domain(req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_text = ""
    tool_calls = []

    async for token in domain.generate(req.message):
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
    try:
        domain, domain_name = _get_domain(req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'domain': domain_name})}\n\n"

        async for token in domain.generate(req.message):
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
        domain, domain_name = _get_domain(req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "domain": domain_name,
        "response": "",
        "tool_calls": [],
    }

    async def _process():
        try:
            async for token in domain.generate(req.message):
                if isinstance(token, ToolEvent):
                    if token.tool_name:
                        tasks[task_id]["tool_calls"].append({
                            "tool": token.tool_name,
                            "args": token.tool_args or {},
                        })
                    continue
                tasks[task_id]["response"] += token
            tasks[task_id]["status"] = "done"
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    asyncio.create_task(_process())

    return {"task_id": task_id, "status": "processing"}


@app.get("/chat/status/{task_id}")
async def chat_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.delete("/chat/status/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    del tasks[task_id]
    return {"status": "deleted", "task_id": task_id}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="AndrewCLI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
