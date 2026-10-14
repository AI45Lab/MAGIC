"""OpenAI-compatible requests shared by generation and evaluation."""

import os

import httpx


async def request(endpoint, route, payload):
    key = os.environ[endpoint["api_key_env"]] if endpoint.get("api_key_env") else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=endpoint.get("timeout", 300)) as client:
        response = await client.post(endpoint["base_url"].rstrip("/") + route,
                                     headers=headers, json={"model": endpoint["model"], **payload})
        response.raise_for_status()
        return response.json()
