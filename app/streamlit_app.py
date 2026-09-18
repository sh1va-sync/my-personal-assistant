from __future__ import annotations

import os
import json
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
MAX_HISTORY_MESSAGES = 20


def _api_url() -> str:
    configured = st.session_state.get("api_url", DEFAULT_API_URL)
    return configured.rstrip("/")


def _new_conversation() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []


def _create_session() -> str:
    response = httpx.post(f"{_api_url()}/api/session", timeout=10.0)
    response.raise_for_status()
    conversation_id = response.json().get("conversation_id")
    if not conversation_id:
        raise RuntimeError("The API returned no conversation ID.")
    return str(conversation_id)


def _send_message(message: str) -> dict[str, Any]:
    prior_messages = st.session_state.messages
    if prior_messages and prior_messages[-1]["role"] == "user" and prior_messages[-1]["content"] == message:
        prior_messages = prior_messages[:-1]
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in prior_messages[-MAX_HISTORY_MESSAGES:]
    ]
    payload = {
        "conversation_id": st.session_state.conversation_id,
        "message": message,
        "history": history,
    }
    response = httpx.post(f"{_api_url()}/api/chat", json=payload, timeout=90.0)
    if response.is_error:
        detail: str
        try:
            body = response.json()
            detail = str(body.get("detail") or body.get("message") or response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}")
    data = response.json()
    if not data.get("message"):
        raise RuntimeError("The API returned an empty assistant message.")
    return data


def _send_stream(message: str) -> str:
    prior_messages = st.session_state.messages
    if prior_messages and prior_messages[-1]["role"] == "user" and prior_messages[-1]["content"] == message:
        prior_messages = prior_messages[:-1]
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in prior_messages[-MAX_HISTORY_MESSAGES:]
    ]
    payload = {
        "conversation_id": st.session_state.conversation_id,
        "message": message,
        "history": history,
    }
    parts: list[str] = []
    with httpx.stream(
        "POST",
        f"{_api_url()}/api/chat/stream",
        json=payload,
        timeout=90.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                parts.append(str(event["text"]))
            elif event["type"] == "error":
                raise RuntimeError(str(event["message"]))
    return "".join(parts)


def _check_health() -> tuple[bool, str]:
    try:
        response = httpx.get(f"{_api_url()}/api/health", timeout=5.0)
        response.raise_for_status()
        return True, "API connected"
    except httpx.HTTPError as exc:
        return False, f"API unavailable: {exc}"


st.set_page_config(
    page_title="Sync",
    page_icon="🤖",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", DEFAULT_API_URL)

st.title("Sync")
st.caption("Shiva's AI assistant — ask about his work, life, or portfolio.")

with st.sidebar:
    st.header("Settings")
    st.session_state.api_url = st.text_input(
        "Backend API URL",
        value=st.session_state.api_url,
        help="The FastAPI server address, without a trailing slash.",
    ).rstrip("/")

    connected, status = _check_health()
    if connected:
        st.success(status)
    else:
        st.error(status)

    if st.button("New conversation", use_container_width=True):
        _new_conversation()
        st.rerun()

    if st.session_state.conversation_id:
        st.caption(f"Conversation: `{st.session_state.conversation_id}`")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            st.caption("Sources: " + ", ".join(item["sources"]))

if prompt := st.chat_input("Message Sync..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = _send_message(prompt)
                st.session_state.conversation_id = result["conversation_id"]
                answer = result["message"]
                st.markdown(answer)
                if result.get("sources"):
                    st.caption("Sources: " + ", ".join(result["sources"]))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": result.get("sources", []),
                    }
                )
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                error = f"Unable to reach the AI agent: {exc}"
                st.error(error)
                st.session_state.messages.pop()
