"""Mid-stream dropout auto-resume (connection drops after text was produced)."""

from tests.conftest import FakeModels


def test_stream_resumes_after_mid_stream_dropout(auth_api, fake_gemini):
    """1st stream: 2 chunks then 503. 2nd stream: continuation completes."""
    fake_gemini.stream_segments = [
        (["The GPU renders ", "frames using "], "STOP"),
        (["thousands of cores."], "STOP"),
    ]
    fake_gemini.stream_fail_after = [2, None]  # first call drops after 2 chunks

    response = auth_api.post("/api/v1/chat/stream", json={"message": "Explain GPUs"})
    assert response.status_code == 200
    text = "".join(line[6:] for line in response.text.split("\n") if line.startswith("data: "))
    assert text == "The GPU renders frames using thousands of cores."
    assert "[error]" not in text
    assert len(fake_gemini.calls) == 2

    # Stored history must hold the complete resumed reply.
    history = auth_api.get(
        f"/api/v1/conversations/{response.headers['X-Conversation-Id']}"
    ).json()
    assistant = [m for m in history["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"] == "The GPU renders frames using thousands of cores."


def test_stream_resume_includes_partial_as_model_context(auth_api, fake_gemini):
    fake_gemini.stream_segments = [
        (["partial teaching content"], "STOP"),
        ([" and the rest."], "STOP"),
    ]
    fake_gemini.stream_fail_after = [1, None]

    auth_api.post("/api/v1/chat/stream", json={"message": "Hi"})
    resume_contents = fake_gemini.calls[1]["contents"]
    assert resume_contents[-2].role == "model"
    assert "partial teaching content" in resume_contents[-2].parts[0].text
    assert resume_contents[-1].role == "user"
    assert "Continue" in resume_contents[-1].parts[0].text


def test_stream_dropout_budget_exhausted_keeps_partial_and_errors(auth_api, fake_gemini):
    """Three consecutive dropouts exhaust the resume budget (2) -> error event."""
    fake_gemini.stream_segments = [
        (["chunk one "], "STOP"),
        (["chunk two "], "STOP"),
        (["chunk three "], "STOP"),
        (["never reached"], "STOP"),
    ]
    fake_gemini.stream_fail_after = [1, 1, 1, None]

    response = auth_api.post("/api/v1/chat/stream", json={"message": "Hi"})
    assert response.status_code == 200
    text = "".join(line[6:] for line in response.text.split("\n") if line.startswith("data: "))
    assert "chunk one chunk two chunk three" in text
    assert "never reached" not in text
    assert "[error]" in text

    # Partial reply persisted so history stays continuous.
    history = auth_api.get(
        f"/api/v1/conversations/{response.headers['X-Conversation-Id']}"
    ).json()
    assistant = [m for m in history["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"] == "chunk one chunk two chunk three"


def test_stream_dropout_resume_does_not_consume_continuation_budget(auth_api, fake_gemini):
    """A dropout-resume + a MAX_TOKENS continuation both happen, in order."""
    fake_gemini.stream_segments = [
        (["part one"], "STOP"),        # drops mid-stream -> resume
        (["part two"], "MAX_TOKENS"),  # cap hit -> continuation
        (["part three"], "STOP"),
    ]
    fake_gemini.stream_fail_after = [1, None, None]

    response = auth_api.post("/api/v1/chat/stream", json={"message": "Hi"})
    text = "".join(line[6:] for line in response.text.split("\n") if line.startswith("data: "))
    assert text == "part onepart twopart three"
    assert "[error]" not in text
    assert len(fake_gemini.calls) == 3
