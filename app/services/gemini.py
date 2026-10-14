"""Gemini chat service on the google-genai SDK (spec section 9).

Two reliability layers keep long replies complete:

1. **Auto-continuation** — thinking-model output budgets are shared between
   hidden "thinking" tokens and the visible reply. If generation stops with
   ``finish_reason=MAX_TOKENS``, the reply is continued from where it stopped
   (up to ``MAX_REPLY_CONTINUATIONS`` times) so answers never truncate
   mid-sentence.
2. **Transient-error retry + model fallback** — Google-side ``503
   UNAVAILABLE`` / high-demand spikes and local network blips (DNS failure,
   reset or timed-out connection) are retried with a short backoff. A model
   that is still failing (overloaded, retired, out of quota) is skipped in
   favour of the next model in ``GEMINI_FALLBACK_MODELS``, so one saturated
   model does not take chat down with it. Only when every model has failed does
   the request surface as a 502.
3. **Mid-stream resume** — if a *connected* stream drops mid-generation
   (network blip, upstream reset), generation transparently resumes from the
   text already produced instead of failing the reply.
"""

import itertools
import logging
import time
from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from ..config import get_settings
from ..errors import AINotConfiguredError, AIUnavailableError, ApiError

logger = logging.getLogger(__name__)

_client: genai.Client | None = None

# Wait between retries for transient upstream failures (seconds).
_RETRY_BACKOFF_SECONDS = 2.0
# Attempts for the *initial* generation call only; continuations are not retried.
_INITIAL_ATTEMPTS = 3
# How many times a *connected* stream may drop mid-generation and be resumed.
_MID_STREAM_RESUMES = 2
# Instruction appended when generation resumes from partial text.
_CONTINUE_PROMPT = "Continue exactly where you left off. Do not repeat any earlier text."


def get_client() -> genai.Client:
    """Return the cached Gemini client, creating it on first use."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise AINotConfiguredError()
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def reset_client() -> None:
    """Used by tests to swap in a fake client."""
    global _client
    _client = None


def _build_contents(history: list, user_message: str) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in history:
        role = "model" if message.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=message.content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    return contents


def _generation_config() -> types.GenerateContentConfig:
    settings = get_settings()
    return types.GenerateContentConfig(
        system_instruction=settings.system_prompt,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        # Gemini 2.5+ "thinks" before answering and those tokens count against
        # max_output_tokens — a small budget starves the visible reply (observed
        # 491 thinking vs 17 reply tokens). THINKING_BUDGET=0 disables it.
        thinking_config=types.ThinkingConfig(
            thinking_budget=settings.thinking_budget,
        ),
        # This service never declares tools, so automatic function calling (AFC)
        # has nothing to call — it only wraps every call in AFC bookkeeping and
        # logs "Direct use of AFC ... is not recommended" on each new process.
        # Disabling it is the SDK's supported way to take the plain path.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
    )


def _finish_reason(response) -> str | None:
    """Extract the finish reason of the first candidate, if any."""
    try:
        candidates = response.candidates or []
        if candidates and candidates[0].finish_reason is not None:
            return candidates[0].finish_reason.name  # e.g. "STOP", "MAX_TOKENS"
    except AttributeError:
        pass
    return None


def _is_transport_error(exc: Exception) -> bool:
    """The request never reached Google: DNS failure, refused/reset or timeout.

    Seen in practice as ``ConnectError: [Errno -3] Temporary failure in name
    resolution`` when the LAN's DNS resolver blinks mid-request. Every model is
    served from the same endpoint, so trying another model cannot help — but the
    condition is usually momentary, so a short retry often does.
    """
    return isinstance(exc, httpx.TransportError)


def _is_transient(exc: Exception) -> bool:
    """Failures worth retrying against the same model.

    Covers Google-side 5xx (503 high demand) and local network blips.
    """
    if isinstance(exc, genai_errors.ServerError):
        return True
    if _is_transport_error(exc):
        return True
    code = getattr(exc, "code", None)
    return isinstance(code, int) and 500 <= code < 600


# Upstream codes that mean "this model cannot serve the request right now":
# 404 = retired or not enabled for this key, 429 = quota exhausted (a different
# model has its own quota). Neither is fixed by retrying the same model.
_MODEL_SWITCH_CODES = frozenset({404, 429})


def _is_model_switchable(exc: Exception) -> bool:
    """True when a different model might succeed where this one failed.

    A transport error is deliberately excluded: the same endpoint serves every
    model, so walking the whole chain would only add latency to an outage the
    chain cannot fix.
    """
    if _is_transport_error(exc):
        return False
    if _is_transient(exc):
        return True
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in _MODEL_SWITCH_CODES


def _call_with_fallback(call, *, what: str) -> tuple[object, str]:
    """Run ``call(model)`` across the configured model chain.

    Each model gets ``_INITIAL_ATTEMPTS`` tries with a short backoff for
    transient failures; a model still failing after that is skipped in favour
    of the next one. Returns ``(result, model_used)``. Raises
    ``AIUnavailableError`` once every model has failed, or immediately for an
    error no other model can fix (bad request, auth failure).
    """
    settings = get_settings()
    chain = settings.model_chain
    last_exc: Exception | None = None

    for index, model in enumerate(chain):
        for attempt in range(_INITIAL_ATTEMPTS):
            try:
                result = call(model)
            except ApiError:
                raise
            except Exception as exc:
                last_exc = exc
                if not (_is_transient(exc) or _is_model_switchable(exc)):
                    logger.warning("%s failed: %s: %s", what, type(exc).__name__, exc)
                    raise AIUnavailableError() from exc
                if _is_transient(exc) and attempt < _INITIAL_ATTEMPTS - 1:
                    logger.warning(
                        "Transient %s failure (%s), retrying in %.1fs",
                        what,
                        type(exc).__name__,
                        _RETRY_BACKOFF_SECONDS,
                    )
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                if not _is_model_switchable(exc):
                    # Not a model problem (e.g. the network is down): handing the
                    # next model the same broken connection only adds delay.
                    logger.warning(
                        "%s failed after %d attempts (%s): %s",
                        what,
                        _INITIAL_ATTEMPTS,
                        type(exc).__name__,
                        exc,
                    )
                    raise AIUnavailableError() from exc
                if index < len(chain) - 1:
                    logger.warning(
                        "%s: model %s is unavailable (%s); falling back to %s.",
                        what,
                        model,
                        type(exc).__name__,
                        chain[index + 1],
                    )
                else:
                    logger.warning(
                        "%s failed on every model %s: %s: %s",
                        what,
                        chain,
                        type(exc).__name__,
                        exc,
                    )
                    raise AIUnavailableError() from exc
            else:
                if index > 0:
                    logger.info("%s served by fallback model %s.", what, model)
                return result, model

    raise AIUnavailableError() from last_exc  # pragma: no cover


# --- Non-streaming ----------------------------------------------------------


def generate_reply(history: list, user_message: str) -> tuple[str, str]:
    """Generate a complete reply, auto-continuing if the output cap is hit.

    Returns (reply_text, model_name).
    """
    settings = get_settings()
    contents = _build_contents(history, user_message)
    parts: list[str] = []

    response, model = _call_with_fallback(
        lambda candidate: get_client().models.generate_content(
            model=candidate,
            contents=contents,
            config=_generation_config(),
        ),
        what="Gemini request",
    )

    text = (response.text or "").strip()
    if not text:
        logger.warning("Gemini returned an empty response.")
        raise AIUnavailableError()
    parts.append(text)

    # Auto-continue while the model ran out of output budget.
    for _ in range(settings.max_reply_continuations):
        if _finish_reason(response) != "MAX_TOKENS":
            break
        logger.info("Reply hit the output cap; continuing from where it stopped.")
        contents = contents + [
            types.Content(role="model", parts=[types.Part(text=text)]),
            types.Content(role="user", parts=[types.Part(text=_CONTINUE_PROMPT)]),
        ]
        try:
            response = get_client().models.generate_content(
                model=model,
                contents=contents,
                config=_generation_config(),
            )
        except ApiError:
            raise
        except Exception as exc:
            # Keep what we have rather than failing the whole reply.
            logger.warning("Continuation failed (%s); returning partial reply.", type(exc).__name__)
            break
        continuation = (response.text or "").strip()
        if not continuation:
            break
        parts.append(continuation)

    return "\n".join(parts), model


# --- Streaming --------------------------------------------------------------


def _open_stream(contents: list[types.Content]):
    """Start a streamed generation, retrying and falling back across models.

    The SDK defers the HTTP request until the first chunk is pulled, so the
    first chunk is consumed eagerly here: a 503/overload on connection
    surfaces in this function and can be retried — or handed to the next model
    in the chain. The returned iterator re-yields that first chunk followed by
    the rest of the stream.
    """

    def start(model: str):
        stream = get_client().models.generate_content_stream(
            model=model,
            contents=contents,
            config=_generation_config(),
        )
        try:
            first = next(stream)  # forces the request; may raise the 503
        except StopIteration:
            # Stream connected but produced nothing at all.
            logger.warning("Gemini stream produced no content.")
            raise AIUnavailableError()
        return itertools.chain([first], stream)

    iterator, _model = _call_with_fallback(start, what="Gemini stream")
    return iterator


def stream_reply(history: list, user_message: str) -> Iterator[str]:
    """Yield reply text chunks, surviving output caps and mid-stream drops."""
    settings = get_settings()
    contents = _build_contents(history, user_message)
    produced_any = False
    continuations_left = settings.max_reply_continuations
    resumes_left = _MID_STREAM_RESUMES

    while True:
        try:
            stream = _open_stream(contents)
        except AIUnavailableError:
            # _open_stream already exhausted its connect-retries.
            if produced_any and resumes_left > 0:
                resumes_left -= 1
                logger.warning("Stream reconnect failed; resuming again (%d left).", resumes_left)
                continue
            raise

        produced = False
        stop_reason: str | None = None
        segment_text: list[str] = []
        try:
            for chunk in stream:
                text = chunk.text or ""
                if text:
                    produced = True
                    produced_any = True
                    segment_text.append(text)
                    yield text
                for candidate in chunk.candidates or []:
                    if candidate.finish_reason is not None:
                        stop_reason = candidate.finish_reason.name
        except ApiError:
            raise
        except Exception as exc:
            if _is_transient(exc) and produced_any and resumes_left > 0:
                resumes_left -= 1
                partial = "".join(segment_text).strip()
                if partial:
                    contents = contents + [
                        types.Content(role="model", parts=[types.Part(text=partial)]),
                        types.Content(role="user", parts=[types.Part(text=_CONTINUE_PROMPT)]),
                    ]
                logger.warning(
                    "Stream dropped mid-generation (%s); auto-resuming (%d left).",
                    type(exc).__name__,
                    resumes_left,
                )
                continue
            logger.warning("Gemini stream failed: %s: %s", type(exc).__name__, exc)
            raise AIUnavailableError() from exc

        produced_any = produced_any or produced
        if not produced_any:
            logger.warning("Gemini stream produced no content.")
            raise AIUnavailableError()

        if stop_reason != "MAX_TOKENS" or continuations_left == 0:
            return
        continuations_left -= 1

        # The output cap cut this segment short: continue from where it ended.
        continued = "".join(segment_text).strip()
        if not continued:
            return
        logger.info("Stream hit the output cap; auto-continuing the reply.")
        contents = contents + [
            types.Content(role="model", parts=[types.Part(text=continued)]),
            types.Content(role="user", parts=[types.Part(text=_CONTINUE_PROMPT)]),
        ]
