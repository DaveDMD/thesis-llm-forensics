"""Real local LLM backend based on HuggingFace ``transformers``.

This module implements :class:`forensic.backends.ModelBackend` against a locally
hosted causal language model. Heavy dependencies (``torch``, ``transformers``)
are imported lazily inside :meth:`TransformersBackend.load`, so importing this
module does not require the ML stack to be installed; only instantiating *and
loading* a backend does.

Model selection is intentionally left to configuration: ``model_id`` is whatever
local or HuggingFace identifier the experiment uses. The concrete target model
for the application phase is a configuration choice; this backend
is model-agnostic by design and is the same code path regardless of the choice,
which also supports *generalising* across open models.

Forensic note
-------------
The backend contributes only the generated text plus optional telemetry
(token counts, first-token latency, finish reason). Hashing, pseudonymisation,
latency wall-clock and the append-only chain are handled by the server and the
forensic logger; nothing forensic is duplicated here.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from .backends import CompletionResult, RetrievedHit, SequenceScore

_DEFAULT_RAG_TEMPLATE = (
    "Answer the question using only the retrieved context below. "
    "If the context is insufficient, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {prompt}\n\n"
    "Answer:"
)


def build_rag_prompt(
    prompt: str,
    hits: Sequence[RetrievedHit],
    *,
    template: str = _DEFAULT_RAG_TEMPLATE,
) -> str:
    """Assemble the augmented generation prompt from retrieved chunk contents.

    Hits without raw ``content`` (e.g. produced by a content-free retriever) are
    represented by their ``chunk_id`` placeholder, so the function never fails
    on a missing chunk body. This is a thesis-side prompt-assembly convention,
    documented as such, not a standard template.
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        body = hit.content if hit.content not in (None, "") else f"<no-content:{hit.chunk_id}>"
        blocks.append(f"[{index}] {body}")
    context = "\n\n".join(blocks) if blocks else "<no-context>"
    return template.format(context=context, prompt=prompt)


def _map_sampling_params(sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    """Map the request sampling parameters to ``transformers.generate`` kwargs.

    ``temperature == 0`` (or absent) selects deterministic greedy decoding,
    matching the temperature=0.0 convention used by the traffic simulators.
    """
    temperature = float(sampling_params.get("temperature", 0.0))
    kwargs: dict[str, Any] = {}
    if temperature <= 0.0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        if "top_p" in sampling_params:
            kwargs["top_p"] = float(sampling_params["top_p"])
        if "top_k" in sampling_params:
            kwargs["top_k"] = int(sampling_params["top_k"])
    if "repetition_penalty" in sampling_params:
        kwargs["repetition_penalty"] = float(sampling_params["repetition_penalty"])
    return kwargs


_VALID_QUANTIZATION = (None, "4bit", "8bit")


def build_bnb_config_kwargs(
    quantization: str | None,
    *,
    compute_dtype: str = "float16",
    double_quant: bool = True,
    quant_type: str = "nf4",
) -> dict[str, Any] | None:
    """Build the keyword arguments for ``transformers.BitsAndBytesConfig``.

    Pure function (returns a plain dict), so the quantization policy can be
    unit-tested without importing ``bitsandbytes`` or loading any weights.
    Returns ``None`` when ``quantization`` is ``None`` (full-precision load).

    The 4-bit defaults follow the QLoRA/NF4 recipe (NF4 quant type, double
    quantization, fp16 compute) used to fit a 7B model into ~4-5 GB of VRAM on
    consumer hardware — the configuration adopted for the thesis application
    model (Mistral-7B-Instruct on an 8.59 GB GPU). Declared as a hardware-driven
    choice; the marginal behavioural effect of 4-bit is a documented limitation.
    """
    if quantization not in _VALID_QUANTIZATION:
        raise ValueError(
            f"quantization must be one of {_VALID_QUANTIZATION}, got {quantization!r}"
        )
    if quantization is None:
        return None
    if quantization == "8bit":
        return {"load_in_8bit": True}
    # 4bit
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_use_double_quant": bool(double_quant),
        "bnb_4bit_compute_dtype": compute_dtype,  # resolved to a torch dtype at load
    }


class TransformersBackend:
    """A :class:`ModelBackend` backed by a local ``transformers`` causal LM."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str | None = None,
        model_hash: str | None = None,
        device: str | None = None,
        torch_dtype: str = "float16",
        trust_remote_code: bool = False,
        rag_prompt_template: str = _DEFAULT_RAG_TEMPLATE,
        quantization: str | None = None,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        # If not supplied, a lightweight identity tag is derived at load time.
        self.model_hash = model_hash
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.rag_prompt_template = rag_prompt_template
        # Validate eagerly so a bad value fails at construction, not at load().
        if quantization not in _VALID_QUANTIZATION:
            raise ValueError(
                f"quantization must be one of {_VALID_QUANTIZATION}, got {quantization!r}"
            )
        self.quantization = quantization
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self._tokenizer = None
        self._model = None
        self._torch = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> "TransformersBackend":
        """Lazily import the ML stack and load tokenizer + model weights."""
        import torch  # noqa: WPS433 (intentional lazy import)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        dtype = getattr(torch, self.torch_dtype, None)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code,
        )

        bnb_kwargs = build_bnb_config_kwargs(
            self.quantization,
            compute_dtype=self.bnb_4bit_compute_dtype,
            double_quant=self.bnb_4bit_use_double_quant,
            quant_type=self.bnb_4bit_quant_type,
        )

        if bnb_kwargs is not None:
            from transformers import BitsAndBytesConfig  # lazy

            # Resolve the compute dtype string to an actual torch dtype.
            if "bnb_4bit_compute_dtype" in bnb_kwargs:
                bnb_kwargs["bnb_4bit_compute_dtype"] = getattr(
                    torch, bnb_kwargs["bnb_4bit_compute_dtype"], torch.float16
                )
            quant_config = BitsAndBytesConfig(**bnb_kwargs)
            # With bitsandbytes, weight placement is handled by device_map; do
            # NOT call .to(device) on a quantized model (it raises).
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
            ).to(self.device)
        self._model.eval()

        if self.model_hash is None:
            # Lightweight identity tag for the manifest. A full weights digest
            # can be supplied explicitly and is anchored separately via the
            # manifest's model_artifacts hashing.
            commit = getattr(self._model.config, "_commit_hash", None)
            self.model_hash = f"hf:{self.model_id}@{commit or self.model_revision or 'local'}"
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("TransformersBackend.load() must be called before generation")

    def score_sequence(self, text: str, *, max_length: int = 1024) -> SequenceScore:
        """Teacher-forced per-token scoring of *text* (single forward pass).

        Returns the log-prob of each actual token plus the mean/std of the
        log-prob distribution at each position (for Min-K%++). White-box: this is
        what a score-based MIA reads; the query traffic is logged separately by
        the server, this only computes the membership signal.
        """
        self._require_loaded()
        torch = self._torch
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        ids = enc["input_ids"]
        with torch.no_grad():
            logits = self._model(**enc).logits  # [1, T, V]
        if ids.shape[1] < 2:
            return SequenceScore(text=text, n_tokens=0, token_logprobs=[])
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)  # [T-1, V]
        targets = ids[0, 1:]                                    # [T-1]
        idx = torch.arange(targets.shape[0], device=self.device)
        tok_lp = log_probs[idx, targets]                       # [T-1]
        mu = log_probs.mean(dim=-1)
        sd = log_probs.std(dim=-1)
        return SequenceScore(
            text=text,
            n_tokens=int(targets.shape[0]),
            token_logprobs=tok_lp.detach().cpu().tolist(),
            token_logprob_mean=mu.detach().cpu().tolist(),
            token_logprob_std=sd.detach().cpu().tolist(),
        )

    # ── generation ────────────────────────────────────────────────────────

    def _apply_system_prompt(self, user_text: str, system_prompt: str | None) -> str:
        """Render the input text, injecting the level-1 system prompt if present.

        When a chat template is available, the system prompt is passed as a
        ``system`` turn (falling back to merging it into the user turn for
        templates that reject a system role, e.g. some Mistral revisions). With no
        chat template, the system prompt is prepended as plain text. When
        ``system_prompt`` is ``None`` the behaviour is unchanged.
        """
        if not system_prompt:
            return user_text
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{system_prompt}\n\n{user_text}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return f"{system_prompt}\n\n{user_text}"

    def _generate(
        self,
        prompt_text: str,
        *,
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        self._require_loaded()
        torch = self._torch
        gen_kwargs = _map_sampling_params(sampling_params)

        prompt_text = self._apply_system_prompt(prompt_text, system_prompt)
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])

        start = time.perf_counter_ns()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                output_scores=expose_logprobs,
                return_dict_in_generate=expose_logprobs,
                **gen_kwargs,
            )
        first_token_ms = max(0, int((time.perf_counter_ns() - start) / 1_000_000))

        sequences = output.sequences if expose_logprobs else output
        generated_ids = sequences[0][prompt_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_token_count = int(generated_ids.shape[0])
        finish_reason = "length" if response_token_count >= max_tokens else "stop"

        token_logprobs = None
        top_k_first_token = None
        if expose_logprobs and getattr(output, "scores", None):
            token_logprobs, top_k_first_token = self._extract_logprobs(
                output.scores, generated_ids
            )

        return CompletionResult(
            text=text,
            response_token_count=response_token_count,
            latency_first_token_ms=first_token_ms,
            finish_reason=finish_reason,
            token_logprobs=token_logprobs,
            top_k_first_token=top_k_first_token,
        )

    def _extract_logprobs(self, scores, generated_ids, *, top_k: int = 3):
        """Extract per-token logprobs and first-token top-k from generate scores.

        ``scores`` is the tuple of per-step logits returned by
        ``generate(output_scores=True)``. We convert to log-softmax and read the
        logprob of each actually-generated token; for the first step we also
        return the top-k candidate ``{"token", "logprob"}`` pairs. The forensic
        logger hashes the candidate tokens, so raw high-probability tokens never
        reach the stream.
        """
        torch = self._torch
        token_logprobs: list[float] = []
        top_k_first: list[dict[str, Any]] | None = None
        for step, logits in enumerate(scores):
            logprob_row = torch.log_softmax(logits[0], dim=-1)
            if step < generated_ids.shape[0]:
                tok_id = int(generated_ids[step])
                token_logprobs.append(round(float(logprob_row[tok_id]), 6))
            if step == 0:
                vals, idx = torch.topk(logprob_row, k=min(top_k, logprob_row.shape[-1]))
                top_k_first = [
                    {
                        "token": self._tokenizer.decode([int(i)]),
                        "logprob": round(float(v), 6),
                    }
                    for v, i in zip(vals, idx)
                ]
        return token_logprobs, top_k_first

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        return self._generate(
            prompt,
            max_tokens=max_tokens,
            sampling_params=sampling_params,
            expose_logprobs=expose_logprobs,
            system_prompt=system_prompt,
        )

    def complete_with_context(
        self,
        prompt: str,
        *,
        retrieval_query: str,
        hits: Sequence[RetrievedHit],
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        augmented = build_rag_prompt(prompt, hits, template=self.rag_prompt_template)
        return self._generate(
            augmented,
            max_tokens=max_tokens,
            sampling_params=sampling_params,
            expose_logprobs=expose_logprobs,
            system_prompt=system_prompt,
        )


__all__ = ["TransformersBackend", "build_rag_prompt", "build_bnb_config_kwargs"]
