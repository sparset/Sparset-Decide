"""Literal label likelihoods; no prior correction or length normalization."""
import time
import torch
import torch.nn.functional as F


def candidate_tokens(engine, question):
    if question.kind == "noul":
        labels = ["false", "true"]
    elif engine.answer_encoding == "letters":
        labels = list(engine.symbols[:len(question.options)])
    else:
        labels = [label for label, _ in question.options]
    rows = [engine.tokenizer.encode(x, add_special_tokens=False) for x in labels]
    special = set(getattr(engine.tokenizer, "all_special_ids", []))
    if any(not row or any(not isinstance(t, int) or not 0 <= t < engine.head.out_features or t in special for t in row) for row in rows):
        raise ValueError("Labels must encode to nonempty ordinary vocabulary tokens")
    if len({tuple(row) for row in rows}) != len(rows):
        raise ValueError("Different labels must have distinct complete token sequences")
    # Termination disambiguates a label that is itself a prefix of another.
    if any(len(a) < len(b) and b[:len(a)] == a for a in rows for b in rows):
        eos = engine.tokenizer.eos_token_id
        if not isinstance(eos, int) or not 0 <= eos < engine.head.out_features:
            raise ValueError("Overlapping label prefixes require an EOS token")
        rows = [row + [eos] for row in rows]
    if any(len(row) > engine.max_label_tokens for row in rows):
        raise ValueError("Label exceeds max_label_tokens (including required termination)")
    return rows


def project_single_tokens(engine, hidden, token_rows):
    key = tuple(dict.fromkeys(t for row in token_rows for t in row))
    if engine.efficient_cache and key in engine._head_rows:
        weights, bias = engine._head_rows[key]
        engine._head_rows.move_to_end(key)
    else:
        ids = torch.tensor(key, device=engine.device)
        weights = engine.head.weight.index_select(0, ids).float()
        bias = None if engine.head.bias is None else engine.head.bias.index_select(0, ids).float()
        if engine.efficient_cache:
            engine._head_rows[key] = (weights, bias)
            while len(engine._head_rows) > 16:
                engine._head_rows.popitem(last=False)
    logits = F.linear(hidden.float(), weights, bias) / engine.temperature
    width = max(map(len, token_rows))
    lookup = {t: i for i, t in enumerate(key)}
    indices = [[lookup[t] for t in row] + [0] * (width-len(row)) for row in token_rows]
    logits = logits.gather(1, torch.tensor(indices, device=engine.device))
    valid = torch.arange(width, device=engine.device)[None, :] < torch.tensor(list(map(len, token_rows)), device=engine.device)[:, None]
    return logits.masked_fill(~valid, -torch.inf).softmax(-1)


def sequence_scores(engine, prompt, candidates, mode):
    """Teacher-force known candidate suffixes, reusing the prompt KV cache."""
    hidden, cache = engine._hidden([prompt], use_cache=True)
    first = (engine.head(hidden).float() / engine.temperature).log_softmax(-1)[0]
    scores = first[torch.tensor([x[0] for x in candidates], device=engine.device)].clone()
    pending = [i for i, row in enumerate(candidates) if len(row) > 1]
    batch_size = engine.branch_batch_size if mode == "parallel" else 1
    calls = 1
    projected = 1
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start+batch_size]
        rows = [candidates[i][:-1] for i in indices]
        lengths = list(map(len, rows)); width = max(lengths)
        pad = engine.tokenizer.pad_token_id
        if pad is None: pad = engine.tokenizer.eos_token_id
        ids = torch.tensor([r+[pad]*(width-len(r)) for r in rows], device=engine.device)
        positions = torch.arange(width, device=engine.device)
        valid = positions[None, :] < torch.tensor(lengths, device=engine.device)[:, None]
        mask = torch.cat((torch.ones((len(rows), len(prompt)), dtype=torch.bool, device=engine.device), valid), 1)
        branch_cache = engine._fork_cache(cache)
        branch_cache.batch_repeat_interleave(len(rows))
        output = engine.backbone(input_ids=ids, attention_mask=mask,
            position_ids=(positions+len(prompt))[None, :].expand(len(rows), -1),
            past_key_values=branch_cache, use_cache=True, return_dict=True)
        suffix_hidden = output.last_hidden_state[valid]
        targets = torch.tensor([t for i in indices for t in candidates[i][1:]], device=engine.device)
        # Limit temporary full-vocabulary logits, especially on small GPUs.
        parts = []
        for offset in range(0, len(targets), 8):
            lp = (engine.head(suffix_hidden[offset:offset+8]).float() / engine.temperature).log_softmax(-1)
            parts.append(lp.gather(1, targets[offset:offset+8, None]).squeeze(1))
        logp = torch.cat(parts); offset = 0
        for i, length in zip(indices, lengths):
            scores[i] += logp[offset:offset+length].sum(); offset += length
        calls += 1; projected += len(targets)
        del branch_cache, output, suffix_hidden, logp, parts
    return scores.softmax(-1).cpu().tolist(), calls, projected


def decide_mixed(engine, context, questions, sequences, candidates, mode):
    engine._sync(); started = time.perf_counter()
    items = list(questions.items())
    single = {key: q for (key, q), cs in zip(items, candidates) if all(len(x)==1 for x in cs)}
    result = engine._decide_raw(context, single, mode=mode) if single else None
    answers = {} if result is None else dict(result["answers"])
    calls = 0 if result is None else result["metadata"]["forward_calls"]
    projected = 0
    for (key, question), prompt, cs in zip(items, sequences, candidates):
        if key in single: continue
        vector, n, positions = sequence_scores(engine, prompt, cs, mode)
        answers[key] = question.answer(vector)
        calls += n; projected += positions
    engine._sync(); elapsed = (time.perf_counter()-started)*1000
    metadata = {
        "mode": mode, "device": str(engine.device), "dtype": str(engine.head.weight.dtype),
        "scoring_dtype": "native output-head matmul; torch.float32 log_softmax and accumulation",
        "attention": getattr(engine.model.config, "_attn_implementation", None),
        "model_revision": getattr(engine.model.config, "_commit_hash", None),
        "prompt_format": engine.prompt_format,
        "question_input_tokens": list(map(len, sequences)), "branch_batch_size": engine.branch_batch_size if mode=="parallel" else 1,
        "forward_calls": calls, "generated_tokens": 0, "temperature": engine.temperature,
        "answer_encoding": {key: "false_true" if q.kind=="noul" else "literal_label_tokens" for key,q in items},
        "answer_token_lengths": {key:list(map(len, cs)) for (key,_),cs in zip(items,candidates)},
        "label_scoring": "complete_sequence_likelihood_without_length_normalization",
        "full_vocabulary_projection_positions": projected,
        "cache_storage": "per_question_prompt_reused_across_label_continuations",
        "single_token_fields": list(single),
        "optimizations": {"efficient_cache":engine.efficient_cache,"cuda_graphs":engine.cuda_graphs,
            "fused_kernels":list(engine.fused_kernels),"multi_token_suffix_cuda_graphs":False,
            "multi_token_suffix_shared_attention":False,"graph_stats":dict(engine._graphs.stats)},
        "branch_and_scoring_ms": elapsed, "total_ms": elapsed,
    }
    if result is not None: metadata["single_token_group_metadata"] = result["metadata"]
    return {"model":engine.model_id,"answers":{key:answers[key] for key in questions},
        "probability_status":"uncalibrated_candidate_sequence_probabilities","metadata":metadata}
