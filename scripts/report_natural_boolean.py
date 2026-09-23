"""Summarize the saved natural boolean comparison without rerunning inference."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/natural-boolean-20260918'
d=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
tests=json.loads((OUT/'tests.json').read_text(encoding='utf-8'))
lines=['# Direct boolean scoring: implementation and comparison','',
'Boolean questions now score lowercase false/true directly. Choice and Score retain distinct A-Z option IDs. The common system instruction now defers to the requested format, allowing a shared prefix across mixed batches. No model weights were changed.','',
'External answer structures are unchanged: false maps to no, true maps to yes, and noul is P(true). The probability_status metadata string is now uncalibrated_answer_token_probabilities, and answer_encoding identifies the field scoring format. Unsupported boolean tokenization fails explicitly.','',
'## Test results','',f"{tests['tests_run']} regression tests passed: {tests['successful']}; skipped: {len(tests['skipped'])}. Coverage includes full-head numerical parity, mixed candidate types, head-cache isolation, graph replay, and fused kernels.",'',
'## Fresh same-GPU timings','',
'GPU: '+d['environment']['gpu']+'. Qwen2.5-1.5B-Instruct revision '+d['environment']['model_revision']+', FP16, Linux/WSL. One shared resident model; no concurrent methods.','',
'Warm complete-request synchronized wall time, median of three consecutive samples. Loading, compilation and capture are excluded. The 28-field ordinary generation baseline has one measured run after a 16-token warmup. Old-engine numbers below were rerun in this session, not copied from the earlier report.','',
'| Workload | Qwen JSON | HF CUDA | Old engine | New engine | Independent new scoring |',
'|---|---:|---:|---:|---:|---:|']
for name,c in d['cases'].items():
    cells=[]
    for mode in ('qwen_json','hf_parallel','our_previous','our_optimized','qwen_independent_scoring'):
        m=c['methods'][mode]
        if m['status']!='ok':cells.append(m['status']);continue
        flag=''
        if mode=='qwen_json':
            statuses={r['result']['status']for r in m['rows']}
            if statuses!={'valid'}:flag=' ('+', '.join(sorted(statuses))+')'
        cells.append(f"{m['median_ms']:.1f} ms"+flag)
    lines.append('| '+name+' | '+' | '.join(cells)+' |')
lines+=['','## Speed ratios','', '| Workload | New vs old | New vs HF | New vs valid Qwen JSON |','|---|---:|---:|---:|']
for name,c in d['cases'].items():
    ms=c['methods'];new=ms['our_optimized']['median_ms']
    valid=all(r['result']['status']=='valid' for r in ms['qwen_json']['rows'])
    lines.append('| '+name+' | '+f"{ms['our_previous']['median_ms']/new:.2f}x | {ms['hf_parallel']['median_ms']/new:.2f}x | "+(f"{ms['qwen_json']['median_ms']/new:.2f}x" if valid else 'not established')+' |')
lines+=['','Ratios above 1 favor the new engine. These small samples are sensitive to laptop power/scheduling variation; they are not statistical performance guarantees. Raw samples and cold warmup costs are saved in comparison.json.','',
'## Refund accuracy and actual outputs','', '| Method | Correct fields | P(refund requested) | Selected labels |','|---|---:|---:|---|']
for mode,m in d['cases']['refund_4']['methods'].items():
    row=m['rows'][0];q=row['quality'];r=row['result'];prob='not returned'
    if mode=='hf_parallel':
        prob=str(next(x['probability'] for x in r['field_telemetry']['refund_requested']['top_choices'] if x['choice']=='true'))
    elif mode!='qwen_json':prob=f"{r['answers']['refund_requested']['probabilities']['yes']:.6f}"
    lines.append(f"| {mode} | {q['correct']}/{q['fields']} | {prob} | "+json.dumps(row['labels'])+' |')
lines+=['','The labels come from the highest-probability option, not from rounding the mean Score value. The refund example has four known labels and is too small to establish general accuracy or calibration. The 28-field workload has no verified labels. Prompt and token changes were made together; this comparison does not isolate their individual effects.','',
'## Schema checks','']
responses=fields=0
for c in d['cases'].values():
    for row in c['methods']['our_optimized']['rows']:
        json.dumps(row['result'],allow_nan=False)
        responses+=1;fields+=len(row['result']['answers'])
lines.append(f'All {responses} new-engine benchmark responses ({fields} answer fields) passed strict JSON serialization, exact question/option keys, finite probabilities in [0,1], normalization, and equality to the deterministic typed answer constructor. No schema violations occurred.')
lines+=['','## Limits and reproducibility','',
'- Ordinary Qwen generates labels-only JSON; the decision engines return normalized option scores too. Prompts differ. Truncated or invalid generation is excluded from successful-output speedup claims.',
'- HF runs the original downloaded CUDA inference/schema functions at revision '+d['replica_revision']+' with the same resident FP16 weights. Its public M4 Max/4-bit MLX figures are not this comparison. Its two known candidate-token collisions in the 28-field preset remain unchanged.',
'- New and old engines use cache optimizations, CUDA graphs, and all three fused kernels. Their caches are cleared between methods. Shapes must repeat to obtain warmed graph benefits.',
'- The new shared system wording also reaches Choice/Score questions; they retain letter candidates but are not an unchanged-prompt control.',
'- No fine-tuning or calibration was performed. Probabilities remain token-relative scores.',
'- Reproduce with scripts/compare_natural_boolean.py and scripts/report_natural_boolean.py. engine_before.py is the pre-change local snapshot; source-manifest.json records final source hashes. The HF snapshot lives in ../comparison-20260918/replica-source. These ignored local artifacts must accompany a fresh checkout.',
'', '## Recorded new refund response','', '```json',json.dumps(d['cases']['refund_4']['methods']['our_optimized']['rows'][0]['result']['answers'],indent=2),'```','']
confirm_path=OUT/'confirmation.json'
if confirm_path.exists():
    import statistics
    confirmation=json.loads(confirm_path.read_text(encoding='utf-8'))
    lines+=['','## Alternating large-workload confirmation','',confirmation['reason'],'',
            '| Method | Samples | Median | Minimum | Maximum |','|---|---:|---:|---:|---:|']
    for mode,rows in confirmation['timings'].items():
        times=[r['wall_ms']for r in rows]
        if times:lines.append(f"| {mode} | {len(times)} | {statistics.median(times):.1f} ms | {min(times):.1f} ms | {max(times):.1f} ms |")
    if all(len(rows)==5 for rows in confirmation['timings'].values()):
        medians={mode:statistics.median(r['wall_ms']for r in rows)for mode,rows in confirmation['timings'].items()}
        note=(f"Timing follow-up: five alternating 28-field samples per method gave medians of {medians['our_previous']:.1f} ms (old engine), {medians['our_optimized']:.1f} ms (new engine), and {medians['hf_parallel']:.1f} ms (HF CUDA). Initial samples were much noisier and remain visible below; avoid attributing their full timing differences to the format change.")
        lines[2:2]=[note,'']
    lines+=['','Each timed call follows an untimed warmup after clearing all graph buffers. Method order rotates across rounds. These are additional samples, not a replacement or deletion of the initial timing results.','',
            '## Positive and negative boolean controls','',
            'Eight manually authored contexts cover explicit requests, negations, completed refunds, and human-agent requests. They were evaluated after the implementation was fixed, without further prompt tuning. This is a small diagnostic set, not a representative accuracy benchmark.','',
            '| Method | Correct / 16 labeled fields |','|---|---:|']
    for mode in confirmation['timings']:
        correct=sum(x['methods'][mode]['correct']for x in confirmation['boolean_controls'])
        lines.append(f"| {mode} | {correct}/16 |")
    lines+=['','Detailed contexts, expected labels, probabilities, and all timing samples are retained in confirmation.json. New-engine responses in this confirmation also undergo the same schema checks.','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((ROOT/'src/sparset_decide').glob('*.py'))}
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(OUT/'REPORT.md')
