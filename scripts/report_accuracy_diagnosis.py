import json,collections

from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'

def main():

 s=json.loads((OUT/'implementation-summary.json').read_text());q=s['quality'];f=s['fresh'];changes=json.loads((OUT/'changed-decisions.json').read_text())

 lines=['# Accuracy diagnosis and optional mitigation','',

 'The original optimized engine scored 208/250 (83.2%) versus 187/250 (74.8%) for the pinned HF replica. The engine was worse specifically on routing and boolean decisions, not overall. Both produced 250/250 valid answer schemas. Original results remain untouched in `../formal-benchmark-20260918/quality.jsonl`.','',

 '## What caused the failures','',

 'The examined failures are semantic predictions made by the same Qwen checkpoint, not JSON/indexing mistakes. Twelve selected failed/control inputs were evaluated with plain execution, CUDA graphs, custom kernels, and both. None changed its chosen answer between these paths. Raw full-vocabulary scores also favored incorrect answers. This rules out those optimizations as the cause of these examined errors; it does not prove every possible execution path bug-free.','',

 'A strong answer preference existed before meaningful context was supplied. For the spelling-request rubric, the neutral context `N/A` assigned coding 95.46%, fast 3.02%, and reasoning 1.52%. The actual spelling request assigned coding 73.12% before correction. Subtracting the neutral log scores made fast the winner at 88.05%. These are uncalibrated relative scores, not measured real-world probabilities.','',

 'The engines use different internal prompts and answer encodings. Sharing weights therefore does not imply identical predictions. Some raw Qwen responses preferred an answer word even when our interface requested a letter. Multiple prompt/encoding alternatives were tested, but did not consistently improve all task types. The earlier narrow readable-prompt test was insufficient to establish reliability.','',

 '## Implemented mitigation','',

 '`--answer-bias contextual` or `DecisionEngine(..., answer_bias="contextual")` enables neutral-context correction. It subtracts log p(answer | N/A, rubric) from log p(answer | real context, rubric), then normalizes. It uses the same frozen weights; no training, labels, workflow-specific keyword rules, or external model are involved. This is answer-bias adjustment, not empirical probability calibration.','',

 'The feature is opt-in; default `none` preserves existing results. It fixes some failures and creates others, so enabling it globally is not justified for every workflow. Workflow JSON and input/output answer structures remain unchanged. The probability-status string explicitly identifies corrected, uncalibrated scores.','',

 'Neutral vectors live in a bounded CPU cache of 128 rubric/configuration keys. Changing only context reuses the vector; new rubrics require additional inference. Duplicate rubrics in a request share one evaluation. Clearing optimization caches also invalidates these vectors. No previous user context or model answer is reused.','',

 '## Results','',

 '| Evaluation | Original engine | Corrected engine | HF replica |','|---|---:|---:|---:|']

 lines += [f'| Original 250 cases | {q["ours"]["correct"]}/250 | {q["corrected"]["correct"]}/250 | {q["hf"]["correct"]}/250 |',f'| Fresh 80 cases | {f["original"]["correct"]}/80 | {f["production_corrected"]["correct"]}/80 | {f["hf"]["correct"]}/80 |','',

 '| Original benchmark group | Original engine | Corrected engine | HF replica |','|---|---:|---:|---:|']

 for g in ['banking','routing','boolean','scoring']:

  v=[s['groups'][m][g] for m in ['ours','corrected','hf']];lines.append('| '+g+' | '+' | '.join(f'{x["correct"]}/{x["n"]}' for x in v)+' |')

 lines += ['',f'Original-suite Brier score (lower is better): original {q["ours"]["brier"]:.4f}, corrected {q["corrected"]["brier"]:.4f}, HF {q["hf"]["brier"]:.4f}. This uses the sum-form multiclass score, range 0–2.',

 f'Compared with the original optimized run, correction fixed {s["changes"]["fixed"]} cases and regressed {s["changes"]["regressed"]}; {s["changes"]["remaining"]} remain incorrect. All 345 production verification outputs passed the probability/choice schema validator. This is observed validity, not a promise of zero runtime errors under arbitrary hardware or unsupported inputs.','',

 'The fresh suite improved from 64/80 to 68/80 on the plain path; one boolean answer worsened. The production result above uses CUDA graphs and the same three custom fusions as the original optimized run. The new document-operation workflow stayed 20/20 before and after plain correction.','',

 '## Latency cost on this laptop','',

 'Same Qwen2.5-1.5B-Instruct FP16 checkpoint, RTX 3050 Laptop 4 GB, Ubuntu/WSL, torch 2.6.0+cu124, transformers 5.5.4. Custom RMSNorm/SwiGLU/RoPE fusions, CUDA graphs, automatic shared attention and length grouping enabled. Five repeated warmed requests per cell; sequential method blocks, so small differences can be measurement/order noise. Quality-run times include cold graph capture and must not be presented as warmed latency.','',

 '| Workload | Original warm median ms | Corrected warm median ms | Original first shape ms | Corrected first shape + new rubric ms |','|---|---:|---:|---:|---:|']

 for name,row in s['speed'].items():

  a=row['none'];b=row['contextual'];lines.append(f'| {name} | {a["warm_median_ms"]:.1f} | {b["warm_median_ms"]:.1f} | {a["first_ms"]:.1f} | {b["first_ms"]:.1f} |')

 lines += ['', 'The correction adds first-use work for each new rubric. An earlier implementation unnecessarily captured neutral-evaluation graphs and retained extra GPU buffers; the final implementation disables capture only during neutral scoring. The earlier measurements remain archived in production-with-neutral-graphs/. Repeated warmed measurements above should have zero neutral forward calls; raw counters are in implementation-summary.json. Dynamic question wording or eviction can trigger new work. These timings compare correction on/off, not the still-unfinished full base/HF startup and speed benchmark.','',

 '## Numerical and engineering checks','']

 for name,row in s['path_parity'].items():lines.append(f'- {name}: {row["cases"]} diagnostic cases, {row["choice_flips"]} label changes versus plain execution, maximum probability difference {100*row["max_probability_delta"]:.3f} percentage points.')

 lines += ['- 13 existing engine tests, 8 workflow tests, and 5 new correction tests passed. New tests cover the correction arithmetic, invalid vectors, cache invalidation/bounds, duplicate-rubric coalescing, dynamic inputs, synthetic warmup isolation, and graph-setting restoration after a neutral-evaluation error.','- Across all 250 corrected predictions, optimized versus plain execution changed no labels. The median per-case maximum score difference was 0.0038 percentage points; 11 cases exceeded 1 percentage point, with a maximum of 3.278 points. Neutral-score division can amplify small FP16 differences. This exceeds the earlier 1-point preference in those cases; it is retained as an opt-in tradeoff because decisions and aggregate accuracy were unchanged. Scores remain uncalibrated.', '- Kernel differences near a decision boundary can still change a label: the uncorrected full plain run was 207/250 versus 208/250 optimized. They do not explain the large confident failures examined here.','',

 '## Methodology and limits','',

 'The original suite contains 100 BANKING77 test examples across 10 intents and 150 constructed routing/boolean/scoring examples. Scoring examples are strongly templated. This is not the full BANKING77 benchmark or a representative production distribution. The 250-case results became development evidence once failures were examined; they are not a held-out estimate of the correction.','',

 'A fixed 41-case subset was used for candidate exploration. The selected neutral correction was then evaluated on the entire original suite. The additional 80 cases were frozen and hashed before evaluating the correction and include 20 non-overlapping BANKING77 rows, 18 new routing prompts, 12 boolean examples, 10 severity examples, and 20 examples of a new document workflow. Some are constructed/templated. No labels were edited to improve results.','',

 'The actual pinned HF PyTorch implementation was run with the same resident checkpoint and FP16 hardware setup. Its output adapter exposes all candidate probabilities and validates the common contract. HF did not receive the new correction. Original normal-Qwen JSON-generation results remain preserved; most failures there were schema-contract failures and must not be confused with incorrect but valid decisions.','',

 '## Remaining wrong decisions','']

 for r in changes['remaining'][:12]:lines.append(f'- `{r["case_id"]}`: {r["context"]} Expected `{r["expected"]}`, got `{r["after"]}`.')

 lines += ['', 'All fixed, regressed and remaining decisions are in `changed-decisions.json`. No claim is made that this correction resolves all semantic errors. More kernel work cannot supply missing judgment; materially higher reliability needs a better decision-trained model or evaluation and calibration against representative application data.','',

 '## Candidate experiments (41-case development subset)','', '| Candidate | Correct |','|---|---:|']

 for fn in ['prompt-experiments.jsonl','catalog-experiments.jsonl','answer-experiments.jsonl','calibration-experiments.jsonl','binary-experiments.jsonl','prefix-experiments.jsonl','instruction-experiments.jsonl']:

  counts=collections.defaultdict(lambda:[0,0])

  for r in map(json.loads,(OUT/fn).read_text().splitlines()):counts[r['variant']][0]+=r['correct'];counts[r['variant']][1]+=1

  for name,(good,n) in counts.items():lines.append(f'| {name.replace(chr(10)," (newline)")} | {good}/{n} |')

 lines += ['', 'Original engine: 33/41. Natural-label scoring and blank-context correction were also examined over the full suite; natural-label scoring did not produce a net improvement and was rejected. Rejected candidates remain diagnostics only; they are not production routing rules.','',

 '## Reproduce','', '```powershell', '$env:HF_HOME = Join-Path (Get-Location) ".cache/decision-engine/huggingface"', '.\\.cache\\decision-engine\\venv\\Scripts\\python.exe -m subset --workflow examples/decision_engine/router.workflow.json --interactive --device cuda --local-files-only --answer-bias contextual', '```','',

 'Run from the Inference Engine repository. Remove the answer-bias flag to reproduce the default behavior. `scripts/verify_accuracy_implementation.py` reruns the production checks in the prepared WSL environment. Raw JSONL, data hashes and `original-source/` preserve the audit trail. No commit or push was made.']

 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

 print(OUT/'REPORT.md')

if __name__=='__main__':main()
