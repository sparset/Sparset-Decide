"""Build the optimization acceptance report from saved measurements."""
import json,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/branch-optimization-20260918'
load=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
stages=load('candidates.json');final=load('confirmation.json');tests=load('tests.json');profile=load('profile-before.json');kernels=load('kernel-benchmarks.json')
assert len(final['cases'])==6 and all(len(rows)==5 for c in final['cases'].values() for rows in c['methods'].values()),'Confirmation incomplete'
lines=['# Branch optimization acceptance report','',
'Qwen2.5-1.5B-Instruct, pinned weights 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, FP16 on RTX 3050 Laptop 4 GB, Linux/WSL, PyTorch 2.6.0+cu124, Transformers 5.5.4. No training or weight changes.','',
'## Enabled and rejected changes','',
'- Enabled automatically: group similar-length questions when padding drops by at least 10%; shared physical prefix attention inside the measured hardware/shape envelope; 16-row attention tiles only for suffix widths <=16; otherwise ordinary/general attention.',
'- CUDA graph captures now reuse one model-scoped stream. This fixes workspace accumulation with varying shapes. Graphs themselves remain opt-in and exact-shape.',
'- Implemented and tested, but kept opt-in: fused residual addition + RMSNorm. Isolated operation timings improved, but end-to-end timings did not improve consistently.',
'- No workflow field names, rubrics, schema edits, or per-user kernel code are required. Unsupported configurations automatically use the ordinary backend.','',
'## Final alternating measurements','',final['method'],'',
'Both reference and automatic paths use the existing three fused kernels and CUDA graphs. The reference disables the new branch features. Both benefit from the stream-reuse fix, so these ratios do not inflate gains by comparing against a memory-leaking benchmark.','',
'| Workload | Reference | Grouping only | Automatic | Speedup | Largest probability shift | Changed labels |','|---|---:|---:|---:|---:|---:|---:|']
max_delta=0.;changes=0;fields=0;responses=0;changed_delta=0.;changed_context_decisions=0
for name,c in final['cases'].items():
    m=c['medians'];delta=max(x['max_probability_delta']for x in c['parity']['automatic']);changed=sum(len(x['changed_labels'])for x in c['parity']['automatic'])
    max_delta=max(max_delta,delta);changes+=changed;changed_delta=max(changed_delta,c['changed_context_parity']['max_probability_delta'])
    changed_context_decisions+=len(c['changed_context_parity']['changed_labels'])
    fields+=c['fields']*len(c['methods']['automatic']);responses+=len(c['methods']['automatic'])
    lines.append(f"| {name} | {m['reference']:.1f} ms | {m['length_only']:.1f} ms | {m['automatic']:.1f} ms | {m['reference']/m['automatic']:.2f}x | {delta*100:.4f} percentage points | {changed} |")
lines+=['',f'Maximum automatic-vs-reference shift: {max_delta*100:.4f} percentage points. Changed selected labels across timed pairs: {changes}. Maximum shift on changed-context checks: {changed_delta*100:.4f} percentage points. Changed-context decision flips: {changed_context_decisions}. The acceptance threshold is an absolute probability difference of 0.01 (one percentage point), not relative percent error.','',
'Probability agreement is not proof of classification accuracy or calibration. No large verified accuracy dataset was introduced. All measured outputs use the same deterministic typed schema.','',
'## Memory and padding','', '| Workload | Reference peak allocated | Automatic peak allocated | Padding tokens before -> after | Shared / short batches |','|---|---:|---:|---|---|']
for name,c in final['cases'].items():
    a=c['methods']['automatic'][0]['result']['metadata']['optimizations']
    memory={m:statistics.median(r['peak_allocated_bytes']for r in rows)/2**20 for m,rows in c['methods'].items()}
    lines.append(f"| {name} | {memory['reference']:.1f} MiB | {memory['automatic']:.1f} MiB | {a['padded_branch_tokens_before']} -> {a['padded_branch_tokens_after']} | {a['shared_branch_batches']} / {a['short_branch_batches']} |")
lines+=['','Peak allocated memory is PyTorch allocation, not total device usage. Driver/desktop memory and allocator reservations are additional.','',
'## Individual-stage measurements','',
'Three warm samples per stage; consecutive method samples with order rotated across workloads. These screen candidates. The alternating confirmation above is preferred for final end-to-end conclusions.','',
'| Workload | Stage | Median | Reference / stage | Maximum probability shift | Changed labels |','|---|---|---:|---:|---:|---|']
for name,c in stages['cases'].items():
    for stage,m in c['methods'].items():
        if m['status']!='ok':lines.append(f"| {name} | {stage} | error | | | |");continue
        p=m['vs_reference'];lines.append(f"| {name} | {stage} | {m['median_ms']:.1f} ms | {m['speedup']:.2f}x | {100*p['max_probability_delta']:.4f} pp | {', '.join(p['changed_labels'])or'none'} |")
lines+=['','Stage short = shared-prefix attention with the short-query dispatch, residual = only the new residual/normalization fusion, length = only length grouping. Combined includes residual fusion; the final automatic profile deliberately excludes residual fusion.','',
'## Kernel observations','',
'Attention accounted for the following portion of summed CUPTI kernel durations in the pre-change engine (not a percentage of full wall time):']
for name,p in profile.items():
    total=sum(k['us']for k in p['kernels']);attn=sum(k['us']for k in p['kernels']if 'fmha'in k['name']or'flash'in k['name'])
    lines.append(f'- {name}: {100*attn/total:.1f}% attention. Matrix multiplications dominate; reducing padded work is therefore valuable.')
lines+=['','The short-tile microbenchmark tested a 16-row tile against a 32-row tile. At 8/16 suffix tokens, it measured roughly 1.18-1.26x kernel speedups with identical sampled attention outputs. At 32/64 tokens it regressed, so automatic dispatch excludes those cases. Some longer-shape control samples showed large timing outliers even for identical kernels; do not use those outliers as optimization evidence.','',
'Fused residual/normalization isolated timings:','', '| Rows | Separate | Fused | Speedup |','|---|---:|---:|---:|']
for x in kernels['residual']:lines.append(f"| {x['rows']} | {x['separate_ms']*1000:.2f} us | {x['fused_ms']*1000:.2f} us | {x['speedup']:.2f}x |")
lines+=['','Microbenchmarks used CUDA-event timing around replay of graphs containing 50 calls, amortizing Python dispatch overhead. Kernel improvements do not imply the same full-request speedup.','',
'## Validation and fallback','',f"{tests['tests_run']} regression tests passed; skipped: {len(tests['skipped'])}. Coverage includes numerical attention references, FP16/BF16, GQA, ragged masks, causal isolation, immutable prefix caches, changed inputs, question-order restoration, CPU fallback, compiler-resource fallback, graph replay, and bounded allocation growth across new graph shapes.",'',
 f'All {responses} timed automatic responses ({fields} answer fields) passed strict JSON serialization, exact question/option keys, and deterministic typed-output reconstruction. Changed-context responses were checked too.','',
'Automatic shared attention is currently limited to Linux, Triton, SDPA, dense Qwen2/Transformers 5.5.4, FP16/SM8.6, head sizes 32/64/128, 128-4096 shared prefix tokens, at least two branches, and suffix width <=512. Other supported engine inputs use the ordinary path; the model still runs without users rewriting kernels. See docs/branch-optimizations.md for explicit controls and experimental on mode.','',
'All speed results are warm. Model loading, JIT compilation, graph capture, and fresh-process startup are excluded. Exact graph-shape reuse is still needed for replay gains. Small local samples on one laptop GPU do not establish universal speedups.','',
'## Reproducibility','',
'- scripts/profile_current_branches.py and profile-before.json: pre-change CUPTI evidence.',
'- scripts/benchmark_branch_candidates.py and candidates.json: separate stages after the stream fix.',
'- scripts/benchmark_branch_kernels.py and kernel-benchmarks.json: isolated kernel experiments. The saved exploration tested small tiles up to 64 tokens; production now restricts them to <=16.',
'- scripts/confirm_branch_optimizations.py and confirmation.json: final five-round alternating measurements and changed-context checks.',
'- scripts/run_optimization_tests.py and tests.json: regression suite.',
'- before_*.py: pre-change local source snapshots. source-manifest.json: final source hashes.',
'- candidates-before-stream-fix.json: interrupted diagnostic run, excluded from accepted timing claims because repeated capture streams accumulated workspaces.',
'- kernel-benchmarks-dispatch-noisy.json: early host-dispatch-sensitive experiment, excluded from kernel claims.','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((ROOT/'src/sparset_decide').glob('*.py'))}
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'max_probability_delta':max_delta,'changed_context_delta':changed_delta,'changed_labels':changes,'changed_context_decisions':changed_context_decisions,'responses':responses,'fields':fields,'tests':tests['tests_run']}))
print(OUT/'REPORT.md')
