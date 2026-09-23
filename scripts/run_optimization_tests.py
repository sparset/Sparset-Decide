"""Run local tiny-model and kernel tests; no downloads."""
import os,sys,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
runtime=Path.home()/'.cache/sparset-decide/profiling-20260918'
if (runtime/'tools/zig-cc').exists():os.environ.setdefault('CC',str(runtime/'tools/zig-cc'))
import torch
torch.set_num_threads(2)
if len(sys.argv)>1:
 suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
else:suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
out=ROOT/os.environ.get('DECISION_TEST_OUTPUT', 'outputs/decision-engine/optimization-20260918')
out.mkdir(parents=True,exist_ok=True)
summary={'tests_run':result.testsRun,'successful':result.wasSuccessful(),'failures':[(str(t),v)for t,v in result.failures],'errors':[(str(t),v)for t,v in result.errors],'skipped':[(str(t),v)for t,v in result.skipped],'selection':sys.argv[1:],'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
(out/('tests-subset.json'if len(sys.argv)>1 else 'tests.json')).write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(not result.wasSuccessful())