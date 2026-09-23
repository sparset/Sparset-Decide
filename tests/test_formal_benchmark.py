import importlib.util,json,unittest
from pathlib import Path
from sparset_decide import Choice
ROOT=Path(__file__).resolve().parents[1]
# Historical benchmark integration checks require local, untracked artifacts.
required = [
    ROOT/'outputs/decision-engine/formal-benchmark-20260918/cases.json',
    ROOT/'outputs/decision-engine/formal-benchmark-20260918/cases.sha256',
    ROOT/'outputs/decision-engine/comparison-20260918/replica-source/core/engine_torch.py',
]
if not all(path.is_file() for path in required):
    raise unittest.SkipTest('Historical benchmark artifacts are not present in this checkout')
spec=importlib.util.spec_from_file_location('formal_benchmark',ROOT/'scripts/run_formal_benchmark.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class BenchmarkValidationTests(unittest.TestCase):
    def test_valid_and_invalid_probability_contracts(self):
        q={'field':Choice('Choose',{'a':'A','b':'B'})}
        good={'field':{'choice':'a','probabilities':{'a':0.8,'b':0.2}}}
        self.assertTrue(module.validate(good,q))
        for obj in [{}, {'field':{'choice':'b','probabilities':{'a':0.8,'b':0.2}}},
                    {'field':{'choice':'a','probabilities':{'a':1.0}}},
                    {'field':{'choice':'a','probabilities':{'a':0.8,'b':0.8}}},
                    {'field':{'choice':'a','probabilities':{'a':float('nan'),'b':0.2}}},
                    {'field':{'choice':'a','probabilities':{'a':True,'b':0}}},
                    {'field':{'choice':'a','probabilities':{'a':0.8,'b':0.2},'extra':0}}]:
            with self.subTest(obj=obj),self.assertRaises((ValueError,TypeError)):module.validate(obj,q)
    def test_duplicate_fields_rejected(self):
        with self.assertRaises(ValueError):json.loads('{"x":1,"x":2}',object_pairs_hook=module.unique_object)
    def test_frozen_labels_belong_to_requested_options(self):
        import hashlib
        path=ROOT/'outputs/decision-engine/formal-benchmark-20260918/cases.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),path.with_suffix('.sha256').read_text().strip())
        cases=module.DATA['cases'];self.assertEqual(len(cases),250)
        self.assertEqual(len({c['id'] for c in cases}),250)
        for c in cases+module.DATA['robustness_cases']:
            for k,v in c['expected'].items():self.assertIn(v,dict(module.question_from_dict(c['questions'][k]).options))
if __name__=='__main__':unittest.main()
