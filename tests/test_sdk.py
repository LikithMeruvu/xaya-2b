import ast
import math
from pathlib import Path
import unittest
from xaya import XAYA
from xaya._inputs import validate, prompt


class FakeEngine:
    max_options = 256
    def decide(self, state, question, options, primitive, image):
        self.last = (options, primitive, image)
        probabilities = [1/len(options)]*len(options)
        return dict(index=0, option=options[0], probabilities=probabilities, latency_ms=1)


class SDKTests(unittest.TestCase):
    def test_prompt_matches_preserved_original(self):
        # Execute only the two audited pure formatting methods, never the legacy imports/loader.
        tree = ast.parse((Path(__file__).resolve().parents[1]/'legacy/inference.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='XayaDecisionModel')
        functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'_option_text','decide'}]
        decide = next(n for n in functions if n.name=='decide')
        decide.decorator_list = []
        decide.body = decide.body[:2]+[ast.Return(value=ast.Name(id='text',ctx=ast.Load()))]
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Original',bases=[],keywords=[],body=functions,decorator_list=[])],type_ignores=[])),'<formatting-only>','exec'),scope)
        original=scope['Original']()
        for primitive in ['choice','score','noul']:
            options=['A',{'label':'B','description':'A different candidate'}]
            self.assertEqual(prompt('line one\nline two','A question?',options,primitive),original.decide('line one\nline two','A question?',options,primitive))

    def test_option_limits_and_bad_inputs(self):
        for options in [[], ['x']*257, 'Yes', [{'label':''}], [123]]:
            with self.assertRaises(ValueError):validate('Question?',options,'choice')
        self.assertEqual(len(validate('Question?',['x']*256,'choice')),256)

    def test_yes_no_order_and_primitive(self):
        engine=FakeEngine(); out=XAYA(engine).yes_no('state','Question?')
        self.assertEqual(engine.last,(['Yes','No'],'noul',None))
        self.assertEqual((out['yes'],out['no']),(.5,.5))

    def test_score_expected_value_and_descriptions(self):
        engine=FakeEngine(); out=XAYA(engine).score('text','Rate quality',levels=[1,3,5],descriptions={1:'Poor',5:'Excellent'})
        self.assertEqual(out['expected_score'],3)
        self.assertEqual(engine.last[1],'score')
        self.assertEqual(engine.last[0][2],{'label':'5','description':'Excellent'})
        for levels in [[1,1],[2,1],[math.nan],[True]]:
            with self.assertRaises(ValueError):XAYA(engine).score('text','Rate quality',levels=levels)

    def test_choice_forwards_image(self):
        engine=FakeEngine(); XAYA(engine).choice('','Question?',['A','B'],image='image.png')
        self.assertEqual(engine.last,(['A','B'],'choice','image.png'))


if __name__=='__main__':unittest.main()
