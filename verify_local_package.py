"""CPU-only integrity check for the prepared, executed submission package."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(p.read_text('utf-8-sig'))

def main():
    models=read(ROOT/'models/manifest.json')
    for name,row in models.items(): assert sha(ROOT/'models'/name)==row['sha256'],name
    ref=read(ROOT/'verification/reference.json')
    assert sha(ROOT/'verification/reference-submission.csv')==ref['submission_sha256']
    for path in (ROOT/'sfseg').rglob('*.py'):
        tree=ast.parse(path.read_text('utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in ('scripts','filament')
    for name in ('cli-verification.json','notebook-verification.json'):
        check=read(ROOT/'verification'/name)
        assert check['passed'] and check['full_test_set'] and check['byte_identical_to_reference']
        assert check['images']==180 and check['instances']==1261
        assert check['submission_sha256']==ref['submission_sha256']
    nb=read(ROOT/'reproduce.executed.ipynb')
    cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in cells)
    assert not any(o['output_type']=='error' for c in cells for o in c.get('outputs',[]))
    assert [c['source'] for c in read(ROOT/'reproduce.ipynb')['cells']]==[c['source'] for c in nb['cells']]
    training=read(ROOT/'verification/optional-training-verification.json')
    assert training['passed'] and training['optimizer_updates']==0 and not training['training_executed']
    clean=read(ROOT/'verification/clean-environment-verification.json')
    assert clean['passed'] and clean['isolated_venv'] and not clean['include_system_site_packages']
    assert clean['verification']['submission_sha256']==ref['submission_sha256']
    assert clean['verification']['byte_identical_to_reference'] and clean['optimizer_updates']==0
    print(json.dumps({'passed':True,'models':len(models),'executed_code_cells':len(cells),
                      'submission_sha256':ref['submission_sha256'],'network_operations_performed':False},indent=2))

if __name__=='__main__':main()
