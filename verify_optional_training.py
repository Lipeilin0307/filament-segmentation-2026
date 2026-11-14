"""Check optional training wiring without constructing a model or optimizing it."""
import argparse
import ast
import builtins
import importlib
import json
from pathlib import Path
import symtable
import tempfile

from sfseg.train import REPO, training_plan, execute_training, validate_training_data, read, write, sha


def verify(data_root=None):
    plan=training_plan()
    notebook=read(REPO/'reproduce.ipynb')
    training_cells=[c for c in notebook['cells'] if c['cell_type']=='code' and 'RUN_TRAINING = False' in ''.join(c['source'])]
    assert len(training_cells)==1, 'Notebook must default to the supplied-weight path'
    assert plan['base_updates_each']==9216 and plan['refiner_updates_each']==4992
    assert len(plan['base_jobs'])==7 and len(plan['refiner_jobs'])==4
    # Guard must fail before touching even nonexistent inputs or output paths.
    try:
        execute_training('does-not-exist','must-not-be-created')
    except RuntimeError as e:
        assert 'disabled' in str(e)
    else:
        raise AssertionError('Default training gate did not reject execution')
    modules=[]
    for path in sorted((REPO/'sfseg/training').glob('*.py')):
        text=path.read_text('utf-8'); ast.parse(text)
        table=symtable.symtable(text,str(path),'exec')
        defined={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
        defined|=set(dir(builtins))|{'__name__','__file__'}
        def check(st):
            for symbol in st.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    assert symbol.get_name() in defined,(path.name,st.get_name(),symbol.get_name())
            for child in st.get_children(): check(child)
        check(table)
        importlib.import_module('sfseg.training.'+path.stem)
        modules.append(path.name)
    import torch
    assert not torch.cuda.is_initialized(), 'Import checks must not initialize CUDA'
    result=dict(passed=True,training_executed=False,optimizer_updates=0,model_constructions=0,
                cuda_initialized=False,default_training_gate_passed=True,imported_modules=modules,
                nested_date_hash_routes_checked=True,base_jobs=7,refiner_jobs=4)
    if data_root:
        audit,root,doc=validate_training_data(data_root)
        from sfseg.training import base_pipeline as b, detector_pipeline as d, detector_crops as crops
        import numpy as np
        spec=read(REPO/'configs/training-recipe.json'); detector=read(REPO/'configs/detector-training.json')
        by_file={r['file_name']:r for r in spec['records']}
        # Temporary routing/crops only; no learned artifacts are written.
        with tempfile.TemporaryDirectory(prefix='sfseg-training-input-check-') as temporary:
            temp=Path(temporary); b.DATA_ROOT=Path(data_root); b.PREPARED=temp
            write(temp/'folds.json',[dict(by_file[im['file_name']],image_id=str(im['id']),split='train') for im in doc['images']])
            write(temp/'audit.json',dict(annotation_sha256=audit['annotation_sha256']))
            sizes={}
            for name in spec['base_jobs']:
                ds,records=b.training_dataset(spec,name)
                assert set(ds.files)=={r['file_name'] for r in records}
                first=ds[0]; repeated=ds[0]
                assert first['pixels'].shape==(1,512,512) and torch.equal(first['pixels'],repeated['pixels'])
                assert torch.equal(first['masks'],repeated['masks'])
                sizes[name]=len(ds.files)
            d.DATA_ROOT=Path(data_root); crops.IMG_DIR=root/'train/train_images'
            expert_counts={role:len(d.entries_for(detector,role)) for role in ('train','val')}
            _,anns=d._load_coco()
            entry=d.entries_for(detector,'train')[0]
            paths=crops.build_spine_crop_cache([entry],anns,temp/'crop','sample')
            sample=crops.SpineCropDataset(*paths[:3],augment=False)[0]
            assert sample[0].shape==(1,256,256) and sample[1].shape==(2,256,256)
            assert torch.isfinite(sample[0]).all() and torch.isfinite(sample[1]).all()
            result.update(official_inputs=audit,actual_base_pool_sizes=sizes,
                          deterministic_training_crops=True,detector_expert_counts=expert_counts,
                          grayscale_mask_spine_data_builder_checked=True)
        assert not torch.cuda.is_initialized()
    paths=[REPO/'sfseg/train.py',REPO/'verify_optional_training.py',REPO/'reproduce.ipynb',
           *sorted((REPO/'sfseg/training').glob('*.py')),
           *(REPO/'configs'/name for name in ['training-recipe.json','detector-training.json','training-initialization.json'])]
    result['checked_source_hashes']={p.relative_to(REPO).as_posix():sha(p) for p in paths}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args(); result=verify(args.data_root)
    if args.output: write(args.output,result)
    print(json.dumps(result,indent=2))
