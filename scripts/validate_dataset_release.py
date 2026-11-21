#!/usr/bin/env python3
"""Validate a staged benchmark release, including every declared Hub config."""
import argparse
import glob
import hashlib
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def validate(root, load_configs=True):
    manifest=json.loads((root/'metadata/checksums.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
    counts=json.loads((root/'metadata/pool_counts.json').read_text())
    pools={}
    for name, expected in counts.items():
        rows=pq.read_table(root/f'data/{name}.parquet').to_pylist()
        assert len(rows)==expected['records'], name
        assert len({r['id'] for r in rows})==len(rows), name
        assert sum(r['eligible_source_labels'] for r in rows)==expected['source_eligible']
        assert all(bool(r['gold_files_source'])==r['eligible_source_labels'] for r in rows)
        assert all(r['gold_files']==r['gold_files_source'] if r['label_policy']=='indexed_non_test_source'
                   else r['gold_files']==r['gold_files_as_released'] for r in rows), name
        pools[name]=rows
    assert sum(len(v) for k,v in pools.items() if k.startswith('core/'))==1265
    assert sum(len(v) for k,v in pools.items() if k.startswith('extensions/'))==1560
    assert len(pools['lockbox/queries'])==539
    assert sum(r['eligible_source_labels'] for r in pools['lockbox/queries'])==525
    assert sum(r['scored_in_paper'] for r in pools['extensions/matched_replication'])==702
    for r in pools['extensions/matched_replication']:
        view=r['query_id'].rsplit('::',1)[1]
        assert r['query']==json.loads(r['source_record_json'])[f'{view}_query']
    assert sum(len(v) for k,v in pools.items() if k.startswith('source_pools/'))==7431
    for name,rows in pools.items():
        if name.startswith('source_pools/'):
            for r in rows:
                original=json.loads(r['source_record_json'])
                assert original['query_id']==r['query_id'] and original['repo']==r['repo']
                assert original['problem_statement']==r['query'] and original['gold_files']==r['gold_files']
    files={}
    for row in pq.read_table(root/'metadata/corpus_files.parquet').to_pylist():
        files.setdefault(row['corpus_id'],set()).add(row['path'])
    for name,rows in pools.items():
        if name.startswith('core/'):
            for r in rows:
                assert set(r['gold_files'])<=files[r['corpus_id']], r['id']
                assert len(r['corpus_commit'])==40
    card=(root/'README.md').read_text()
    meta=yaml.safe_load(card.split('---',2)[1])
    if load_configs:
        from datasets import load_dataset
        for config in meta['configs']:
            for spec in config['data_files']:
                paths=sorted(glob.glob(str(root/spec['path'])))
                assert paths, config['config_name']
                expected=sum(pq.read_metadata(p).num_rows for p in paths)
                actual=load_dataset(str(root),name=config['config_name'],split=spec['split'],
                                    cache_dir=str(Path(tempfile.gettempdir())/'repo-premium-validation'))
                assert len(actual)==expected,(config['config_name'],len(actual),expected)
                print('loaded',config['config_name'],spec['split'],len(actual),flush=True)
    print('PASS: checksums, label policies, membership, source-record preservation, corpus joins, and Hub configurations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--skip-load',action='store_true')
    a=p.parse_args();validate(a.root,not a.skip_load)
