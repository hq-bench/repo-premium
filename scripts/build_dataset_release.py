#!/usr/bin/env python3
"""Export benchmark-only Hub files from a pinned paper-artifact snapshot.

Requires pyarrow, numpy and the repository's run_retrieval dependencies.
Inputs are read-only. The output directory must be empty. Source query pools
are retained losslessly as JSON strings; final evaluation membership is joined
against the actual cached runs, not reconstructed by applying new filters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from smoke_test_django import strip_pr_boilerplate

ARCHIVE = '97c0c94062634d8266a7238546169123a8ef2125'
TEST_DIR = re.compile(r'(^|/)(tests?|testing|__tests__|testdata|testscripts|unittests|__snapshots__)/')
TEST_FILE = re.compile(r'(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py|[^/]*_test\.go|[^/]*\.(test|spec)\.[cm]?[jt]sx?|[^/]*\.snap|[^/]*Tests?\.java)$')
EXTS = ('.py', '.pyi', '.go', '.rs', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.vue', '.java')
SCHEMA = pa.schema([(k, pa.string()) for k in ['id', 'query_id', 'pool', 'repo', 'query', 'base_commit', 'corpus_id',
    'corpus_commit', 'corpus_status', 'language', 'label_policy', 'source_path', 'source_record_json']]
    + [('gold_files', pa.list_(pa.string())), ('gold_files_source', pa.list_(pa.string())),
       ('gold_files_as_released', pa.list_(pa.string())), ('eligible_source_labels', pa.bool_()),
       ('scored_in_paper', pa.bool_())])


def read(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def digest(b):
    return hashlib.sha256(b).hexdigest()


def source_gold(gold, exts):
    return [g for g in gold if g.endswith(exts) and not TEST_DIR.search(g) and not TEST_FILE.search(g)]


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def make_corpora(manifest, local_corpora, out, git_repos=None):
    corpora = {}
    files = []
    for item in json.loads(manifest.read_text())['repos']:
        repo = item['org_repo']; key = repo.replace('/', '__')
        checkout = git_repos/key if git_repos else Path(item['clone_path'])
        for label, root in [('shared', local_corpora/'repos_py'/key), ('2024q1', local_corpora/'repos_2024q1_py'/key)]:
            commit = item['snapshot_sha'] if label == 'shared' else git(checkout, 'rev-list', '--all', '--before=2024-04-01T00:00:00Z', '-1').decode().strip()
            # NUL-separated output preserves non-ASCII paths and symlink targets.
            blobs = {}
            for entry in git(checkout, 'ls-tree', '-r', '-z', commit).split(b'\0'):
                if not entry:
                    continue
                header, path = entry.split(b'\t', 1)
                mode, _, blob = header.decode().split()
                if path.decode().endswith('.py'):
                    blobs[path.decode()] = (mode, blob)
            paths = sorted(p.relative_to(root).as_posix() for p in root.rglob('*.py'))
            assert set(paths) == set(blobs), (repo, label, set(paths)^set(blobs))
            cid = f'{repo}@{commit}'
            block = []
            for path in paths:
                payload = (root/path).read_bytes()
                mode, blob = blobs[path]
                if mode == '120000':
                    target = git(checkout, 'cat-file', 'blob', blob).decode()
                    resolved = (Path(path).parent / target).as_posix()
                    import posixpath
                    resolved = posixpath.normpath(resolved)
                    expected = git(checkout, 'show', f'{commit}:{resolved}')
                    assert payload == expected, (repo, path, 'symlink content')
                else:
                    assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest() == blob, (repo, label, path)
                block.append(dict(corpus_id=cid, path=path, sha256=digest(payload), git_blob=blob, mode=mode))
            corpus_hash = digest(''.join(f"{r['path']}\0{r['sha256']}\n" for r in block).encode())
            corpora[key, label] = dict(corpus_id=cid, repo=repo, commit=commit, code_extensions=['.py'],
                n_files=len(block), files_sha256=corpus_hash, upstream_url=f'https://github.com/{repo}',
                status='git_tree_and_materialized_contents_verified', symlinks='read target contents as retrieval runner does',
                files={r['path'] for r in block})
            files.extend(block)
    (out/'metadata').mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(files), out/'metadata/corpus_files.parquet', compression='zstd')
    return corpora


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--python-manifest', type=Path, required=True)
    p.add_argument('--local-corpora', type=Path, required=True)
    p.add_argument('--git-repos', type=Path, help='Portable directory of owner__repo Git clones; overrides legacy clone paths')
    p.add_argument('--archive-info', type=Path, required=True)
    p.add_argument('--lockbox-manifest', type=Path, action='append', required=True,
                   help='Runner TSV; repeat with corrected rows last. Only portable flags are exported.')
    a = p.parse_args()
    assert not a.out.exists() or not any(a.out.iterdir()), 'output must be empty'
    a.out.mkdir(parents=True, exist_ok=True)
    source, out = a.source, a.out
    info = json.loads(a.archive_info.read_text()); assert info['sha'] == ARCHIVE
    driver={}
    for manifest_path in a.lockbox_manifest:
        for r in csv.DictReader(manifest_path.open(),delimiter='\t'):
            driver[r['pool'],r['key']]={k:r[k] for k in ['pool','key','code_exts','camel','max_seq_length','strip','n','max_gold','batch']}
    corpus = make_corpora(a.python_manifest, a.local_corpora, out, a.git_repos)
    (out/'metadata/python_snapshots.json').write_text(json.dumps({'repos':[
        {k:r[k] for k in ['org_repo','snapshot_sha']} for r in json.loads(a.python_manifest.read_text())['repos']]},indent=2)+'\n')
    provenance = []; counts = {}; datasets = {}

    def row(pool, original, path, *, gold=None, released=None, query=None, cid=None, exts=EXTS,
            policy='indexed_non_test_source', scored=True, query_id=None, language='python', shared=True):
        r = original; repo = r['repo']; qid = query_id or r.get('query_id', r.get('task_id'))
        gold = list(gold if gold is not None else r.get('gold_files', r.get('main_gold_files', [])))
        released = list(released if released is not None else r.get('orig_gold_files', gold))
        gs = source_gold(gold, exts)
        c = cid or {}
        commit = c.get('commit', '')
        status = c.get('status', 'not_recorded_in_paper_archive')
        if not shared:
            commit = r.get('base_commit', '')
            status = 'task_base_commit_recorded_not_reconstructed'
        corpus_id = c.get('corpus_id', f'{repo}@{commit}' if commit else f'legacy-shared:{repo}')
        if c.get('files') is not None:
            assert set(gs) <= c['files'], (pool, qid, set(gs)-c['files'])
        text = query if query is not None else strip_pr_boilerplate(r['problem_statement'])
        primary = gold if policy == 'as_released' else gs
        return dict(id=f'{pool}/{repo}/{qid}', query_id=qid, pool=pool, repo=repo, query=text,
                    base_commit=r.get('base_commit', ''), corpus_id=corpus_id, corpus_commit=commit,
                    corpus_status=status, language=language, label_policy=policy, source_path=path,
                    source_record_json=json.dumps(r, ensure_ascii=False, sort_keys=True),
                    gold_files=primary, gold_files_source=gs, gold_files_as_released=released,
                    eligible_source_labels=bool(gs), scored_in_paper=scored)

    def write(name, rows):
        assert rows and len({r['id'] for r in rows}) == len(rows), (name, 'duplicate IDs')
        path = out/f'data/{name}.parquet'; path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression='zstd')
        counts[name] = dict(records=len(rows), source_eligible=sum(r['eligible_source_labels'] for r in rows),
                            repositories=len({r['repo'] for r in rows}), scored_in_paper=sum(r['scored_in_paper'] for r in rows))
        datasets[name] = rows

    # Lossless, explicitly non-evaluation source pools retain every original row.
    for directory in sorted((source/'release').iterdir()):
        if not directory.is_dir():
            continue
        rs = []
        for path in sorted(directory.glob('*.jsonl')):
            for n, r in enumerate(read(path), 1):
                x = row(f'source_{directory.name}', r, str(path.relative_to(source)), policy='as_released',
                        scored=False, language='mixed')
                # Curated_RAW and its processed companion can share IDs; keep source identity.
                x['id'] += f'/{path.name}:{n}'
                x['query'] = r['problem_statement']
                rs.append(x)
        write(f'source_pools/{directory.name}', rs)
    assert sum(v['records'] for k,v in counts.items() if k.startswith('source_pools/')) == 7431

    # Curated pool: select the 472 evaluated IDs, not the whole 500-row base.
    rs = []
    for path in sorted((source/'smoke_test_results/canon202607/jina05_cellA_c1000').glob('*/smoke_test_django.jsonl')):
        key = path.parent.name; qpath = source/f'release/qrels/{key}.jsonl'
        q = {r['query_id']:r for r in read(qpath)}
        for cached in read(path):
            r = q[cached['query_id']]
            assert r['gold_files'] == cached['gold_files']
            rs.append(row('cur_pre', r, str(qpath.relative_to(source)), gold=cached['gold_files'], cid=corpus[key,'shared'], exts=('.py',)))
    write('core/cur_pre', rs)

    # Harmonized pools: final RA is qrels_h24 + out24, not the older 135-row copy.
    for cell, pool in [('cellB','raw_post'), ('cellRA','raw_pre_2023'), ('cellRB','raw_pre_2022')]:
        qdir = source/('smoke_test_results/harmonized202610/qrels_h24/cellRA' if cell=='cellRA' else f'local_inputs_202610/qrels_harmonized/{cell}')
        cdir = source/('smoke_test_results/harmonized202610/out24/dense_cellRA_c1000' if cell=='cellRA' else 'smoke_test_results/harmonized202610/out/dense_all_c1000')
        rs = []
        for path in sorted(qdir.glob('*.jsonl')):
            key = path.stem; cp = cdir/key/'smoke_test_django.jsonl'
            cached = {r['query_id']:r for r in read(cp)} if cp.exists() else {}
            for r in read(path):
                hit = cached[r['query_id']]
                assert r['gold_files'] == hit['gold_files'], (pool,r['query_id'])
                rs.append(row(pool,r,str(path.relative_to(source)),cid=corpus[key,'2024q1' if cell=='cellRA' else 'shared'],exts=('.py',)))
        write(f'core/{pool}',rs)
    assert {k:v['records'] for k,v in counts.items() if k.startswith('core/')} == {
        'core/cur_pre':472,'core/raw_post':288,'core/raw_pre_2023':138,'core/raw_pre_2022':367}

    # Cross-language and curated extensions retain cached membership and both label views.
    groups = [
      ('cross_language', [('r9_vuejs_2023_jina-0.5b_c1000','qrels_r9_vuejs_2023',('.ts','.js','.vue','.tsx','.jsx'),'js_ts'),
       ('r9b_ts_c1000','qrels_r9b_ts',('.ts','.tsx','.js','.mjs','.cjs'),'typescript'),
       ('r9c_go_c1000','qrels_r9c_go',('.go',),'go'),('r9d_rust_c1000','qrels_r9d_rust',('.rs',),'rust')]),
      ('multi_swe_bench',[('cur_vuejs_ts_jina05_c1000','qrels_cur_vuejs_ts',EXTS,'js_ts'),
       ('cur_tokio_rs_jina05_c1000','qrels_cur_tokio_rs',EXTS,'rust'),('cur_mui_ts_jina05_c1000','qrels_cur_mui_ts',EXTS,'js_ts')]),
      ('heldout_python',[('heldout_jina05_c1000','qrels_heldout',('.py',),'python')])]
    for pool, specs in groups:
        rs=[]
        for run,qd,exts,lang in specs:
            for path in sorted((source/'smoke_test_results'/run).glob('*/smoke_test_django.jsonl')):
                key=path.parent.name;qp=source/f'release/{qd}/{key}.jsonl'
                qs={r['query_id']:r for r in read(qp)}
                for hit in read(path):
                    r=qs[hit['query_id']]
                    assert set(hit['gold_files'])==set(r['gold_files'])
                    gs=source_gold(hit['gold_files'],exts)
                    if pool=='heldout_python' and not gs:
                        continue
                    # Empty source-label rows stay in original cross-language membership.
                    rs.append(row(pool,r,str(qp.relative_to(source)),gold=hit['gold_files'],exts=exts,language=lang,scored=bool(gs)))
        write(f'extensions/{pool}',rs)

    # Fixed 539-query lockbox: do not silently change membership when labels change.
    task_repos=dict(l.split()[:2] for l in (source/'deepswe_pull/task_repo.tsv').read_text().splitlines() if l.strip())
    lock=[];deep=[];lock_protocol=[]
    for path in sorted((source/'smoke_test_results/qpp_lockbox202610').glob('*.jsonl')):
        pool,key=path.stem.split('__',1)
        flags=driver[pool,key]
        qp=source/f'smoke_test_results/gen_lockbox_qrels/{pool}/{key}.jsonl'
        qs={r['query_id']:r for r in read(qp)}
        for hit in read(path):
            original=qs[hit['query_id']]
            # Exclude generated text: this is the original retrieval query.
            r={k:v for k,v in original.items() if k in ['query_id','repo','problem_statement','gold_files','base_commit','language','reference_view']}
            assert set(r['gold_files']) == set(hit['gold_files'])
            exts=('.go',) if pool=='r9_go' else ('.rs',) if pool=='r9_rust' else ('.ts','.tsx','.js','.mjs','.cjs') if pool=='r9_ts' else EXTS
            language=r.get('language') or {'r9_ts':'typescript','r9_go':'go','r9_rust':'rust'}[pool]
            text=strip_pr_boilerplate(hit['query']) if flags['strip']=='1' else hit['query']
            x=row('lockbox',r,str(qp.relative_to(source)),query=text,gold=hit['gold_files'],exts=exts,
                  policy='as_released',language=language,shared=pool!='deepswe')
            lock.append(x)
            lock_protocol.append(dict(id=x['id'],original_pool=pool,key=key,strip_pr_boilerplate=flags['strip']=='1',
                code_extensions=flags['code_exts'].split(','),
                tokenize_camelcase=flags['camel']=='1',
                max_seq_length=int(flags['max_seq_length']),dense_chunk_chars=1000,chunk_overlap_lines=10,
                bm25_chunk_chars=1000000000,cached_ranking_depth=50))
            if pool=='deepswe':
                assert r['repo']==task_repos[key]
                y=dict(x,id=x['id'].replace('lockbox/','deepswe/',1),pool='deepswe',label_policy='indexed_non_test_source',gold_files=x['gold_files_source'])
                deep.append(y)
    write('lockbox/queries',lock);write('extensions/deepswe',deep)
    (out/'metadata/lockbox_protocol.json').write_text(json.dumps(lock_protocol,indent=2)+'\n')
    with (out/'metadata/lockbox_driver.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['pool','key','code_exts','camel','max_seq_length','strip','n','max_gold','batch'],delimiter='\t')
        writer.writeheader();writer.writerows(driver[k] for k in sorted(driver))
    assert len(lock)==539

    matched=[];taskpath=source/'scaleup_pull/splits_v2/retrieval_tasks_v2.jsonl'
    cached_tasks={r['rfilename'].split('/')[-2] for r in info['siblings'] if r['rfilename'].startswith('scaleup_pull/experiment_v1/retrieval/jina-0.5b/') and r['rfilename'].endswith('/smoke_test_django.jsonl')}
    for r in read(taskpath):
        for view in ['raw','curated']:
            matched.append(row('matched_replication',r,str(taskpath.relative_to(source)),gold=r['main_gold_files'],
                query=r[f'{view}_query'],query_id=r['task_id']+'::'+view,
                exts=EXTS,language=r['language_group'],shared=False,
                scored=r['task_id'] in cached_tasks and bool(source_gold(r['main_gold_files'],EXTS))))
    write('extensions/matched_replication',matched)
    assert [counts[f'extensions/{k}']['records'] for k in ['cross_language','multi_swe_bench','heldout_python','deepswe','matched_replication']]==[458,134,139,109,720]
    assert sum(r['scored_in_paper'] for r in matched)==702

    # Store portable provenance and immutable source checksums without legacy machine paths.
    corpus_rows=[]
    for c in corpus.values():
        corpus_rows.append({k:v for k,v in c.items() if k!='files'})
    (out/'metadata/corpora.json').write_text(json.dumps(corpus_rows,indent=2)+'\n')
    archive_files={r['rfilename']:r for r in info['siblings']}
    for name,rs in datasets.items():
        for path in sorted({r['source_path'] for r in rs}):
            r=archive_files[path]
            provenance.append(dict(dataset=name,archive_path=path,git_blob=r['blobId'],sha256=digest((source/path).read_bytes())))
    (out/'metadata/provenance.json').write_text(json.dumps(dict(archive_repo='hq-bench/repo-premium',archive_revision=ARCHIVE,
        source_files=provenance,corpus_manifest_sha256=digest(a.python_manifest.read_bytes())),indent=2)+'\n')
    (out/'metadata/pool_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    # Explain the old 135-row RA setup by exact ID/label differences.
    # The canonical export itself is pinned to the final 138 scored records.
    (out/'metadata/release_notes.json').write_text(json.dumps(dict(archive_revision=ARCHIVE,
        core_records=1265,extension_records=1560,lockbox_records=539,source_pool_records=7431,
        lockbox_source_eligible=sum(r['eligible_source_labels'] for r in lock),
        matched_tasks=360,matched_scored_tasks=351,
        raw_pre_2023='Final 138-query qrels_h24/cellRA joined to out24/dense_cellRA_c1000; supersedes the 135-query earlier-snapshot copy.',
        corpus_coverage='Core Python corpora verified against Git objects and local evaluated .py contents. Shared extension snapshot SHAs were not recorded in the legacy release and are explicitly marked unavailable, never inferred from PR base commits.',
        labels='Core uses indexed non-test source files. Lockbox primary labels retain original release membership and labels; source label alternatives and eligibility are explicit. Cross-language rows with empty source gold remain to preserve original membership.',
        source_pools='All 7431 original release records retained losslessly in source_record_json. These are mining/source variants, not additional independent evaluation splits.',
        privacy='All query records are derived from the already published scrubbed archive. No generated query text is used in benchmark queries.'),indent=2)+'\n')
    checks={str(p.relative_to(out)):digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'metadata/checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    main()
