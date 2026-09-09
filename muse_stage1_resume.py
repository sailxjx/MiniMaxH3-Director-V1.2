"""Validated native-audio Stage1 prefix reuse for Muse Director.

No sampler, media post-processing or original-cache mutation lives here.
The Director retains ownership of decode, native carry and suffix sampling.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

SCHEMA = 'muse_native_stage1_resume_v1'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def timeline_prefix(timeline, count):
    data = {key: value for key, value in timeline.items() if key not in ('chunk_frames', 'chunks')}
    data['chunk_frames'] = timeline.get('chunk_frames', [])[:count]
    data['chunks'] = timeline.get('chunks', [])[:count]
    return data


def reference_hashes(timeline, input_root):
    root = Path(input_root).resolve()
    found = {}
    def visit(value):
        if isinstance(value, dict):
            name = value.get('file')
            if isinstance(name, str) and name:
                path = (root / name).resolve()
                if root not in path.parents or not path.is_file():
                    raise ValueError('Reference file missing or outside input root')
                found[name] = {'remote_name': name, 'sha256': digest(path)}
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(timeline)
    return [found[key] for key in sorted(found)]


def validate_manifest(raw, contract, timeline, prompts, start, end, output_root, input_root):
    spec = json.loads(raw) if isinstance(raw, str) else raw
    if spec.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported native Stage1 resume manifest')
    total = len(timeline.get('chunk_frames', []))
    if not 1 <= start < total or end != total - 1:
        raise ValueError('Native Stage1 resume must reuse a nonempty prefix and regenerate the full suffix')
    if spec.get('contract') != contract:
        raise ValueError('Native Stage1 resume sampling contract changed')
    if timeline_prefix(spec['timeline'], start) != timeline_prefix(timeline, start):
        raise ValueError('Native Stage1 resume prefix timeline/references changed')
    allowed = (Path(output_root) / 'latent/erase_tomorrow/muse_stage1_scout').resolve()
    directory = Path(spec['candidate_directory']).resolve()
    if allowed not in directory.parents or not directory.is_dir():
        raise ValueError('Resume bundle must exist under the Stage1 scout output root')
    entries = spec.get('chunks', [])
    if len(entries) < start:
        raise ValueError('Resume bundle has insufficient prefix chunks')
    selected = []
    for index in range(start):
        item = dict(entries[index])
        if item.get('index') != index:
            raise ValueError('Resume prefix chunk ordering mismatch')
        path = directory / ('chunk_%04d.pt' % (index + 1))
        if allowed not in path.resolve().parents:
            raise ValueError('Resume chunk resolves outside the scout output root')
        if not path.is_file() or digest(path) != item.get('sha256', '').lower():
            raise ValueError('Resume prefix latent missing or hash mismatch: ' + path.name)
        if item.get('requested_frames') != timeline['chunk_frames'][index]:
            raise ValueError('Resume prefix frame count changed')
        if item.get('prompt_sha256') != hashlib.sha256(prompts[index].encode('utf-8')).hexdigest():
            raise ValueError('Resume prefix prompt changed')
        if not isinstance(item.get('trim_frames'), int) or item['trim_frames'] < 0:
            raise ValueError('Resume prefix needs its exact original context trim')
        if not isinstance(item.get('decoded_frames'), int) or not isinstance(item.get('delivered_frames'), int):
            raise ValueError('Resume prefix needs decoded/delivered frame evidence')
        if item['decoded_frames'] - item['trim_frames'] != item['delivered_frames']:
            raise ValueError('Resume prefix frame equation mismatch')
        item['source_path'] = str(path)
        seed = timeline['chunks'][index].get('seed_override', contract['seed'])
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError('Resume prefix seed must be uint64')
        item['expected_seed'] = seed
        selected.append(item)
    media_root = Path(input_root).resolve()
    for media in spec.get('reference_files', []):
        path = (media_root / media['remote_name']).resolve()
        if media_root not in path.parents or not path.is_file() or digest(path) != media['sha256'].lower():
            raise ValueError('Resume reference bytes missing or changed')
    if not spec.get('reference_files'):
        raise ValueError('Resume manifest needs reference byte hashes')
    return dict(spec, prefix=selected, start=start, end=end)


def load_prefix(entry, loader, contract):
    # Verify immediately before loading the trusted project-owned pickle.
    path = Path(entry['source_path'])
    if digest(path) != entry['sha256'].lower():
        raise ValueError('Resume source changed after preflight')
    payload = loader(path)
    latent = payload.get('latent') if isinstance(payload, dict) else None
    if not isinstance(latent, dict) or 'samples' not in latent:
        raise ValueError('Resume requires a true AV latent payload')
    if latent.get('_muse_seed_used') != entry.get('expected_seed', contract['seed']):
        raise ValueError('Resume payload seed mismatch')
    if latent.get('_muse_first_pass_steps_used') != contract['first_pass_steps']:
        raise ValueError('Resume payload Stage1 step mismatch')
    if '_muse_steps_used' in latent and latent['_muse_steps_used'] != contract.get('steps'):
        raise ValueError('Resume payload total steps mismatch')
    if hashlib.sha256(payload.get('prompt', '').encode('utf-8')).hexdigest() != entry['prompt_sha256']:
        raise ValueError('Resume payload prompt mismatch')
    return payload


def copy_prefix(entry, destination):
    source = Path(entry['source_path'])
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.resolve() == source.resolve():
        raise ValueError('Resume output must be a new independent bundle')
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    if digest(target) != entry['sha256'].lower():
        raise ValueError('Copied resume payload hash mismatch')


def record_chunk(directory, index, contract, timeline, prompt, requested_frames,
                 decoded_frames, trim_frames, delivered_frames, references, reused_from=None):
    root = Path(directory)
    path = root / 'native_resume.json'
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['contract'] != contract or record['timeline'] != timeline:
            raise ValueError('Cannot mix resume records from different runs')
    else:
        record = {'schema_version': SCHEMA, 'contract': contract, 'timeline': timeline, 'reference_files': references,
                  'candidate_directory': str(root), 'chunks': []}
    if index != len(record['chunks']):
        raise ValueError('Resume output must record chunks in order')
    record['chunks'].append({'index': index, 'sha256': digest(root / ('chunk_%04d.pt' % (index + 1))),
        'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        'requested_frames': int(requested_frames), 'decoded_frames': int(decoded_frames),
        'trim_frames': int(trim_frames), 'delivered_frames': int(delivered_frames),
        'reused_from': reused_from, 'sampling_executed': reused_from is None})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)
