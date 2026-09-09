"""Per-group Refine audio policy; imported by the scoped source adapter."""
import json


def resolve_controls(timeline_data, group_audio_slots, chunk_count, audio_slots,
                     preserve_stage1_latents, raw_latent_carry_test):
    timeline = json.loads(timeline_data) if isinstance(timeline_data, str) and timeline_data else (timeline_data or {})
    groups = (timeline or {}).get('chunks', [])
    explicit = json.loads(group_audio_slots) if group_audio_slots else None
    if explicit is not None and (not isinstance(explicit, list) or len(explicit) != chunk_count):
        raise ValueError('group_audio_slots must contain one slot list per saved group')
    if groups and len(groups) != chunk_count:
        raise ValueError('Refine timeline differs from saved group count')
    controls = []
    for index in range(chunk_count):
        group = groups[index] if groups else {}
        disabled = group.get('disable_previous_audio', False)
        if not isinstance(disabled, bool):
            raise ValueError('disable_previous_audio must be a JSON boolean')
        if disabled and (index == 0 or not preserve_stage1_latents or not raw_latent_carry_test
                         or group.get('generation_mode') != 'Reference'):
            raise ValueError('Video-only Refine requires a preserved Reference continuation')
        if disabled and explicit is None:
            raise ValueError('Video-only Refine requires explicit per-group voice slots')
        slots = explicit[index] if explicit is not None else [i + 1 for i, a in enumerate(audio_slots) if a is not None]
        if (not isinstance(slots, list) or any(type(s) is not int or s not in (1, 2, 3) for s in slots)
                or len(set(slots)) != len(slots)):
            raise ValueError('Audio slots must be unique one-based ids 1..3')
        if any(audio_slots[s - 1] is None for s in slots):
            raise ValueError('Selected Refine audio slot is not connected')
        refs = {f'ref_audio_{i}': audio_slots[s - 1] for i, s in enumerate(slots)} or None
        controls.append({'disable_previous_audio': disabled, 'slots': slots, 'references': refs})
    return controls


def inject_video_only(latent, source, carry_length, context_class):
    import importlib
    from .muse_minimax_director import _video_only_carry_inject, align_frame_count
    importlib.import_module(context_class.__module__)._require_h3_mask_support()
    return _video_only_carry_inject(latent, source['samples'].unbind()[0],
                                  align_frame_count(max(5, int(carry_length))))


def save_sampled(latent, directory, index, metadata):
    """Persist real sampled AV tensors before decode, without touching source bundles."""
    if not directory:
        return
    import hashlib
    from pathlib import Path
    import folder_paths
    import torch
    base = Path(folder_paths.get_output_directory()).resolve()
    root = (base / directory).resolve()
    if not root.is_relative_to(base) or root == base:
        raise ValueError('Refine checkpoint must be in a job-owned output subdirectory')
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'chunk_{index + 1:04d}.av.pt'
    streams = tuple(t.detach().to('cpu') for t in latent['samples'].unbind())
    with path.open('xb') as output:
        torch.save({'streams': streams, 'metadata': metadata}, output)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    if len(restored['streams']) != 2 or any(not torch.equal(a, b) for a, b in zip(streams, restored['streams'])):
        raise ValueError('Refine checkpoint readback failed')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    record = {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size,
              'shapes': [list(t.shape) for t in streams], 'metadata': metadata,
              'loadability_pass': True, 'storage_policy': 'remote_only', 'local_copy': False}
    with path.with_suffix('.json').open('x', encoding='utf-8') as output:
        json.dump(record, output, indent=2)
