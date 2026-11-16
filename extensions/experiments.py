"""Cached operators, resumable generation, and judging for notebooks 01–02."""

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import gc
import hashlib
import json
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import ROOT

KEYS = ['N', 'R', 'operator', 'drift_weight', 'lambda']
METRICS = ['along_v', 'change', 'orthogonal', 'push_energy']


@dataclass(frozen=True)
class Sweep:
    notebook: str
    model_name: str = 'Qwen/Qwen3-4B'
    concept: str = 'football'
    prompt: str = 'Once upon a time'
    trials: int = 1000
    lambdas: tuple = tuple(range(1, 9))
    r_values: tuple = (1.0, 10.0)
    last_n: tuple = (4, 8, None)
    operators: tuple = ('Identity', 'Mean Jacobian', 'Random')
    drift_weights: tuple = (0.0, 2.0)
    q: float = 0.1
    qf: float = 0.1
    seed: int = 0
    random_seed: int = 0
    direction_samples: int = 200
    jacobian_samples: int = 50
    max_ctx: int = 32
    direction_batch: int = 8
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.95
    generation_batch: int = 500
    trace_batch: int = 100
    record_layer_actuation: bool = False
    judge_batch: int = 256
    judge_chunk: int = 2048
    dtype: str = 'auto'
    device_map: str = 'auto'
    run_name: str | None = None

    @property
    def model_dir(self):
        return ROOT / 'artifacts' / self.model_name.rsplit('/', 1)[-1].lower()

    @property
    def run_dir(self):
        return self.model_dir / self.notebook / 'review' / (self.run_name or f'{self.concept}_n{self.trials}')

    @property
    def cache_dir(self):
        return self.model_dir / 'operators'


def _metadata(config):
    path = config.run_dir / 'config.json'
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved['settings'] != json.loads(json.dumps(asdict(config))):
        raise ValueError(f'{path} describes different settings; choose a new run_name to keep that run intact')
    return saved


def _save(frame, path):
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False, compression='gzip' if path.suffix == '.gz' else None)
    temporary.replace(path)


def _prepare(config):
    import torch
    from .data import concept_direction, load_concept_prompts, load_model
    from .jacobian import efficient_mean_jacobian

    index_path = config.cache_dir / 'index.json'
    if index_path.exists() and json.loads(index_path.read_text())['model'] != config.model_name:
        raise ValueError(f'{index_path} belongs to a different model')
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    tokenizer, model = load_model(config.model_name, dtype=config.dtype, device_map=config.device_map)
    layers, hidden = len(model.model.layers), model.get_input_embeddings().weight.shape[1]
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if not index_path.exists():
        index_path.write_text(json.dumps({'model': config.model_name}, indent=2) + '\n')
    identity = torch.eye(hidden)
    e_path = config.cache_dir / f'E_{config.concept}_n{config.direction_samples}_seed{config.seed}.pt'
    a_path = config.cache_dir / f'A_{config.concept}_n{config.jacobian_samples}_ctx{config.max_ctx}_seed{config.seed}.pt'
    if e_path.exists():
        E = torch.load(e_path, map_location='cpu', weights_only=True)
    else:
        E = concept_direction(model, tokenizer, config.concept, config.direction_samples, config.direction_batch)
        torch.save(E, e_path)
    if E.shape != (layers + 1, hidden):
        raise ValueError(f'{e_path} does not match the selected model')
    operators = {'Identity': identity.expand(layers, -1, -1)}
    if {'Mean Jacobian', 'Random'} & set(config.operators):
        if a_path.exists():
            A = torch.load(a_path, map_location='cpu', weights_only=True)
        else:
            random.seed(config.seed + 1)
            targets, _ = load_concept_prompts(config.concept, config.jacobian_samples)
            A = efficient_mean_jacobian(model, tokenizer, targets, max_ctx=config.max_ctx).cpu()
            torch.save(A, a_path)
        if A.shape != (layers, hidden, hidden):
            raise ValueError(f'{a_path} does not match the selected model')
        operators['Mean Jacobian'] = A
        if 'Random' in config.operators:
            noise = torch.randn(layers, hidden, hidden, generator=torch.Generator().manual_seed(config.random_seed))
            scale = torch.linalg.matrix_norm(A - identity) / torch.linalg.matrix_norm(noise)
            operators['Random'] = identity + scale[:, None, None] * noise
    return tokenizer, model, E, {name: operators[name] for name in config.operators}


def _grid(config, layers):
    scopes = list(dict.fromkeys(layers if n is None else n for n in config.last_n))
    grid = pd.DataFrame(product(scopes, config.r_values, config.operators,
                                config.drift_weights, config.lambdas), columns=KEYS)
    baseline = pd.DataFrame([[0, 0.0, 'Unsteered', 0.0, 0.0]], columns=KEYS)
    grid = pd.concat([baseline, grid], ignore_index=True)
    grid['path'] = [config.run_dir / 'cells' / f'n{n}_r{r:g}_{op.lower().replace(" ", "_")}_w{w:g}_l{strength:g}.csv'
                    for n, r, op, w, strength in grid.itertuples(index=False, name=None)]
    return grid


def _layer_path(config, path):
    return config.run_dir / 'layer_cells' / (path.name + '.gz')


def _completed(config, grid):
    return grid.path.map(lambda path: path.exists() and (
        not config.record_layer_actuation or _layer_path(config, path).exists()))


def _aggregate_actuation(config, grid):
    """Rebuild layer outputs solely from committed condition checkpoints."""
    if not config.record_layer_actuation:
        return
    paths = grid.loc[_completed(config, grid), 'path']
    if paths.empty:
        return
    records = pd.concat([pd.read_csv(_layer_path(config, path)) for path in paths], ignore_index=True)
    _save(records, config.run_dir / 'layer_actuation.csv.gz')
    summary = records.groupby(KEYS + ['layer']).agg(
        push_energy=('push_energy', 'mean'), n=('push_energy', 'size')).reset_index()
    _save(summary, config.run_dir / 'layer_actuation_summary.csv')


def _collect(config, tokenizer, model, controller, settings, progress):
    import torch
    from .steering import sample, trace, completion_metrics

    seed = config.seed + (config.lambdas.index(settings['lambda']) if controller is not None else 0)
    torch.manual_seed(seed)
    prompt_length = len(tokenizer(config.prompt)['input_ids'])
    rows, layer_rows = [], []
    for start in range(0, config.trials, config.generation_batch):
        ids = sample(model, tokenizer, config.prompt, min(config.generation_batch, config.trials - start),
                     controller=controller, lmbda=settings['lambda'], max_new_tokens=config.max_new_tokens,
                     temperature=config.temperature, top_p=config.top_p)
        for offset in range(0, len(ids), config.trace_batch):
            batch = ids[offset:offset + config.trace_batch]
            if controller is None:
                metrics = [dict.fromkeys(METRICS, 0.0) for _ in batch]
            else:
                replay = trace(model, controller, settings['lambda'], batch, prompt_length)
                metrics = completion_metrics(replay, per_layer=True) if config.record_layer_actuation else completion_metrics(replay)
                del replay
            texts = tokenizer.batch_decode(batch[:, prompt_length:], skip_special_tokens=True)
            for i, (text, values) in enumerate(zip(texts, metrics)):
                identity = settings | dict(trial=start + offset + i, seed=seed)
                if config.record_layer_actuation:
                    energies = (values.pop('layer_push_energy') if controller is not None
                                else [0.0] * len(model.model.layers))
                    layer_rows.extend(identity | dict(layer=layer, push_energy=energy)
                                      for layer, energy in enumerate(energies))
                rows.append(identity | values | dict(completion=text.strip()))
        progress.update(len(ids))
    return pd.DataFrame(rows), pd.DataFrame(layer_rows)


def generate_run(config):
    """Resume committed conditions and rebuild optional per-layer outputs."""
    metadata = _metadata(config)
    if metadata and any((config.run_dir / name).exists() for name in ('summary.csv', 'results.csv.gz')):
        if not config.record_layer_actuation or (config.run_dir / 'layer_actuation_summary.csv').exists():
            return
    grid = _grid(config, metadata['layers']) if metadata else None
    if grid is not None and _completed(config, grid).all():
        _aggregate_actuation(config, grid)
        return
    import torch
    from .data import load_model
    from .steering import QwenLQRSteering, DriftLQRSteering

    tokenizer, model, E, operators = _prepare(config)
    nominal, controller = None, None
    try:
        if metadata is None:
            config.run_dir.mkdir(parents=True, exist_ok=True)
            metadata = dict(settings=asdict(config), layers=E.shape[0] - 1,
                            dtype=str(next(model.parameters()).dtype))
            (config.run_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
            grid = _grid(config, metadata['layers'])
        (config.run_dir / 'cells').mkdir(exist_ok=True)
        if config.record_layer_actuation:
            (config.run_dir / 'layer_cells').mkdir(exist_ok=True)
        pending = grid.loc[~_completed(config, grid)]
        if pending.drift_weight.ne(0).any():
            _, nominal = load_model(config.model_name, dtype=config.dtype, device_map=config.device_map)
        with tqdm(total=len(grid) * config.trials, initial=(len(grid) - len(pending)) * config.trials,
                  desc='Generating completions', unit='completion') as progress:
            for values, cells in pending.groupby(KEYS[:-1], sort=False):
                N, R, name, weight = values
                if name == 'Unsteered':
                    controller = None
                else:
                    controller = (QwenLQRSteering(model, operators[name], E, config.q, R, config.qf,
                                                  steer_last_n_layers=int(N))
                                  if weight == 0 else DriftLQRSteering(
                                      model, nominal, operators[name], E, config.q, R, config.qf,
                                      drift_weight=weight, steer_last_n_layers=int(N)))
                for row in cells[KEYS + ['path']].itertuples(index=False, name=None):
                    settings, path = dict(zip(KEYS, row[:-1])), row[-1]
                    progress.set_postfix(N=N, R=R, operator=name, w=weight, lmbda=settings['lambda'])
                    completions, actuation = _collect(config, tokenizer, model, controller, settings, progress)
                    if config.record_layer_actuation:
                        _save(actuation, _layer_path(config, path))
                    _save(completions, path)
                controller = None
    finally:
        del controller, model, nominal, operators
        gc.collect()
        torch.cuda.empty_cache()
    _aggregate_actuation(config, grid)


def summarize(results):
    means = ['concept', 'coherent', 'both', *METRICS]
    summary = results.groupby(KEYS).agg(**{name: (name, 'mean') for name in means},
                                       median_push_energy=('push_energy', 'median'), n=('concept', 'size')).reset_index()
    for name in ('concept', 'coherent', 'both'):
        summary[f'{name}_se'] = np.sqrt(summary[name] * (1 - summary[name]) / summary.n)
    return summary


def judge_run(config):
    """Judge saved completions in batches; resumed judging never regenerates text."""
    metadata = _metadata(config)
    if metadata and (config.run_dir / 'summary.csv').exists():
        return
    if metadata is None:
        raise FileNotFoundError('Generate the run before judging it')
    if (config.run_dir / 'results.csv.gz').exists():
        results = pd.read_csv(config.run_dir / 'results.csv.gz', keep_default_na=False)
        results = results.loc[results.operator.isin(('Unsteered', *config.operators))]
        _save(summarize(results), config.run_dir / 'summary.csv')
        return
    grid = _grid(config, metadata['layers'])
    results = pd.concat([pd.read_csv(path, keep_default_na=False) for path in grid.path], ignore_index=True)
    _save(results, config.run_dir / 'unjudged.csv')
    verdicts_path = config.run_dir / 'judge_cache.csv'
    verdicts = (pd.read_csv(verdicts_path, keep_default_na=False) if verdicts_path.exists()
                else pd.DataFrame(columns=['completion', 'concept', 'coherent']))
    unique = results[['completion']].drop_duplicates()
    pending = unique.loc[~unique.completion.isin(verdicts.completion)]
    if not pending.empty:
        import torch
        from .judge import build_judge, concept_criterion, COHERENCE

        judge = build_judge(batch_size=config.judge_batch, device_map=config.device_map)
        try:
            with tqdm(total=len(unique), initial=len(unique) - len(pending),
                      desc='Judging unique completions', unit='completion') as progress:
                for start in range(0, len(pending), config.judge_chunk):
                    batch = pending.iloc[start:start + config.judge_chunk].copy()
                    batch['concept'] = judge(concept_criterion(config.concept), config.prompt, batch.completion, desc='Concept')
                    batch['coherent'] = judge(COHERENCE, config.prompt, batch.completion, desc='Coherence')
                    verdicts = pd.concat([verdicts, batch], ignore_index=True)
                    _save(verdicts, verdicts_path)
                    progress.update(len(batch))
        finally:
            del judge
            gc.collect()
            torch.cuda.empty_cache()
    results = results.merge(verdicts, on='completion', how='left', validate='many_to_one')
    results[['concept', 'coherent']] = results[['concept', 'coherent']].astype(bool)
    results['both'] = results.concept & results.coherent
    _save(results, config.run_dir / 'results.csv.gz')
    _save(summarize(results), config.run_dir / 'summary.csv')


def replay_alignment(config, *, reuse=None):
    """Replay the full saved grid, optionally seeding checkpoints from a previous replay."""
    import torch
    from .data import load_model
    from .steering import QwenLQRSteering, DriftLQRSteering, trace, alignment_metrics

    if config.trials < 1 or config.trace_batch < 1:
        raise ValueError('trials and trace_batch must be positive')
    metadata = _metadata(config)
    if metadata is None:
        raise FileNotFoundError('Replay requires a saved run')
    source_path = config.run_dir / 'results.csv.gz'
    with source_path.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    output_dir = config.run_dir / 'alignment'
    settings = dict(source_sha256=source_hash, sweep=asdict(config),
                    protocol='Prompt and saved completion tokenized separately; visible completion tokens only. '
                             'Original token IDs, stripped whitespace, and special tokens are unavailable.')
    settings = json.loads(json.dumps(settings))
    output_dir.mkdir(exist_ok=True)
    settings_path = output_dir / 'config.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError(f'{settings_path} describes a different replay')
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    if (output_dir / 'summary.csv').exists():
        return pd.read_csv(output_dir / 'summary.csv')
    grid = _grid(config, metadata['layers'])
    grid = grid.loc[grid.operator.ne('Unsteered')].copy()
    (output_dir / 'cells').mkdir(exist_ok=True)
    grid['path'] = grid.path.map(lambda path: output_dir / 'cells' / (path.name + '.gz'))
    paths = {tuple(row[:-1]): row[-1] for row in grid.itertuples(index=False, name=None)}
    columns = KEYS + ['trial', 'seed', 'completion', 'concept', 'coherent', 'both']
    source = pd.read_csv(source_path, keep_default_na=False)[columns]
    source = source.merge(grid[KEYS], on=KEYS, validate='many_to_one').sort_values(KEYS + ['trial'])
    counts = source.groupby(KEYS).size()
    if len(counts) != len(grid) or not counts.eq(config.trials).all():
        raise ValueError('Saved run does not contain the requested replay conditions and sample counts')
    identity = KEYS + ['trial', 'seed']
    source_index = pd.MultiIndex.from_frame(source[identity])
    if source.duplicated(KEYS + ['trial']).any():
        raise ValueError('Saved run contains duplicate trials')
    csv_options = dict(keep_default_na=False,
                       na_values={name: [''] for name in ('change', 'along_v', 'along_fraction', 'off_fraction')})
    if reuse is not None:
        reuse = Path(reuse)
        previous = json.loads((reuse / 'config.json').read_text())
        if any(previous[key] != settings[key] for key in ('source_sha256', 'sweep', 'protocol')):
            raise ValueError(f'{reuse} describes a different source run or replay protocol')
        imported = pd.read_csv(reuse / 'results.csv.gz', **csv_options)
        imported = imported.merge(source[identity], on=identity, validate='one_to_one')
        for key, cell in imported.groupby(KEYS):
            if not paths[key].exists():
                _save(cell, paths[key])
    cached = [pd.read_csv(path, **csv_options) for path in grid.path if path.exists()]
    if cached:
        cached = pd.concat(cached, ignore_index=True)
        cached_index = pd.MultiIndex.from_frame(cached[identity])
        if cached_index.has_duplicates or not cached_index.isin(source_index).all():
            raise ValueError('Replay checkpoints contain duplicate or unexpected trials')
        pending = source.loc[~source_index.isin(cached_index)]
    else:
        pending = source
    del cached
    if not pending.empty:
        for name in (f'E_{config.concept}_n{config.direction_samples}_seed{config.seed}.pt',
                     f'A_{config.concept}_n{config.jacobian_samples}_ctx{config.max_ctx}_seed{config.seed}.pt'):
            if not (config.cache_dir / name).exists():
                raise FileNotFoundError(f'Replay requires the existing cache: {config.cache_dir / name}')
        tokenizer, model, E, operators = _prepare(config)
        nominal, controller = None, None
        try:
            if pending.drift_weight.ne(0).any():
                _, nominal = load_model(config.model_name, dtype=config.dtype, device_map=config.device_map)
            prompt_ids = tokenizer(config.prompt)['input_ids']
            with tqdm(total=len(source), initial=len(source) - len(pending),
                      desc='Replaying alignment', unit='completion') as progress:
                for (N, R, name, weight), cells in pending.groupby(KEYS[:-1], sort=False):
                    controller = (QwenLQRSteering(
                        model, operators[name], E, config.q, R, config.qf, steer_last_n_layers=N)
                        if weight == 0 else DriftLQRSteering(
                            model, nominal, operators[name], E, config.q, R, config.qf,
                            drift_weight=weight, steer_last_n_layers=N))
                    for strength, cell in cells.groupby('lambda', sort=False):
                        progress.set_postfix(N=N, R=R, operator=name, w=weight, lmbda=strength)
                        path = paths[N, R, name, weight, strength]
                        rows = pd.read_csv(path, **csv_options).to_dict('records') if path.exists() else []
                        for start in range(0, len(cell), config.trace_batch):
                            batch = cell.iloc[start:start + config.trace_batch]
                            continuations = tokenizer(batch.completion.tolist(), add_special_tokens=False)['input_ids']
                            lengths = torch.tensor([len(ids) for ids in continuations], device=model.device)
                            width = max(1, max(map(len, continuations)))
                            sequences = torch.full((len(batch), len(prompt_ids) + width), tokenizer.eos_token_id,
                                                   dtype=torch.long, device=model.device)
                            for i, ids in enumerate(continuations):
                                tokens = prompt_ids + ids
                                sequences[i, :len(tokens)] = torch.tensor(tokens, device=model.device)
                            replay = trace(model, controller, strength, sequences, len(prompt_ids))
                            replay['valid'] = replay['valid'] & (torch.arange(width, device=model.device)[None, :] < lengths[:, None])
                            values = alignment_metrics(replay)
                            rows.extend(record | value for record, value in zip(batch.to_dict('records'), values))
                            del replay
                            _save(pd.DataFrame(rows), path)
                            progress.update(len(batch))
                    controller = None
        finally:
            del controller, model, nominal, operators
            gc.collect()
            torch.cuda.empty_cache()
    results = pd.concat([pd.read_csv(path, **csv_options) for path in grid.path], ignore_index=True)
    _save(results, output_dir / 'results.csv.gz')
    metrics = ['change', 'along_v', 'along_fraction', 'off_fraction', 'concept', 'coherent', 'both']
    summary = results.groupby(KEYS).agg(
        **{name: (name, 'mean') for name in metrics},
        off_fraction_se=('off_fraction', 'sem'), n=('off_fraction', 'count'), samples=('trial', 'size')).reset_index()
    _save(summary, output_dir / 'summary.csv')
    return summary
