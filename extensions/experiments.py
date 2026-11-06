"""Cached operators, resumable generation, and judging for notebooks 02–03."""

from dataclasses import asdict, dataclass
from itertools import product
import gc
import json
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import ROOT

KEYS = ['N', 'R', 'every', 'operator', 'drift_weight', 'lambda']
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
    every: tuple = (1,)
    operators: tuple = ('Identity', 'Mean Jacobian', 'Random', 'Steering DMD')
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
    from .dmd import steering_dmd
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
    dmd_path = config.cache_dir / (f'DMD_{config.concept}_n{config.direction_samples}_seed{config.seed}'
                                   f'_q{config.q:g}_r10_qf{config.qf:g}.pt')
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
    if 'Steering DMD' in config.operators:
        reload_model = not dmd_path.exists() and next(model.parameters()).dtype != torch.float32
        if reload_model:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            tokenizer, model = load_model(config.model_name, dtype=torch.float32, device_map=config.device_map)
        cached = steering_dmd(dmd_path, model, tokenizer, E, config.concept, q=config.q, r=10, qf=config.qf)
        operator = torch.empty(layers, hidden, hidden)
        for layer, (left, basis) in enumerate(cached['factors']):
            torch.addmm(identity, left, basis.T, out=operator[layer])
        operators['Steering DMD'] = operator
        if reload_model:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            tokenizer, model = load_model(config.model_name, dtype=config.dtype, device_map=config.device_map)
    return tokenizer, model, E, {name: operators[name] for name in config.operators}


def _grid(config, layers):
    scopes = list(dict.fromkeys(layers if n is None else n for n in config.last_n))
    grid = pd.DataFrame(product(scopes, config.r_values, config.every, config.operators,
                                config.drift_weights, config.lambdas), columns=KEYS)
    baseline = pd.DataFrame([[0, 0.0, 0, 'Unsteered', 0.0, 0.0]], columns=KEYS)
    grid = pd.concat([baseline, grid], ignore_index=True)
    grid['path'] = [config.run_dir / 'cells' / f'n{n}_r{r:g}_m{m}_{op.lower().replace(" ", "_")}_w{w:g}_l{strength:g}.csv'
                    for n, r, m, op, w, strength in grid.itertuples(index=False, name=None)]
    return grid


def _collect(config, tokenizer, model, controller, settings, progress):
    import torch
    from .steering import sample, trace, completion_metrics

    seed = config.seed + (config.lambdas.index(settings['lambda']) if controller is not None else 0)
    torch.manual_seed(seed)
    prompt_length = len(tokenizer(config.prompt)['input_ids'])
    rows = []
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
                metrics = completion_metrics(replay)
                del replay
            texts = tokenizer.batch_decode(batch[:, prompt_length:], skip_special_tokens=True)
            rows.extend(settings | values | dict(trial=start + offset + i, seed=seed, completion=text.strip())
                        for i, (text, values) in enumerate(zip(texts, metrics)))
        progress.update(len(ids))
    return pd.DataFrame(rows)


def generate_run(config):
    """Resume completed conditions; never load models when the final results already exist."""
    metadata = _metadata(config)
    if metadata and any((config.run_dir / name).exists() for name in ('summary.csv', 'results.csv.gz')):
        return
    grid = _grid(config, metadata['layers']) if metadata else None
    if grid is not None and grid.path.map(lambda path: path.exists()).all():
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
        pending = grid.loc[~grid.path.map(lambda path: path.exists())]
        if pending.drift_weight.ne(0).any():
            _, nominal = load_model(config.model_name, dtype=config.dtype, device_map=config.device_map)
        with tqdm(total=len(grid) * config.trials, initial=(len(grid) - len(pending)) * config.trials,
                  desc='Generating completions', unit='completion') as progress:
            for values, cells in pending.groupby(KEYS[:-1], sort=False):
                N, R, every, name, weight = values
                if name == 'Unsteered':
                    controller = None
                else:
                    kwargs = dict(every=int(every), steer_last_n_layers=int(N))
                    controller = (QwenLQRSteering(model, operators[name], E, config.q, R, config.qf, **kwargs)
                                  if weight == 0 else DriftLQRSteering(
                                      model, nominal, operators[name], E, config.q, R, config.qf,
                                      drift_weight=weight, **kwargs))
                for row in cells[KEYS + ['path']].itertuples(index=False, name=None):
                    settings, path = dict(zip(KEYS, row[:-1])), row[-1]
                    progress.set_postfix(N=N, R=R, m=every, operator=name, w=weight, lmbda=settings['lambda'])
                    _save(_collect(config, tokenizer, model, controller, settings, progress), path)
                controller = None
    finally:
        del controller, model, nominal, operators
        gc.collect()
        torch.cuda.empty_cache()


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
