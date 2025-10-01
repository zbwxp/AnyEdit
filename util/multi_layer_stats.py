import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm

from util.nethook import TraceDict
from util.globals import *  # STATS_DIR if needed by callers
from util.runningstats import (
    CombinedStat,
    Mean,
    NormMean,
    SecondMoment,
    save_cached_state,
    load_cached_state,
    make_loader,
    FixedSubsetSampler,
)
from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def _infer_lengths(model, batch_tokens: Optional[int]):
    # Infer maxlen and npos from model.config similar to util/layer_stats
    if hasattr(model.config, "n_positions"):
        maxlen = npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        maxlen = npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        maxlen = npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        maxlen = npos = model.config.seq_length
    else:
        raise NotImplementedError

    # Model specific overrides
    if hasattr(model.config, "model_type") and "mistral" in model.config.model_type:
        if hasattr(model.config, "sliding_window") and model.config.sliding_window:
            maxlen = npos = model.config.sliding_window or 4096
        else:
            maxlen = npos = 4096
    if hasattr(model.config, "model_type") and "qwen2" in model.config.model_type:
        maxlen = npos = 4096

    if batch_tokens is not None and batch_tokens < maxlen:
        maxlen = batch_tokens
    return maxlen, npos


def _get_dataset(model, tokenizer, ds_name: str, batch_tokens: Optional[int]):
    # Mirror util/layer_stats.py: load from local disk paths
    path_map = {
        "wikipedia": "/root/code/AnyEdit/wiki/wikipedia-20220301.en.110k",
        "wikitext": "/root/code/AnyEdit/wiki/wikitext-103-raw-v1",
    }
    raw_ds = load_from_disk(path_map["wikipedia"])  # currently always wikipedia
    maxlen, _ = _infer_lengths(model, batch_tokens)
    return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)


def _file_for(stats_dir: Path, model_name: str, ds_name: str, layer_name: str, precision: str, to_collect: List[str], size_suffix: str) -> Path:
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    return stats_dir / file_extension


def multi_layer_stats(
    model,
    tokenizer,
    layer_names: List[str],
    stats_dir: str,
    ds_name: str,
    to_collect: List[str] = ("mom2",),
    sample_size: Optional[int] = None,
    precision: Optional[str] = None,
    batch_tokens: Optional[int] = None,
    batch_size: int = 100,
    force_recompute: bool = False,
    progress=tqdm,
) -> Dict[str, CombinedStat]:
    """
    Collect statistics for multiple layers in a single dataset pass.
    Returns a dict: layer_name -> CombinedStat
    Periodically saves per-layer NPZ caches and supports resume.
    """
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    # Determine token budget suffix
    _, npos = _infer_lengths(model, batch_tokens)
    if batch_tokens is None:
        batch_tokens = npos * 3
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix

    # Model name normalization
    model_name = model.config._name_or_path.rsplit("/")[-1]
    stats_dir = Path(stats_dir)

    # Dataset and loader
    ds = _get_dataset(model, tokenizer, ds_name, batch_tokens)
    total_size = (sample_size or len(ds))

    # Prepare per-layer stats and filenames
    layer_stats: Dict[str, CombinedStat] = {
        ln: CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect}) for ln in layer_names
    }
    filenames: Dict[str, Path] = {
        ln: _file_for(stats_dir, model_name, ds_name, ln, precision, to_collect, size_suffix)
        for ln in layer_names
    }

    # Load checkpoints and determine a consistent resume point
    processed_per = {ln: 0 for ln in layer_names}
    if not force_recompute:
        for ln in layer_names:
            ckpt = load_cached_state(str(filenames[ln]), args={}, quiet=True)
            if ckpt is not None:
                try:
                    layer_stats[ln].load_state_dict(ckpt)
                    processed_per[ln] = int(ckpt.get("processed", 0))
                except Exception:
                    processed_per[ln] = 0

    min_processed = min(processed_per.values()) if len(processed_per) else 0
    max_processed = max(processed_per.values()) if len(processed_per) else 0

    # If all layers complete, short-circuit
    if max_processed >= total_size and not force_recompute:
        if progress is None:
            pass
        else:
            print("All layers already cached. Returning loaded stats.")
        return layer_stats

    # If mismatch across layers, reset those ahead to the minimum to avoid double count
    if any(v != min_processed for v in processed_per.values()):
        print(
            f"Warning: resume mismatch across layers: {processed_per}. "
            f"Resetting advanced layers to min={min_processed} to avoid double-count."
        )
        for ln, v in processed_per.items():
            if v != min_processed:
                layer_stats[ln] = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
                processed_per[ln] = min_processed

    start_index = min_processed

    # Build loader for remaining range
    sampler = FixedSubsetSampler(list(range(start_index, total_size)))
    loader = make_loader(
        ds,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        num_workers=2,
    )

    # Move stats to CUDA to match feature device
    for st in layer_stats.values():
        try:
            st.to_("cuda")
        except Exception:
            pass

    remaining = total_size - start_index
    batch_count = -(-remaining // batch_size)
    save_every_groups = 50

    with torch.no_grad():
        group_idx = 0
        processed = start_index
        for batch_group in progress(loader, total=batch_count):
            group_idx += 1
            group_n = 0
            for batch in batch_group:
                group_n += int(batch["input_ids"].shape[0])
                batch = dict_to_(batch, "cuda")
                with TraceDict(
                    model,
                    layers=layer_names,
                    retain_input=True,
                    retain_output=False,
                    stop=True,
                ) as tr:
                    _ = model(**batch)
                for ln in layer_names:
                    feats = flatten_masked_batch(tr[ln].input, batch["attention_mask"]).to(dtype)
                    layer_stats[ln].add(feats)
            processed += group_n

            if (group_idx % save_every_groups == 0) or (processed >= total_size):
                for ln in layer_names:
                    try:
                        save_cached_state(str(filenames[ln]), layer_stats[ln], {"processed": processed, "total": total_size})
                        print(f"Saved checkpoint for {ln} at processed={processed}")
                    except Exception as e:
                        print(f"Warning: periodic save failed for {ln} at processed={processed}: {e}")

    # Final save
    for ln in layer_names:
        try:
            save_cached_state(str(filenames[ln]), layer_stats[ln], {"processed": total_size, "total": total_size})
        except Exception as e:
            print(f"Warning: final save failed for {ln}: {e}")

    return layer_stats

