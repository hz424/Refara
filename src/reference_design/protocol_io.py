"""Strict two- or three-block readers for the declared protocol."""
from pathlib import Path
from .io import BlockTable, _columns, _numbers, _key, align, integer, table


def read_protocol_blocks(path: Path, genes: tuple[str, ...]) -> BlockTable:
    headers, rows = table(path)
    columns = _columns(headers, {'allocation', 'depth', 'block'}, {'cells_per_block'}, path)
    values = align(_numbers(rows, columns, path), columns, genes, path)
    keys = [_key(row, path) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{path}: duplicate allocation/depth/block rows')
    groups = {key[:2] for key in keys}
    block_sets = {tuple(sorted(k[2] for k in keys if k[:2] == group)) for group in groups}
    if len(block_sets) != 1 or next(iter(block_sets)) not in [('B1',), ('B2',), ('B1', 'B2'), ('B1', 'B2', 'B3')]:
        raise ValueError(f'{path}: every group needs the same supported block set (B1, B2, B1/B2 or B1/B2/B3)')
    counts = None
    if 'cells_per_block' in headers:
        counts = {key: integer(row['cells_per_block'], 'Cell count', 1) for key, row in zip(keys, rows)}
        for group in groups:
            sizes = {count for key, count in counts.items() if key[:2] == group}
            if len(sizes) != 1 or next(iter(sizes)) % group[1]:
                raise ValueError(f'{path}: inconsistent block sizes or stratum depth')
    return BlockTable(genes, dict(zip(keys, values)), counts)
