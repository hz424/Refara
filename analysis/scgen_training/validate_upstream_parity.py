#!/usr/bin/env python3
"""Optional exact CPU parity check against the original patched scGen runtime.

The main reconstruction entry does not need this runtime. This check requires
scGen 2.1.1, scvi-tools 0.20.3 and their original compatibility patch; provide
the package directory with --scgen-source. It checks every supplied CV state.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from reconstruct_checkpoint_risks import (BoundCompanion, EPOCHS, RATES, InferenceState,
                                           load_fold, predict_risks, save_json)
from review_companion import file_sha, module_from, require

VAE_SHA = '00348b640b5ea36355c4cdf335a89f74621e9d4d231b2d855c1190905441a527'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--companion-dir', type=Path, required=True)
    p.add_argument('--scgen-source', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    require(file_sha(args.scgen_source / 'scgen/_scgenvae.py') == VAE_SHA, 'Original patched scGen source differs')
    sys.path.insert(0, str(args.scgen_source))
    from scgen._scgenvae import SCGENVAE
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    bound = BoundCompanion(args.companion_dir)
    original = module_from(bound.path('04_CODE/train_scgen_robustness.py'), 'original_training_diagnostic')
    records = []
    for fold in (f'fold{i:02}' for i in range(1, 8)):
        arrays = load_fold(bound, fold)
        for lr in RATES:
            for epoch in EPOCHS:
                path = bound.path(f'05_TRAINING_RECORDS/runs/cv/{fold}_lr{lr}/checkpoints/epoch{epoch:03}.ckpt')
                portable = InferenceState(path, 'cpu')
                module = SCGENVAE(n_input=2000, n_hidden=512, n_latent=100, n_layers=2, dropout_rate=.2, kl_weight=.00005)
                module.load_state_dict(portable.state, strict=True)
                module.eval()
                with torch.inference_mode():
                    latent_original, _ = original.encode(module, arrays['X_train'], torch)
                    latent_portable = portable.batches(arrays['X_train'], portable.encode_tensor)
                    require(np.array_equal(latent_original, latent_portable), 'Encoder outputs differ')
                    validation_latent, _ = original.encode(module, arrays['X_val'], torch)
                    require(np.array_equal(validation_latent, portable.batches(arrays['X_val'], portable.encode_tensor)),
                            'Validation encoder outputs differ')
                    decoded_original = original.decode(module, validation_latent, torch)
                    decoded_portable = portable.batches(validation_latent, portable.decode_tensor)
                    require(np.array_equal(decoded_original, decoded_portable), 'Decoder outputs differ')
                    expected, _ = original.diagnostic(module, arrays, torch)
                    actual = predict_risks(portable, arrays)
                maximum = max(abs(a['risk'] - e['risk']) for a, e in zip(actual['per_donor_task'], expected['per_donor_task']))
                require(maximum == 0., 'Donor/task prediction risk differs')
                records.append({'fold': fold, 'learning_rate': lr, 'epoch': epoch,
                                'training_encoder_max_abs_error': 0., 'validation_encoder_max_abs_error': 0.,
                                'validation_decoder_max_abs_error': 0., 'donor_task_risk_max_abs_error': maximum})
                print(json.dumps({'complete': len(records), **records[-1]}), flush=True)
    save_json(args.out_dir / 'UPSTREAM_EXACT_PARITY_RECEIPT.json', {
        'status': 'PASS_EXACT_UPSTREAM_INFERENCE_PARITY_ALL_42_CHECKPOINTS', 'checkpoint_count': len(records),
        'device': 'cpu', 'torch': torch.__version__, 'numpy': np.__version__,
        'upstream_vae_sha256': VAE_SHA, 'archived_diagnostic_sha256': file_sha(bound.path('04_CODE/train_scgen_robustness.py')),
        'portable_inference_sha256': file_sha(Path(__file__).with_name('reconstruct_checkpoint_risks.py')),
        'validation_script_sha256': file_sha(Path(__file__)), 'records': records,
        'comparison': 'Array equality for every training/validation posterior mean and validation decoded state; exact equality for all 1200 donor/task risks.'})


if __name__ == '__main__':
    main()
