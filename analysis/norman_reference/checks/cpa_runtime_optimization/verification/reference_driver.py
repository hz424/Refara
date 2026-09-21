#!/usr/bin/env python3
"""Train fixed CPA on admitted Norman conditions; predict from separate NTC cells.

Input expression must already be whole-transcriptome CP10K-log1p normalized
and subset to the frozen training-control feature panel. This driver does
not select genes, read observed test responses, or compute benchmark scores.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

MODEL = dict(n_latent=128, recon_loss='gauss', doser_type='logsigm',
             n_hidden_encoder=256, n_layers_encoder=3,
             n_hidden_decoder=256, n_layers_decoder=3,
             n_hidden_doser=128, n_layers_doser=2,
             dropout_rate_encoder=0., dropout_rate_decoder=0.,
             variational=False, seed=17,
             use_batch_norm_encoder=True, use_layer_norm_encoder=False,
             use_batch_norm_decoder=True, use_layer_norm_decoder=False)
PLAN = dict(lr=1e-4, wd=1e-6, n_steps_pretrain_ae=None,
            n_epochs_pretrain_ae=5, n_steps_kl_warmup=None,
            n_epochs_kl_warmup=None, n_steps_adv_warmup=None,
            n_epochs_adv_warmup=10, n_epochs_mixup_warmup=0,
            n_epochs_verbose=10, mixup_alpha=0., adv_steps=3,
            reg_adv=1., pen_adv=1., n_hidden_adv=64, n_layers_adv=2,
            use_batch_norm_adv=False, use_layer_norm_adv=False,
            dropout_rate_adv=.1, adv_lr=1e-4, adv_wd=1e-7,
            doser_lr=1e-4, doser_wd=1e-7, step_size_lr=45,
            do_clip_grad=False, gradient_clip_value=1., adv_loss='cce')
EPOCHS = 200
BATCH_SIZE = 512


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def runtime():
    names = ('numpy', 'torch', 'cpa-tools', 'scvi-tools', 'pytorch-lightning', 'anndata')
    return {name: importlib.metadata.version(name) for name in names}


def set_seed():
    import numpy as np
    import torch
    import scvi
    import pytorch_lightning as pl
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    scvi.settings.seed = 17
    pl.seed_everything(17, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def check_adata(adata, condition_key):
    import numpy as np
    from scipy import sparse
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell and feature identifiers must be unique')
    if condition_key not in adata.obs:
        raise ValueError('Missing condition labels')
    labels = adata.obs[condition_key].astype(str)
    if labels.isin(['', 'nan', 'None']).any():
        raise ValueError('Missing condition identity')
    for label in labels.unique():
        terms = label.split('+')
        if len(terms) > 2 or len(set(terms)) != len(terms):
            raise ValueError('Only singles and distinct two-gene combinations admitted')
        if len(terms) > 1 and 'ctrl' in terms:
            raise ValueError('Singles must be GENE, not GENE+ctrl')
    data = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError('Input must be finite nonnegative CP10K-log1p values')
    if data.size and data.max() > np.log1p(10000) + 1e-4:
        raise ValueError('Input exceeds whole-transcriptome CP10K-log1p bound')


def build_model(adata, condition_key):
    import cpa
    adata.obs['cpa_condition'] = adata.obs[condition_key].astype(str)
    adata.obs['cpa_dose'] = adata.obs['cpa_condition'].map(
        lambda x: '+'.join(['1.0'] * len(x.split('+'))))
    adata.obs['cpa_split'] = 'train'
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None
    cpa.CPA.setup_anndata(adata, perturbation_key='cpa_condition',
        dosage_key='cpa_dose', control_group='ctrl', batch_key=None,
        categorical_covariate_keys=[], is_count_data=False, max_comb_len=2)
    return cpa.CPA(adata, split_key='cpa_split', train_split='train',
        valid_split='__absent_validation__', test_split='__absent_test__', **MODEL)


def fit(adata, condition_key, output, protocol_sha, use_gpu, epochs=EPOCHS, smoke=False, input_sha="synthetic"):
    import numpy as np
    import pandas as pd
    import torch
    import pytorch_lightning as pl
    from cpa._data import AnnDataSplitter
    from cpa._task import CPATrainingPlan
    from cpa._utils import CPA_REGISTRY_KEYS
    from scvi.train import TrainRunner
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'state.json').exists():
        raise FileExistsError('Refusing to replace completed CPA state')
    set_seed()
    check_adata(adata, condition_key)
    labels = adata.obs[condition_key].astype(str)
    if 'ctrl' not in set(labels):
        raise ValueError('Training controls required')
    if not smoke and adata.n_vars != 2000:
        raise ValueError('Frozen production panel contains exactly 2000 features')
    if not smoke and not use_gpu:
        raise RuntimeError('Production CPA training requires an allocated CUDA GPU')
    model = build_model(adata, condition_key)
    if len(model.valid_indices) or len(model.test_indices):
        raise AssertionError('Unexpected validation/test cells in training object')
    mapping = model.adata_manager.registry['field_registries'][
        CPA_REGISTRY_KEYS.PERTURBATION_KEY]['state_registry']['categorical_mapping']
    weights = [(adata.n_obs / int((labels == str(c)).sum())) - 1. for c in mapping]
    splitter = AnnDataSplitter(model.adata_manager,
        train_indices=model.train_indices, valid_indices=model.valid_indices,
        test_indices=model.test_indices, batch_size=BATCH_SIZE, use_gpu=use_gpu)
    plan = CPATrainingPlan(model.module, model.covars_encoder,
        n_adv_perts=len(mapping), drug_weights=weights, **PLAN)
    class Progress(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            self.started = time.monotonic()

        def on_train_epoch_end(self, trainer, pl_module):
            progress = dict(completed_epochs=int(trainer.current_epoch) + 1,
                total_epochs=epochs, elapsed_seconds=time.monotonic() - self.started)
            with (output/'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(progress) + '\n')
            if progress['completed_epochs'] % 10 == 0 or progress['completed_epochs'] == 1:
                print(json.dumps(progress), flush=True)

    runner = TrainRunner(model, training_plan=plan, data_splitter=splitter,
        max_epochs=epochs, use_gpu=use_gpu, early_stopping=False,
        enable_checkpointing=False, enable_progress_bar=False,
        simple_progress_bar=False, enable_model_summary=False,
        logger=False, benchmark=False, deterministic=True,
        num_sanity_val_steps=0, check_val_every_n_epoch=None,
        limit_val_batches=0, log_every_n_steps=1, callbacks=[Progress()])
    start = time.monotonic()
    runner()
    elapsed = time.monotonic() - start
    history = pd.DataFrame(dict(plan.epoch_history))
    if history['epoch'].tolist() != list(range(epochs)):
        raise AssertionError('Did not complete every fixed epoch')
    if not history['mode'].eq('train').all():
        raise AssertionError('Unexpected non-training history')
    if not np.isfinite(history['recon_loss']).all():
        raise FloatingPointError('Nonfinite reconstruction loss')
    state = {key: value.detach().cpu() for key, value in model.module.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError('Nonfinite checkpoint tensor')
    torch.save(state, output / 'state.pt')
    history.to_csv(output / 'training_history.tsv', sep='\t', index=False)
    pd.DataFrame({'cell_id': adata.obs_names}).to_csv(output / 'train_cells.tsv', sep='\t', index=False)
    pd.DataFrame({'feature_id': adata.var_names}).to_csv(output / 'genes.tsv', sep='\t', index=False)
    labels.value_counts().rename_axis('condition').rename('n_train_cells').to_csv(
        output / 'training_conditions.tsv', sep='\t')
    record = dict(native_output='state', input_scale='whole-transcriptome CP10K-log1p',
        gene_count=adata.n_vars, train_cell_count=adata.n_obs,
        condition_key=condition_key, perturbation_encoder=model.pert_encoder,
        covariate_encoder=model.covars_encoder, model=MODEL, plan=PLAN,
        epochs=epochs, seed=17, batch_size=BATCH_SIZE,
        training_seconds=elapsed, smoke_test=smoke,
        validation_rows=0, test_rows=0, state_selection='fixed terminal epoch',
        protocol_sha256=protocol_sha, training_input_sha256=input_sha, runtime=runtime(),
        script_sha256=sha256(__file__), state_sha256=sha256(output/'state.pt'),
        gene_axis_sha256=sha256(output/'genes.tsv'),
        train_cells_sha256=sha256(output/'train_cells.tsv'))
    write_json(output/'state.json', record)
    return model, record


def predict(state_dir, controls_path, tasks_path, output, batch_size=1024):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import torch
    from cpa._module import CPAModule
    state_dir, output = Path(state_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = json.loads((state_dir/'state.json').read_text())
    for filename, key in [('state.pt', 'state_sha256'), ('genes.tsv', 'gene_axis_sha256'), ('train_cells.tsv', 'train_cells_sha256')]:
        if sha256(state_dir/filename) != record[key]:
            raise ValueError(f'Frozen model artifact hash differs: {filename}')
    controls = ad.read_h5ad(controls_path)
    check_adata(controls, record['condition_key'])
    if not controls.obs[record['condition_key']].astype(str).eq('ctrl').all():
        raise ValueError('Inference object may contain only non-targeting controls')
    genes = pd.read_csv(state_dir/'genes.tsv', sep='\t', dtype=str)['feature_id'].tolist()
    if controls.var_names.astype(str).tolist() != genes:
        raise ValueError('Control and model feature axes differ')
    train_cells = set(pd.read_csv(state_dir/'train_cells.tsv', sep='\t', dtype=str)['cell_id'])
    if train_cells.intersection(controls.obs_names):
        raise ValueError('Inference controls overlap optimizer-training cells')
    tasks = pd.read_csv(tasks_path, sep='\t', dtype=str)
    if 'condition' not in tasks or tasks['condition'].duplicated().any():
        raise ValueError('Tasks need a unique condition column')
    train_conditions = set(pd.read_csv(state_dir/'training_conditions.tsv', sep='\t')['condition'])
    encoder = record['perturbation_encoder']
    for condition in tasks['condition']:
        terms = condition.split('+')
        if len(terms) != 2 or len(set(terms)) != 2 or 'ctrl' in terms:
            raise ValueError('Every test condition must be a distinct two-gene combination')
        if condition in train_conditions or '+'.join(reversed(terms)) in train_conditions:
            raise ValueError('Test combination occurred in training')
        if any(term not in train_conditions for term in terms):
            raise ValueError('Every constituent must have a seen single-gene condition')
        if any(term not in encoder for term in terms):
            raise ValueError('Constituent absent from fitted CPA encoder')
    module = CPAModule(n_genes=len(genes), n_perts=len(encoder),
        covars_encoder=record['covariate_encoder'], **record['model'])
    module.load_state_dict(torch.load(state_dir/'state.pt', map_location='cpu', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device).eval()
    X = controls.X
    latent = np.empty((controls.n_obs, record['model']['n_latent']), dtype=np.float32)
    native_check_max_abs = 0.
    with torch.inference_mode():
        for start in range(0, controls.n_obs, batch_size):
            slab = X[start:start+batch_size]
            slab = slab.toarray() if sparse.issparse(slab) else np.asarray(slab)
            latent[start:start+len(slab)] = module.encoder(
                torch.as_tensor(slab, dtype=torch.float32, device=device)).cpu().numpy()
        outputs = []
        for task_index, condition in enumerate(tasks['condition']):
            indices = torch.tensor([[encoder[t] for t in condition.split('+')]], dtype=torch.long, device=device)
            doses = torch.ones((1, 2), dtype=torch.float32, device=device)
            shift = module.pert_network(indices, doses)
            filename = f'state_task_{task_index:04d}.npy'
            destination = output/filename
            if destination.exists():
                raise FileExistsError(destination)
            prediction = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32,
                shape=(controls.n_obs, len(genes)))
            for start in range(0, controls.n_obs, batch_size):
                z = torch.as_tensor(latent[start:start+batch_size], device=device)
                y = module.generative(z+shift)['px'].loc.cpu().numpy()
                if not np.isfinite(y).all():
                    raise FloatingPointError('Nonfinite CPA state prediction')
                prediction[start:start+len(y)] = y
            n_check = min(16, controls.n_obs)
            x_check = X[:n_check]
            x_check = x_check.toarray() if sparse.issparse(x_check) else np.asarray(x_check)
            full = module.inference(torch.as_tensor(x_check, dtype=torch.float32, device=device),
                {'true':indices.repeat(n_check,1)}, {'true':doses.repeat(n_check,1)}, {})
            native = module.generative(full['z'])['px'].loc.cpu().numpy()
            cached = np.asarray(prediction[:n_check])
            native_check_max_abs = max(native_check_max_abs, float(np.max(np.abs(native-cached))))
            if not np.allclose(native, cached, rtol=2e-5, atol=2e-6):
                raise AssertionError('Cached/native CPA inference disagreement')
            prediction.flush()
            del prediction
            outputs.append(dict(task_index=task_index, condition=condition,
                file=filename, sha256=sha256(destination)))
    pd.DataFrame(outputs).to_csv(output/'predictions.tsv', sep='\t', index=False)
    pd.DataFrame({'cell_id':controls.obs_names}).to_csv(output/'control_cells.tsv', sep='\t', index=False)
    pd.DataFrame({'feature_id':genes}).to_csv(output/'genes.tsv', sep='\t', index=False)
    write_json(output/'prediction_record.json', dict(native_output='state',
        conditioning='separate NTC expression plus query combination identity',
        n_tasks=len(tasks), n_controls=controls.n_obs, n_genes=len(genes),
        checkpoint_sha256=record['state_sha256'], controls_sha256=sha256(controls_path),
        tasks_sha256=sha256(tasks_path), script_sha256=sha256(__file__),
        training_control_overlap=0, treated_cells_used_as_input=0,
        output_scale=record['input_scale'], runtime=runtime(),
        native_inference_check_max_abs=native_check_max_abs,
        native_inference_check_controls_per_task=min(16,controls.n_obs)))


def smoke(output):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from cpa._utils import CPA_REGISTRY_KEYS
    output = Path(output)
    rng = np.random.default_rng(991)
    labels = np.repeat(['ctrl','A','B','C','A+B'],64)
    adata = ad.AnnData(np.log1p(rng.poisson(2., (len(labels),32))).astype('float32'),
        obs=pd.DataFrame({'condition':labels}, index=[f'train_{i}' for i in range(len(labels))]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(32)]))
    model, record = fit(adata,'condition',output/'state','synthetic',False,epochs=2,smoke=True)
    controls = adata[:17].copy()
    controls.obs_names = [f'control_{i}' for i in range(17)]
    # Remove the training registry: these are independent synthetic inputs.
    controls = ad.AnnData(controls.X, obs=controls.obs[['condition']].copy(),var=controls.var.copy())
    controls.write_h5ad(output/'controls.h5ad')
    pd.DataFrame({'condition':['B+C']}).to_csv(output/'tasks.tsv',sep='\t',index=False)
    predict(output/'state',output/'controls.h5ad',output/'tasks.tsv',output/'predictions',8)
    # Compare cached encode/add/decode against CPA's native inference+generative path.
    x = torch.as_tensor(controls.X, dtype=torch.float32)
    idx = torch.tensor([[model.pert_encoder['B'],model.pert_encoder['C']]]*len(x),dtype=torch.long)
    dose = torch.ones((len(x),2),dtype=torch.float32)
    model.module.cpu().eval()
    with torch.inference_mode():
        inf = model.module.inference(x, {'true':idx}, {'true':dose}, {})
        native = model.module.generative(inf['z'])['px'].loc.numpy()
    cached = np.load(output/'predictions/state_task_0000.npy')
    maxerr=float(np.max(np.abs(native-cached)))
    if not np.allclose(native,cached,rtol=2e-5,atol=2e-6):
        raise AssertionError(f'Cached/native inference disagreement {maxerr}')
    baseline = np.asarray(controls.X.mean(axis=0), dtype=np.float64)
    state = cached.astype(np.float64).mean(axis=0)
    target = rng.normal(size=32)
    effect=state-baseline
    residual_error=float(np.max(np.abs((state-target)-(effect-(target-baseline)))))
    if residual_error > 1e-12:
        raise AssertionError('Consistent state/effect conversion failed')
    write_json(output/'smoke.json',dict(status='PASS', native_vs_cached_max_abs=maxerr,
        equivalent_residual_max_abs=residual_error, training_rows=len(labels),
        test_constituents_seen_as_singles=True, test_combination_seen_in_training=False,
        production_data_used=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    train=subs.add_parser('train')
    train.add_argument('--input',required=True,type=Path)
    train.add_argument('--protocol',required=True,type=Path)
    train.add_argument('--condition-key',default='condition')
    train.add_argument('--output',required=True,type=Path)
    pred=subs.add_parser('predict')
    pred.add_argument('--state',required=True,type=Path)
    pred.add_argument('--controls',required=True,type=Path)
    pred.add_argument('--tasks',required=True,type=Path)
    pred.add_argument('--output',required=True,type=Path)
    pred.add_argument('--batch-size',default=1024,type=int)
    sm=subs.add_parser('smoke')
    sm.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.command=='train':
        import anndata as ad
        import torch
        if not args.protocol.is_file():
            raise FileNotFoundError('Source-free protocol must be frozen before training')
        fit(ad.read_h5ad(args.input),args.condition_key,args.output,
            sha256(args.protocol),torch.cuda.is_available(), input_sha=sha256(args.input))
    elif args.command=='predict':
        predict(args.state,args.controls,args.tasks,args.output,args.batch_size)
    else:
        smoke(args.output)

if __name__=='__main__':
    main()
