# Reference-operation audit

Use the planner to check whether a change in control cells requires different
predictions or a new score for existing predictions. The planner compares
reference declarations; run the scorer separately to evaluate the predictions.

```sh
reference-design audit plan before.json after.json --output outputs/operation_plan
reference-design audit demo --output outputs/operation_demo
```

Use an absent or empty output directory. The plan writes proposed operations to
`AUDIT.json` with `executed: false`. The synthetic demo executes the operations
on toy predictions, checks that equivalent representations preserve scores,
and rejects mismatched input identities and corrupted arrays.
To replay the empirical results in `evidence/table1/`, use
`evidence/replay_table1.py`.

<a id="declare-the-quantity-and-its-dependencies"></a>

## Describe the prediction

For a model that predicts treated expression from control block B1, with the
observed effect measured against B2, use:

```json
{
  "task": "reference_effect",
  "output_kind": "state",
  "input_conditioned": true,
  "model_input_id": "B1",
  "prediction_reference_id": "B1",
  "observation_reference_id": "B2",
  "storage_encoding": "state",
  "storage_baseline_id": null
}
```

`task` is `reference_effect` or `treated_state`; `output_kind` is `state` or
`native_effect`. Storage can be `state` or `effect`. A storage baseline is
required when storage differs from the native output kind. A native effect
rendered as a treated state additionally needs `native_state_baseline_id`.
Inactive role IDs may be omitted or null. Unknown fields are rejected, including
misspelled role names. Optional `metric_coordinate_id` and
`candidate_coordinate_id` currently support only `aligned_state`.

| Change | Prediction operation | Scoring operation |
| --- | --- | --- |
| Model input used by a predictor | Select or generate the prediction for the new input | Score that prediction under the declared references |
| Prediction reference for a state-derived effect | Reuse the model-native state | Re-centre it and rescore |
| Observation reference for an effect target | Reuse the model-native output | Rebuild the observed effect and rescore |
| Storage representation or storage baseline | Encode/decode the same native output consistently | Preserve the score when the underlying prediction and target are unchanged |
| Prediction reference for a native effect | Reuse the emitted effect | No prediction centring |
| Unused model-input role | Reuse the prediction | No change from that role alone |

For an input-conditioned native effect, select the effect predicted from the
new input; no prediction baseline is subtracted. Declare a new model if its fit
or input dependence changes. Feature scales, metric centre and retrieval
candidates stay fixed in this planner.

The Python functions `plan_change`, `encode_prediction`, `decode_prediction`
and `canonical_components` are available from `reference_design`. Encoding
records the prediction digest and the actual storage-baseline digest. Decoding
rejects changed baseline values even if the baseline ID is unchanged.
`canonical_components` returns the prediction, target, residual and scores.
For effect targets, optional correlation and retrieval metrics compare aligned
states. Supply their metric centre and candidate collection; unavailable metrics
remain NaN. These metrics cannot be substituted with calculations on arbitrary
effect coordinates. MSE and RMSE are calculated separately for each fit.

<a id="optional-generation-records-consumed-by-the-scorer"></a>

## Check a prediction's generation record

Add `"generation_record": "generation.json"` to a model in a protocol plan
or legacy `score` manifest to check its generation record before writing scores.
The check uses the selected prediction, reference and membership paths.
Without this field, the scorer relies on the model declaration.

A record has exactly these top-level fields:

```json
{
  "schema": "reference_design.generation_record.v1",
  "model": {
    "name": "conditioned_state", "kind": "state",
    "conditioning": "block", "representation": "native"
  },
  "artifacts": {
    "prediction": {"path": "conditioned.tsv", "sha256": "<actual SHA256>"},
    "controls": {"path": "controls.tsv", "sha256": "<actual SHA256>"},
    "membership": {"path": "membership.tsv", "sha256": "<actual SHA256>"},
    "references": {"path": "references.tsv", "sha256": "<actual SHA256>"},
    "baseline": null
  },
  "input_keys": [["0", 1, "B1"], ["0", 1, "B2"]],
  "checkpoint": {"path": "checkpoint.bin", "sha256": "<actual SHA256>"},
  "code": [{"path": "generate.py", "sha256": "<actual SHA256>"}]
}
```

Paths resolve relative to the record. List every allocation/depth/block key in
your reference table; the example above shows only a subset. Model context and
selected paths must match the scoring inputs. Every listed file, including the
checkpoint and generation code, must exist and match its hash. For a
deterministic prediction rule, supply the saved rule as the checkpoint. The
check compares file bytes without interpreting the checkpoint format.

Block-conditioned records require controls, membership and reference artifacts.
The normal scorer also verifies membership and reconstructed reference means.
Fixed models use an empty `input_keys` list. Artifact fields are null only when
that artifact is absent from the scoring invocation; an active representation
baseline must be bound. Missing or stale input, membership, prediction,
checkpoint, code or baseline files stop scoring. The recorded paths must match those
used for scoring, even if another file has identical contents.

<a id="three-distinct-evidence-levels"></a>

## What each check establishes

| Level | What was checked |
| --- | --- |
| `declaration` | Supplied prediction semantics; no generation record |
| `generation_record_binding` | The selected files, model context and complete keys match a supplied generation record |
| `independently_recomputed` | A trusted in-process callback was executed and its numeric predictions exactly matched the supplied predictions |

A matching record confirms the supplied files and declarations. Establishing
which inputs the original model used, how training and evaluation were separated,
and whether models had equal access to information requires execution records.
`verified_input_use` therefore remains false even when recomputation matches the
supplied predictions.

Only Python callers can request the third level:

```python
from reference_design import verify_generation_record

evidence = verify_generation_record(
    record_path, prediction_path=prediction_path, model_name=model_name,
    kind="state", conditioning="block", representation="native",
    controls_path=controls_path, membership_path=membership_path,
    references_path=references_path, expected_input_keys=complete_keys,
    recompute=trusted_independent_predictor, expected_predictions=scored_values,
)
```

`trusted_independent_predictor(record)` must return numeric arrays, or a complete
mapping of keys to arrays. The application chooses and trusts this callback;
the record cannot supply code to execute, a PASS flag or a verification level.
Shapes, keys and finite numeric values must match exactly, and recorded files
must remain unchanged during the call. Choose an independent callback that
recomputes predictions rather than returning the supplied arrays.
