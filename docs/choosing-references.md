# Choosing references for your prediction task

Decide what the model should predict and which information it will have at deployment before choosing control references.

| Role | How the reference is used | When it applies |
|---|---|---|
| Model input (`Cmodel`) | Supplies untreated cells or a control summary to the predictor | The predictor uses controls at prediction time |
| Prediction centring (`Cpred`) | Converts a predicted treated state into a response by subtracting a baseline | A state prediction is scored as an effect |
| Observation centring (`Cobs`) | Defines the measured response as treated expression minus a baseline | The endpoint is a measured effect |

These roles can share cells. A two-block design can pair model input with prediction centring and keep observation controls separate. The five-pattern diagnostic uses three blocks to vary these assignments.

<a id="start-with-the-delivered-prediction"></a>

## Start with what the model predicts

For a **treated-state target**, compare predicted and measured treated expression directly. The model can still use controls as input; effect centring is unnecessary.

For a **fresh-reference effect target**, measure the response against a separately sampled control block. When donor controls are available to the model, use the first block for its input and prediction baseline, and the second for the observed response.

For a **shared-reference effect target**, use the same controls for observation and prediction centring. This evaluates the response relative to that baseline. Choose this target to match the application, before inspecting model rankings.

Native effects are already on the response scale. Subtract a prediction baseline from a state to score it as an effect. If the saved state already had a baseline subtracted, supply that baseline to restore the state first.

Either output type can use control inputs or be input-free. Primary scoring and the balanced diagnostic accept fixed or block-conditioned predictions of both kinds. Match conditioned predictions to their input block; native effects receive no prediction-centring subtraction.

## Match the available information

Record the donor controls, covariates and external references available at prediction time. Give all models the same permitted information, even if they use different subsets, and retain the input-block ID with each prediction.

Evaluator-held controls can measure the observed effect even when the model has no donor controls. They cannot be used to produce its prediction. An input-free state model can be scored directly on treated expression. To return donor-specific effects at deployment, it needs an available baseline or a baseline prediction evaluated as part of the pipeline. Measuring an effect also requires a justified observation baseline.

Split training, validation and evaluation for the intended generalization, such as new donors or perturbations. Fit features and scales on permitted training data; select models and checkpoints on validation data. Fix reference assignments before held-out evaluation and record any choices informed by evaluation results.

## Choose blocks and depth

Choose feasible depths and report strata and weights. Equal weighting of strata can yield a different target mixture from pooling cells. Prespecify several depths to assess depth sensitivity.

A two-block fresh-reference comparison needs enough cells for both blocks; the balanced S/M/P/O/D diagnostic needs three equally sized blocks in each stratum. The allocator samples disjoint blocks and records membership. Repeated allocations measure sensitivity within the supplied cell pool.

Use `reference-design allocate --blocks 2` for a two-block plan or `--blocks 3` for the balanced diagnostic. The AnnData manifest offers the same choice. Each stratum needs `blocks × depth` controls; two-block comparisons remain possible when three blocks are infeasible.

Partially overlapping blocks, unequal depths and other weights can support other comparisons, but the balanced identity below does not cover them.

## Interpret the comparison

Utility is negative mean squared error: a positive pairwise margin favours the first named model. Fix task and acquisition-unit weights before inspecting results, and compare the full chosen model family on the same tasks.

In the balanced three-block diagnostic, `d_S` and `d_D` compare fixed native-effect predictions with state predictions under shared and fully separated references: effect-model utility minus state-model utility. They satisfy `d_D = d_S + V`, with `V` the nonnegative reference-distance term. A strict reversal occurs when `-V < d_S < 0`. Keep treated observations, features, scales and native effects fixed; supply state predictions for each input block; score balanced assignments before averaging.

The software scores input-conditioned native effects descriptively and marks the theorem unavailable for those pairs. With control means `(0, 1, 2)` and treated mean zero, `E(Cmodel)=Cmodel` has S-MSE `20/3` and D-MSE `14/3`: changing input/reference pairing changes its score. Retraining, other losses and other weights need separate analysis. Choose a practically meaningful difference separately from the numerical threshold for floating-point ties.

To interpret expected squared-error differences as differences against the underlying biological effect, observation error must have zero conditional mean given that effect and the compared predictions. Disjoint cells alone are insufficient: donor structure, capture effects and finite-pool sampling also matter.

Inference needs justified acquisition units and a procedure for the intended hypothesis. Tasks, programs, repeated fits and control allocations may share information. The [statistical interpretation](STATISTICAL_INTERPRETATION.md) and [unit-level comparison guide](DESIGN_AUDIT_CONTRACT.md) explain the available procedures.

## Keep the result interpretable

Save the target, deployment inputs, what each output represents, baseline sources, block memberships, depths and strata weights with the predictions. Record training and selection history, all compared models, aggregation weights and the acquisition units used for inference. For biological programs, identify the measurement reference and record how the model was selected as well as how its predictions were scored.

The [reference-design guide](REFERENCE_DESIGN.md) gives file formats and commands. The [paper reproduction guide](REPRODUCING.md) lists analysis inputs and figure replays.
