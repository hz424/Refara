# Norman training and reference comparisons

The CPA labels in `figure_source`, `reference_results` and `direct_effect_results` denote the validation-selected Gaussian checkpoint: per-epoch scheduling, adversarial weight 10 and epoch 400. It was fitted once to all 68,002 original training cells.

`validation_summary.tsv` retains all candidate and secondary validation results. The common-target files contain the original and selected CPA, additive, compositional ridge and no-change predictions evaluated with B1 input/prediction centring and B2 observation centring. This task differs from the complete S/M/P/O/D comparison.

The reference tables retain 55 tasks, 30 allocations and six depths. Direct-effect predictions, baseline states, targets, features, scales and memberships are fixed. The old/new files preserve both CPA procedures; original model results remain in the preceding source release. Capture partitions and allocations describe one pooled experiment.
