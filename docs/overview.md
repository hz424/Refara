# How control cells enter a model comparison

![Control cells used as model inputs and scoring baselines, and the condition for a ranking reversal.](assets/figure-1.png)

A perturbation effect is measured relative to untreated cells. These control
cells can serve three purposes: a model may use them as input, their mean
expression may be subtracted from predicted expression to obtain a predicted
effect, and they provide the baseline for measuring the observed effect.

Refara's primary comparison, O, uses one control block for model inputs
and the baseline of a predicted treated state, and a separate block for the
observed effect. A direct-effect prediction `d` and its equivalent state
representation `C_model + d` receive the same O score: subtracting `C_model`
recovers `d`. This keeps the comparison consistent when the same prediction is
expressed in different forms.

When the same control mean is subtracted from predicted and observed expression,
it cancels from their difference. The mean squared error between the effects is
then identical to the error between the original expression values. Separate
control means can change that error, and with it the ranking of models that
predict effects directly and models that predict treated expression. Panel c
shows when a ranking reverses in the balanced comparison.

To compare different input controls, use predictions generated from each set of
cells. If only the scoring references change, you can reuse the model's output
and recalculate the effects and scores. The [operation guide](REFERENCE_AUDIT.md)
walks through these changes.

The three-block design in the figure is used to examine each control role
separately. [Choosing references](choosing-references.md) explains the available
designs and the assumptions behind the ranking comparison. Specify the roles,
budgets and sampling strata, then compare all declared allocations to identify
which conclusions are sensitive to these choices.
