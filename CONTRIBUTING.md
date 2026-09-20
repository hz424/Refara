# Contributing

Follow the [installation instructions](README.md#install), then run
`python -m pytest -q` in that environment. Use small synthetic inputs when
testing changes to scoring or input checks.

A bug report should include the software and Python versions, the command you
ran, the first error message, what you expected and a small input that others
can use. For a score discrepancy, also include the prediction target, output
type, control roles, gene scaling and aggregation. A run receipt can help;
remove private paths or data before sharing it.

For a scoring fix, include an independently calculated example that fails
before the change and passes afterward. Keep the primary comparison, reference
sensitivity checks and official VCC metrics distinct. When input formats change,
update the examples and guides with the tests.

Changes to scientific inputs or expected results need a scientific explanation
and updated source records. Keep source hashes and licences with the data.
