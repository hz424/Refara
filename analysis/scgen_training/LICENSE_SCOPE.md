# Licence scope

The reconstruction, rendering and verification programs are BSD-3-Clause;
see [LICENSE](LICENSE). They implement the fixed inference equations with
PyTorch operators and import the original scGen runtime only for optional
parity checks. Bundled fonts retain the licence in `fonts/LICENSE_LIBERATION`.

The separate confidential companion contains GSE162632-derived prepared
expression, results, fitted states and execution records. Its data and
checkpoint terms are separate from this software licence. It is supplied for
journal review, not as a public data deposit. Retain the source study attribution
and notices accompanying those inputs.

The historical runtime uses scGen commit
`d79e1f04233c30f9a4eb5b8d57718127909807d7` with the `qzm`/`qzv` compatibility patch.
The upstream licence file states GPL-3.0, while its package metadata states MIT.
Those notices and the patch are recorded in
`capsules/gse162632_scgen/environment/THIRD_PARTY.md` at the repository root.
The repository does not vendor or relicense scGen.
