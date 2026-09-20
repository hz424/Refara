# Independent verification

The final score cube, summaries and reader example passed independent checks. The metric and real-data checkers reconstruct predictions from the original bound arrays and use separate score formulas. They do not call the production input adapter or scoring implementation.

| Check | Coverage | Largest absolute difference |
|---|---|---:|
| Raw Norman metric sample | 6,480 prediction rows; 25,920 metric values | 1.78 × 10⁻¹⁵ |
| Complete metric interfaces | All previous direct-state and paired-effect scores; 297,000 balanced MSE rows | Included in the bound metric receipt |
| Summary reconstruction | 432 role contrasts, 432 interactions, 864 pattern means and 288 workflow rows | 5.55 × 10⁻¹⁶ |
| Action contract and numerical checks | 384 factorial cases, 3,456 simultaneous-change flags, six task switches and twelve rejected pipeline-contract changes | 8.88 × 10⁻¹⁶ |
| Complete reader example | 95 operations; 1,520 saved before/after components | 9.99 × 10⁻¹⁶ |

The 29 arrays bundled with the reader example match the original sources exactly. Its operations cover 42 Norman cases, 51 PBMC cases and two Kang cases. CellFlow retains all five fits, with losses computed separately before averaging. All 285 omitted-role checks agree with an independent dependency table: 156 omissions remove active information, while 129 omit an unused role and remain valid.

The scalar centroid checks agree exactly. The complete metric audit also checks all six permutations of the three block labels, fixed-output and native-effect controls, and cancellation of a common scoring centre for state outputs. Every one of the 10,368 reported descriptive gap counts agrees with the independent contrast enumeration.

One development refinement separated a task-type change from a changed finite observation reference or native-effect state mapping. The original factorial case selection was retained; `ACTION_API_REFINEMENT.json` records the revised field meanings. `DEVELOPMENT_ACTION_CHECK.json` preserves the first-pass receipt. The final checker also verifies that equal scores do not conceal a changed observed target, and that nonfinite retrieval candidates are rejected.

Final receipts are `INDEPENDENT_METRIC_CHECK.json`, `INDEPENDENT_SUMMARY_CHECK.json`, `INDEPENDENT_ACTION_CHECK.json` and `INDEPENDENT_REAL_ACTION_CHECK.json`. The final action module hash is `ba362112577d6fd3bd452634b4c97280b85e3861994a5e91390beee06c922aea`. The real-data receipt binds the contents of the final demonstration by hash, independently of its directory name.

These checks establish computational and declared-action correctness. The example uses retrospective cached data; it does not measure external organizer error rates, human task time or an information advantage over the same Systema scores with explicit manual role reasoning. Raw-source QA requires the original workspace caches. The bundled reader example can be replayed separately without those caches.
