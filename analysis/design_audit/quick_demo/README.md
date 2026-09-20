# Synthetic quick demo

This example uses synthetic, pre-aggregated utilities for two methods and six
declared acquisition units. The values test the audit and do not represent
either method's performance.

With shared references, `METHOD_A` ranks above `METHOD_B`. When the three
reference roles are separated, the ranking reverses. A second configuration
uses the same values but omits an independence justification; the audit then
returns `WITHHELD` instead of a directional result.

From the repository root, run:

```bash
bash scripts/run_demo.sh
```

The command needs Python 3.10 but no external data or Python packages. It
creates a temporary virtual environment and writes the verified output bundles
to the directory printed at the end.
