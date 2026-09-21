# Third-party runtime

The container checks out scGen commit
`d79e1f04233c30f9a4eb5b8d57718127909807d7` and applies the bundled
`qzm`/`qzv` compatibility patch before installation. No scGen source or fitted
checkpoint is redistributed in this repository. The licence file at that
commit states GPL-3.0, whereas its package metadata states MIT; users should
consult the upstream repository before redistribution. Applying the patch does
not change those upstream terms.

`requirements-cu117.txt` records the complete dependency snapshot from the
qualified Python 3.10.17 environment. The container installs those exact
versions and verifies the patched scGen source file before building the
package.
