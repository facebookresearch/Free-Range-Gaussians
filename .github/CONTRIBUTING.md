# Contributing

We welcome bug reports, documentation improvements, and inference fixes.
Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Issues

For a bug report, include the command, software and CUDA versions, error output,
and a small reproducible scene where possible. Share `run.json` when reporting
reconstruction behavior. Keep access tokens and model weights out of issues.

## Pull requests

The project's source of truth is an internal Meta repository. Community changes
are reviewed on GitHub, then imported and landed internally by maintainers.
ShipIt exports the accepted changes back to this repository.

1. Fork the repository and create a branch from `main`.
2. Read the [README](../README.md), [usage notes](../docs/USAGE.md), and
   [agent instructions](../AGENTS.md).
3. Keep changes focused and update the relevant documentation.
4. Validate a bundled scene. For inference changes, include a CUDA example and
   describe the outputs, settings, and any differences in behavior.
5. Open a pull request with a description of the change and its verification.

An accepted pull request may be closed when its changes are exported from the
internal repository.

Use the released sampler defaults, including photometric guidance 50, and keep
checkpoint compatibility unless a method change has been discussed with the
maintainers. Use a black background for comparable renders.

## Contributor License Agreement

Complete Meta's [Contributor License Agreement](https://code.facebook.com/cla)
when requested by the contribution checks. This is required to accept external
contributions.

## License

Code contributions are licensed under [MIT](../LICENSE). Model weights and
outputs retain their separate [FAIR Noncommercial Research License](../MODEL_LICENSE).
