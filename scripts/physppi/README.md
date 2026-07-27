# PhysPPI Operations

These scripts are the supported command-line entry points for PhysPPI dataset
screening, semantic training configuration, launch, monitoring, and checkpoint
validation.

Run each command from the repository root with the OpenFold CUDA environment:

```bash
.pixi/envs/openfold3-cuda12/bin/python scripts/physppi/<command>.py --help
```

Reusable screening logic lives in
`openfold3.projects.of3_all_atom.physppi.screening`. Command wrappers may
import that package; production package code must never import `scripts`.
