## Directory summary
WGSL shader source for sparse public-tree CFR. Modules are grouped by solver phase so production, terminal, and all-in table kernels can be reviewed independently.

### Source files
- `core.ts`: Regret matching, belief/reach propagation, average policy, backup, gather, and scatter kernels.
- `opponent.ts`: Opponent-policy and regret-weight construction kernels, including aggregate variants.
- `terminal.ts`: Showdown rank aggregation and terminal value kernels.
- `allIn.ts`: All-in table value kernels and optimized 1326-hand variants.
