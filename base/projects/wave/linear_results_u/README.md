# Linear history decoding

Sheet: 64×64; padding: circular; readout: u.
Inputs: Bernoulli(0.5) × N(0,1) × 0.025, every 4 steps.
Samples: 16384 train / 4096 validation / 4096 test per configuration and seed.
Independent trajectories per split. Ridge chosen on validation; test R² averaged across input channels.
Delay 0 is measured immediately after injection; delay -1 is the future-input control.
Tables show mean across seeds. Full scores and standard deviations are in CSV files.

| K | rho | lag 0 | lag 1 | lag 4 | lag 8 | lag 16 | lag 32 | lag 64 | future |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.975 | 1.000 | 1.000 | 0.999 | 0.998 | 0.990 | 0.833 | -0.005 | -0.005 |
| 64 | 0.99 | 0.985 | 0.963 | 0.955 | 0.939 | 0.885 | 0.710 | 0.245 | -0.005 |
| 64 | 0.995 | 0.907 | 0.832 | 0.812 | 0.787 | 0.720 | 0.590 | 0.290 | -0.004 |
| 64 | 1 | 0.283 | 0.171 | 0.151 | 0.148 | 0.141 | 0.131 | 0.107 | -0.003 |
| 128 | 0.975 | 0.993 | 0.980 | 0.966 | 0.927 | 0.735 | 0.219 | -0.005 | -0.005 |
| 128 | 0.99 | 0.904 | 0.824 | 0.781 | 0.723 | 0.576 | 0.286 | -0.002 | -0.004 |
| 128 | 0.995 | 0.730 | 0.605 | 0.553 | 0.512 | 0.420 | 0.256 | 0.024 | -0.004 |
| 128 | 1 | 0.170 | 0.115 | 0.097 | 0.094 | 0.087 | 0.075 | 0.050 | -0.003 |
| 256 | 0.975 | 0.939 | 0.862 | 0.776 | 0.621 | 0.273 | -0.003 | -0.004 | -0.004 |
| 256 | 0.99 | 0.721 | 0.583 | 0.499 | 0.414 | 0.246 | 0.025 | -0.002 | -0.004 |
| 256 | 0.995 | 0.499 | 0.361 | 0.294 | 0.251 | 0.166 | 0.067 | 0.003 | -0.003 |
| 256 | 1 | 0.107 | 0.073 | 0.057 | 0.054 | 0.046 | 0.033 | 0.010 | -0.003 |
