# Linear history decoding

Sheet: 64×64; padding: circular; readout: both.
Inputs: Bernoulli(0.5) × N(0,1) × 0.025, every 4 steps.
Samples: 16384 train / 4096 validation / 4096 test per configuration and seed.
Independent trajectories per split. Ridge chosen on validation; test R² averaged across input channels.
Delay 0 is measured immediately after injection; delay -1 is the future-input control.
Tables show mean across seeds. Full scores and standard deviations are in CSV files.

| K | rho | lag 0 | lag 1 | lag 4 | lag 8 | lag 16 | lag 32 | lag 64 | future |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 | 0.099 | -0.011 |
| 64 | 0.99 | 1.000 | 0.998 | 0.997 | 0.996 | 0.993 | 0.974 | 0.777 | -0.010 |
| 64 | 0.995 | 0.985 | 0.961 | 0.955 | 0.948 | 0.929 | 0.870 | 0.656 | -0.009 |
| 64 | 1 | 0.274 | 0.275 | 0.269 | 0.261 | 0.247 | 0.220 | 0.181 | -0.007 |
| 128 | 0.975 | 1.000 | 1.000 | 0.999 | 0.998 | 0.988 | 0.800 | -0.010 | -0.010 |
| 128 | 0.99 | 0.985 | 0.958 | 0.945 | 0.926 | 0.869 | 0.649 | 0.032 | -0.009 |
| 128 | 0.995 | 0.859 | 0.799 | 0.776 | 0.745 | 0.675 | 0.499 | 0.108 | -0.008 |
| 128 | 1 | 0.131 | 0.170 | 0.171 | 0.164 | 0.151 | 0.130 | 0.091 | -0.006 |
| 256 | 0.975 | 0.995 | 0.980 | 0.957 | 0.907 | 0.696 | -0.001 | -0.009 | -0.009 |
| 256 | 0.99 | 0.852 | 0.782 | 0.733 | 0.660 | 0.488 | 0.108 | -0.005 | -0.008 |
| 256 | 0.995 | 0.562 | 0.537 | 0.498 | 0.445 | 0.339 | 0.133 | 0.005 | -0.007 |
| 256 | 1 | 0.085 | 0.108 | 0.107 | 0.099 | 0.086 | 0.062 | 0.024 | -0.006 |
