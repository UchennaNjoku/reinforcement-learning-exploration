## Multi-Seed Summary Table (Best-Checkpoint Selection)

Trained on `easy_open`. Evaluated seed 99, 200 episodes per condition per seed.

Checkpoint selected per run: highest greedy capture rate on easy_open, then lowest avg_steps as tiebreaker.

Values shown as mean ± std across 3 seeds.


| Condition | Map | Capture Rate | Avg Steps | Collision Rate | N seeds |
|-----------|-----|:------------:|:---------:|:--------------:|:-------:|
| No-Comm | easy_open | 100.0% ±0.0 | 9.6 ±0.9 | 0.1% ±0.0 | 3 |
| No-Comm | center_block | 77.7% ±28.8 | 121.1 ±97.6 | 12.3% ±15.9 | 3 |
| No-Comm | split_barrier | 65.8% ±26.2 | 173.1 ±78.8 | 38.1% ±34.6 | 3 |
|  |  |  |  |  |  |
| Comm-4 | easy_open | 100.0% ±0.0 | 9.7 ±0.3 | 0.0% ±0.0 | 3 |
| Comm-4 | center_block | 91.0% ±12.7 | 73.7 ±67.2 | 0.5% ±0.5 | 3 |
| Comm-4 | split_barrier | 70.7% ±18.3 | 159.8 ±66.6 | 4.1% ±4.0 | 3 |
|  |  |  |  |  |  |
| Comm-16 | easy_open | 100.0% ±0.0 | 9.0 ±0.2 | 0.0% ±0.0 | 3 |
| Comm-16 | center_block | 95.2% ±6.1 | 63.6 ±50.3 | 0.9% ±1.0 | 3 |
| Comm-16 | split_barrier | 79.5% ±12.7 | 139.4 ±56.1 | 3.7% ±2.7 | 3 |
|  |  |  |  |  |  |

---

## Checkpoint Selection Details

| Condition | Subdir | Selected Ep | Cap (easy_open) | Avg Steps |
|-----------|--------|:-----------:|:---------------:|:---------:|
| No-Comm | baseline_v3 | ep004000 | 100.0% | 9.1 |
| No-Comm | baseline_s1 | ep005000 | 100.0% | 10.8 |
| Comm-4 | comm4_v2 | ep004000 | 100.0% | 9.3 |
| Comm-4 | comm4_s1 | ep003500 | 100.0% | 10.1 |
| Comm-4 | comm4_s2 | ep001500 | 100.0% | 9.8 |
| Comm-16 | comm16_v2 | ep003500 | 100.0% | 9.3 |
| Comm-16 | comm16_s1 | ep004500 | 100.0% | 8.9 |
| Comm-16 | comm16_s2 | ep004000 | 100.0% | 8.8 |
| No-Comm | baseline_s3 | ep005000 | 100.0% | 8.8 |
