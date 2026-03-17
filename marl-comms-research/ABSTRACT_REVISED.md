Title:
Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit

Authors:
U. NJOKU(1) and J. CALDERON(1).

Affiliation:
(1) Dept. of Computer Science, Bethune-Cookman Univ., 640 Dr. Mary McLeod
Bethune Blvd., Daytona Beach, FL 32114.

Text:
Partial observability makes multi-agent coordination difficult because each agent must act on
incomplete local information while still contributing to a coherent team strategy. Pursuit
domains provide a simple but visually intuitive testbed for this problem: multiple pursuers must
coordinate to capture an evasive target despite limited fields of view and map-dependent
constraints on movement. This study investigates whether small learned discrete communication
channels improve coordination in a partially observable multi-agent pursuit task, and whether the
resulting messages are interpretable at the symbol level.

We study three conditions in a grid-world pursuit environment with three pursuers and one prey:
no explicit communication, a 4-symbol communication channel, and a 16-symbol communication
channel. All models are trained with shared-parameter deep Q-learning on a fixed training map
and evaluated both in-distribution and on unseen transfer maps. To reduce sensitivity to late
training degradation, model selection is based on a uniform best-checkpoint sweep rather than
final-checkpoint evaluation alone. Across matched multi-seed runs, all conditions solve the
training map, indicating that explicit communication is not necessary for strong in-distribution
performance on the easiest layout.

The main benefit of communication appears in generalization. Communication-enabled agents
achieve higher mean capture rates and lower variance on unseen maps than the no-communication
baseline, with the 16-symbol channel producing the most reliable transfer performance. These
results suggest that learned messages primarily help agents coordinate under novel spatial
configurations rather than dramatically improving asymptotic performance on the training map.
At the same time, interpretability analysis shows that the learned channels are used extensively
but do not form clean human-readable symbol systems: both communication conditions exhibit
near-maximum entropy, minimal inter-agent specialization, little temporal structure, and no
reliable symbol-level success/failure correlation on the easiest map. The emergent protocol is
therefore functional but distributed, supporting coordination without yielding simple discrete
semantic categories.

Overall, the results indicate that compact learned communication can improve robustness in
partially observable multi-agent pursuit, but that increased channel capacity matters for transfer
and the resulting protocols may resist straightforward symbolic interpretation.

Keywords:
multi-agent reinforcement learning, emergent communication, cooperative pursuit,
partial observability, transfer generalization, discrete communication, interpretability,
decentralized coordination, grid-world environments
