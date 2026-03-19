# PPT Script Draft 2

## Title

**Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit**

Target length: 15 minutes  
Target audience: mixed / bio-heavy, not necessarily deep RL specialists  
Core thesis:

> The main value of communication in this project was not better performance on the easiest known map, but better robustness when the agents had to generalize to new layouts.

---

## Slide 1 — Title

### On the slide

- Title
- Your name
- Bethune-Cookman University
- Simple grid-world visual or one pursuit screenshot

### Script

“Good [morning/afternoon], everyone. My name is Uchenna Njoku, and today I’ll be presenting my work on emergent communication in multi-agent reinforcement learning.”

“In simple terms, I studied whether a team of artificial agents can learn to communicate with each other in order to coordinate more effectively when none of them can see the full environment.”

“The short version of the talk is this: communication did not matter much on the easiest training map, but it became more useful when the agents had to generalize to new, unseen maps.”

“So the real story is not just whether communication works, but where it helps and what kind of communication the agents actually learned.”

---

## Slide 2 — Motivation

### On the slide

Title: **Why study communication in multi-agent systems?**

- real systems often have only local information
- team success depends on coordination under uncertainty
- communication has bandwidth and latency costs
- we want useful, compact learned communication

Possible icons:

- drones
- robots
- autonomous vehicles
- distributed sensing systems

### Script

“The motivation for this work comes from a practical problem in distributed intelligent systems.”

“In many real-world settings, there is not one agent with perfect information making all decisions. Instead, there are multiple agents, each with only a local view, trying to act together.”

“Examples include teams of drones searching a disaster site, warehouse robots sharing space, autonomous vehicles coordinating locally, or distributed sensors monitoring a changing environment.”

“In all of those cases, three things are usually true.”

“First, each agent sees only part of the world. Second, success depends on coordination, not just individual intelligence. And third, communication is not free. It has bandwidth limits, delays, and practical costs.”

“So the question becomes: can agents learn a compact communication strategy that improves teamwork without requiring a hand-designed protocol?”

---

## Slide 3 — Real-world framing

### On the slide

Title: **Why this matters beyond a grid world**

- search-and-rescue robotics
- swarm coordination in uncertain environments
- distributed autonomy under incomplete information

Highlighted line:

**Communication matters most when the environment changes or becomes ambiguous**

### Script

“One reason this question matters is that the most interesting real-world environments are not static and perfectly predictable.”

“Imagine a search-and-rescue scenario. One robot sees part of the environment because of debris or walls. Another sees a different region. No single robot has the full picture.”

“Or imagine a fleet of warehouse robots. Each knows its own local obstacles, but efficient routing depends on what nearby teammates are doing.”

“So this project is not really about a toy game. It is about a broader scientific question: how do teams stay effective when each member has incomplete information?”

“My results suggest that communication may matter less when the task is already easy and familiar, and more when the environment changes and the agents need to adapt.”

---

## Slide 4 — Task setup

### On the slide

Title: **Task setup**

- 3 pursuers
- 1 evasive prey
- each pursuer sees only a 7x7 local window
- goal: capture the prey efficiently

Visual:

- grid world with one agent’s local field of view shown

### Script

“To study this question in a controlled way, I used a pursuit task in a grid-world environment.”

“There are three pursuer agents and one prey.”

“The prey follows an evasive policy, meaning it tries to move away from nearby pursuers.”

“Each pursuer sees only a local 7 by 7 patch around itself rather than the full grid. That property is called partial observability.”

“Partial observability simply means the agent does not have complete state information. It has to make decisions from an incomplete local view.”

“So even if one pursuer can see the prey, it may not know where its teammates are or what they are about to do.”

“The team objective is to capture the prey as efficiently and consistently as possible.”

---

## Slide 5 — Research questions

### On the slide

Title: **Research questions**

- Can learned communication improve coordination under partial observability?
- Does it help mainly on the training map, or on transfer to unseen maps?
- Are the learned messages interpretable?

### Script

“From that setup, I focused on three main questions.”

“First: can the agents learn a useful communication strategy under partial observability?”

“Second: if communication helps, where does it help most? Does it improve performance on the training environment, or does it mainly help when the environment changes?”

“And third: if the agents do learn to communicate, can we interpret those messages in a clean, human-readable way?”

“That last question is important because successful communication is not automatically the same thing as understandable communication.”

“A system may rely on internal signals that are highly functional, but still not easy for humans to interpret.”

---

## Slide 6 — Experimental conditions

### On the slide

Title: **Three conditions**

| Condition | Description |
|-----------|-------------|
| No-Comm | agents act only on local observations |
| Comm-4 | agents send one of 4 symbols |
| Comm-16 | agents send one of 16 symbols |

Footer note:

- same training framework across conditions

### Script

“I compared three experimental conditions.”

“The first had no explicit communication. Agents could only use their own observations.”

“The second allowed each agent to send one of four possible symbols each step.”

“The third expanded that communication channel to sixteen possible symbols.”

“Importantly, the training framework was kept as consistent as possible across all three conditions, so the main thing changing was the communication capacity.”

“This lets us ask not only whether communication helps, but whether a larger communication channel helps more.”

---

## Slide 7 — Method in plain language

### On the slide

Title: **How the agents learn**

Diagram:

- Observation -> neural network -> movement action
- For comm conditions: Observation + received messages -> neural network -> movement action + outgoing message

Bullets:

- deep Q-learning
- shared parameters across pursuers
- messages received at the next step

### Script

“I used a reinforcement learning method called deep Q-learning.”

“Reinforcement learning means the agents learn by trial and error from rewards and penalties.”

“The neural network estimates which actions are likely to lead to better long-term outcomes.”

“All three pursuers share the same underlying model. That is called parameter sharing.”

“Parameter sharing means the agents learn from pooled experience instead of each starting from scratch with a separate model.”

“In the communication conditions, the messages sent by teammates at one time step become part of the receiving agent’s input at the next step.”

“So communication becomes one more source of information the agent can use when choosing its next move.”

---

## Slide 8 — Environment and evaluation design

### On the slide

Title: **Environment design and evaluation**

- trained on one fixed map: `easy_open`
- transfer tested on unseen maps: `center_block`, `split_barrier`
- greedy evaluation with best-checkpoint selection
- 3 matched seeds per condition

Optional tiny note:

- checkpoint chosen by best greedy capture on `easy_open`

### Script

“All models were trained on the same simple training map called `easy_open`.”

“Then I evaluated them on unseen transfer maps called `center_block` and `split_barrier`.”

“Evaluation was done greedily, meaning no exploration noise was added during testing.”

“One important detail is checkpoint selection.”

“Instead of always using the final checkpoint, I swept the saved checkpoints for each run and selected the one with the highest greedy capture rate on the training map, using average steps as a tie-breaker.”

“That was important because some runs degraded late in training, and final-checkpoint-only comparisons turned out to be misleading.”

“I used three matched seeds per condition for the main comparison.”

---

## Slide 9 — Training-map result

### On the slide

Title: **Result 1: all conditions solve the training map**

Show:

- training curve figure or a small easy-open comparison table

Main takeaway text:

**Communication was not necessary for strong in-distribution performance**

### Script

“The first main result is that all three conditions solved the training map.”

“With best-checkpoint selection, all three reached 100 percent greedy capture on `easy_open`, with similar average step counts, roughly 9 to 11 steps.”

“That means I cannot honestly claim that communication was necessary to solve the easiest environment.”

“And that is actually useful, because it keeps the interpretation honest.”

“Communication is not being credited for something that the no-communication baseline can already do well.”

“So if communication is valuable here, the more interesting question becomes: where does it add value beyond the easiest familiar case?”

---

## Slide 10 — Main quantitative result: transfer

### On the slide

Title: **Result 2: communication helps transfer**

Use the main table, simplified:

| Condition | easy_open | center_block | split_barrier |
|-----------|-----------|--------------|---------------|
| No-Comm | 83.8% ± 22.9 | 68.5% ± 44.5 | 67.0% ± 45.3 |
| Comm-4 | 100% ± 0 | 91.0% ± 12.7 | 70.7% ± 18.3 |
| Comm-16 | 100% ± 0 | 95.2% ± 6.1 | 79.5% ± 12.7 |

Big highlight:

**Comm-16 transfers most reliably**

### Script

“The clearest effect of communication appeared on the unseen transfer maps.”

“On the training map, all conditions were strong. But on `center_block` and `split_barrier`, the communication-enabled agents had better mean transfer performance and much lower variability than the no-communication baseline.”

“The strongest condition overall was `Comm-16`.”

“On `center_block`, `Comm-16` reached about 95 percent capture on average. On `split_barrier`, it reached about 80 percent.”

“By comparison, the no-communication baseline was much more unstable across seeds.”

“So the main value of communication in this project was not better asymptotic performance on the easiest known map. It was better generalization and robustness when the map changed.”

Pause.

“That is the central empirical finding of the project.”

---

## Slide 11 — Interpreting the transfer result

### On the slide

Title: **What does the transfer result mean?**

- training map can be solved without explicit communication
- unseen maps expose coordination brittleness
- communication helps maintain performance under distribution shift
- larger channel size helps more than smaller channel size

### Script

“One way to interpret this is that on the training map, agents can develop habits that work well in that familiar environment.”

“But when the layout changes, those habits are less reliable.”

“At that point, teammate information becomes more valuable.”

“Communication can help agents coordinate around obstacles, reduce uncertainty about what teammates are doing, and adapt their pursuit behavior in unfamiliar layouts.”

“A second important point is that channel size mattered.”

“The sixteen-symbol channel outperformed the four-symbol channel on transfer, which means compact communication did not fully close the gap to the larger channel.”

“So the most defensible interpretation is that communication helps generalization, and more channel capacity helps further.”

---

## Slide 12 — Interpretability result

### On the slide

Title: **Did the agents learn an interpretable language?**

Show a small summary:

- Comm-4 entropy: 1.997 / 2.0 bits
- Comm-16 entropy: 3.985 / 4.0 bits
- no strong role differentiation
- no temporal structure
- capture/escape analysis not meaningful on easy map

### Script

“After seeing that communication helped transfer, I asked a second question: what kind of communication did the agents learn?”

“The answer was not what a human might intuitively hope for.”

“The channels were used heavily, but they did not organize into a neat symbolic language.”

“Both communication conditions had near-maximum entropy, which means the agents used nearly the full vocabulary instead of collapsing onto a few symbols.”

“There was very little inter-agent specialization, meaning the three agents did not settle into clearly different message roles.”

“There was also little temporal structure across early, middle, and late parts of the episode.”

“And on the easy training map, the capture-versus-escape comparison was not meaningful because the agents almost never failed in the late training logs.”

“So the clean conclusion is that the communication was functional, but not simply interpretable at the single-symbol level.”

---

## Slide 13 — Why the null interpretability result still matters

### On the slide

Title: **Why this still matters**

- useful communication does not imply human-readable semantics
- performance and interpretability are different goals
- learned protocols may be distributed rather than symbolic

### Script

“I want to emphasize that this is not a failed result.”

“In fact, it is scientifically useful.”

“There is a common intuition that if agents communicate successfully, they must also create a neat, human-readable symbolic system.”

“But those are different goals.”

“The agents are optimizing for task success, not for human interpretability.”

“So it is completely plausible that they develop distributed signaling patterns that are useful internally but do not map neatly onto simple symbolic categories like ‘go left’ or ‘I found the prey.’”

“That distinction matters for explainability, AI safety, and human-AI interaction.”

“A system can be effective without being naturally transparent.”

---

## Slide 14 — Qualitative behavior and GIFs

### On the slide

Title: **What the agents looked like qualitatively**

Show:

- one easy-map GIF
- one transfer-map GIF, ideally `comm16_center`

Optional caption:

**Agents sometimes used containment behavior at chokepoints instead of pure chase behavior**

### Script

“I also rendered rollout visualizations to see what the policies looked like qualitatively.”

“On the training map, all conditions could capture the prey reliably.”

“On obstacle maps, the more interesting behavior was at chokepoints.”

“In some successful episodes, the agents did not simply sprint straight at the prey. Instead, they appeared to hold positions and restrict escape routes.”

“That can look like hesitation at first, but the safer interpretation is containment rather than direct chase behavior.”

“These qualitative rollouts are useful because they make the coordination story more visually intuitive, especially for an audience that may not spend time reading tables of reinforcement learning metrics.”

---

## Slide 15 — Limitations

### On the slide

Title: **Limitations**

- simplified grid-world environment
- fixed prey policy
- communication remains hard to interpret
- transfer was tested on a small family of related maps
- larger-map stress tests showed checkpoint sensitivity

### Script

“Like any study, this project has limitations.”

“First, the environment is intentionally simplified. It is a grid-world pursuit task, not a physical robotics platform.”

“Second, the prey uses a fixed evasive policy rather than learning adaptively.”

“Third, while communication improved transfer, the learned protocol was still difficult to interpret.”

“Fourth, the transfer maps were related obstacle layouts rather than a very broad distribution of environments.”

“And finally, when I explored larger-map zero-shot transfer, I found that transfer performance could be highly checkpoint-sensitive, especially for the communication model. So generalization here is real, but it is not unlimited.”

---

## Slide 16 — Conclusion

### On the slide

Title: **Conclusion**

Three main points:

- all conditions solved the easy training map
- communication improved mean transfer performance and stability
- learned messages were useful but not cleanly interpretable

Bottom line:

**Communication mainly improved adaptability, not just training-map performance**

### Script

“To conclude, this project asked whether compact learned communication helps coordination in partially observable multi-agent pursuit.”

“The first main result was that communication was not necessary for strong performance on the easiest training map.”

“The second, and more important, result was that communication improved transfer to unseen environments, with the sixteen-symbol condition performing most reliably.”

“And third, the communication channel was clearly functional, but it did not resolve into a simple human-readable language.”

“So the key takeaway is this: in this project, communication did not mainly help the agents solve the familiar environment better. It helped the team remain more robust when the world changed.”

“In that sense, communication supported adaptability more than memorization.”

“Thank you.”

---

## Optional backup slide — Stress-test nuance

### On the slide

Title: **Backup: larger-map stress test**

- larger single-barrier map `large_split`
- baseline and comm16 both transferred
- best transfer checkpoint differed from best training-map checkpoint

Possible numbers:

- Baseline `ep004000`: 93.5% capture, 88.08 steps
- Comm-16 `ep005000`: 91.0% capture, 97.49 steps

### Script

“As a final exploratory test, I also evaluated the models on a larger single-barrier map.”

“The interesting finding there was not that communication clearly won, but that transfer became checkpoint-sensitive.”

“The communication model looked weak when I used the checkpoint that was best on the training map, but it transferred much better with a later checkpoint.”

“So that backup result suggests that in-distribution optimality and out-of-distribution robustness do not necessarily peak at the same point in training.”

---

## Timing guide

- Slide 1: 0:40
- Slide 2: 1:00
- Slide 3: 1:00
- Slide 4: 1:00
- Slide 5: 0:55
- Slide 6: 0:55
- Slide 7: 1:10
- Slide 8: 1:00
- Slide 9: 1:10
- Slide 10: 1:25
- Slide 11: 1:05
- Slide 12: 1:20
- Slide 13: 1:00
- Slide 14: 1:00
- Slide 15: 0:55
- Slide 16: 1:00

Total: about 15 minutes

---

## Quick definitions for a mixed audience

Use these exact lines when needed:

> “Partial observability means each agent only has a local, incomplete view.”

> “Reinforcement learning means learning by trial and error from rewards and penalties.”

> “Parameter sharing means the agents use the same underlying model rather than each learning completely separately.”

> “Transfer means performance on new environments the agents were not trained on.”

> “Entropy here basically means how spread out or varied the message usage is.”

---

## What to avoid saying

Avoid:

- “The agents learned a language” without qualifying it
- “Communication improved everything”
- “This proves it works in the real world”
- “The agents understood the environment”

Prefer:

- “learned a communication protocol”
- “improved transfer”
- “suggests”
- “under this experimental setting”

---

## Strongest one-sentence thesis

Use this near the beginning and the end:

> “The main value of communication in this project was not better performance on the easiest known map, but better robustness when the agents had to generalize.”
