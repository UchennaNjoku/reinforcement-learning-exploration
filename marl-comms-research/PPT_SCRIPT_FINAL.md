# PPT Script — Final
**"Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit"**
U. Njoku, J. Calderon — Bethune-Cookman University
**Target: 15 minutes | Audience: mixed, not necessarily deep RL specialists**

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

"Good [morning/afternoon], everyone. My name is Uchenna Njoku, and today I'll be presenting my work on emergent communication in multi-agent reinforcement learning."

"In simple terms, I studied whether a team of artificial agents can learn to communicate with each other in order to coordinate more effectively when none of them can see the full environment."

"The short version of the talk is this: communication did not matter much on the easiest training map, but it became more useful when the agents had to generalize to new, unseen maps."

"So the real story is not just whether communication works, but where it helps and what kind of communication the agents actually learned."

---

## Slide 2 — Motivation

### On the slide
Title: **Why study communication in multi-agent systems?**
- real systems often have only local information
- team success depends on coordination under uncertainty
- communication has bandwidth and latency costs
- we want useful, compact learned communication

Icons: drones / robots / autonomous vehicles / distributed sensing systems

### Script

"The motivation for this work comes from a practical problem in distributed intelligent systems."

"In many real-world settings, there is not one agent with perfect information making all decisions. Instead, there are multiple agents, each with only a local view, trying to act together."

"Examples include teams of drones searching a disaster site, warehouse robots sharing space, autonomous vehicles coordinating locally, or distributed sensors monitoring a changing environment."

"In all of those cases, three things are usually true."

"First, each agent sees only part of the world. Second, success depends on coordination, not just individual intelligence. And third, communication is not free — it has bandwidth limits, delays, and practical costs."

"So the question becomes: can agents learn a compact communication strategy that improves teamwork without requiring a hand-designed protocol?"

---

## Slide 3 — Real-world framing

### On the slide
Title: **Why this matters beyond a grid world**
- search-and-rescue robotics
- swarm coordination in uncertain environments
- distributed autonomy under incomplete information

Highlighted line: **Communication matters most when the environment changes or becomes ambiguous**

### Script

"One reason this question matters is that the most interesting real-world environments are not static and perfectly predictable."

"Imagine a search-and-rescue scenario. One robot sees part of the environment because of debris or walls. Another sees a different region. No single robot has the full picture."

"Or imagine a fleet of warehouse robots. Each knows its own local obstacles, but efficient routing depends on what nearby teammates are doing."

"So this project is not really about a toy game. It is about a broader scientific question: how do teams stay effective when each member has incomplete information?"

"My results suggest that communication may matter less when the task is already easy and familiar, and more when the environment changes and the agents need to adapt."

---

## Slide 4 — Task setup

### On the slide
Title: **Task setup**
- 3 pursuers, 1 evasive prey
- each pursuer sees only a 7×7 local window
- goal: capture the prey efficiently

Visual: grid world with one agent's local field of view shown

### Script

"To study this question in a controlled way, I used a pursuit task in a grid-world environment."

"There are three pursuer agents and one prey."

"The prey follows an evasive policy, meaning it tries to move away from nearby pursuers."

"Each pursuer sees only a local 7 by 7 patch around itself rather than the full grid. That property is called partial observability."

"Partial observability simply means the agent does not have complete state information. It has to make decisions from an incomplete local view."

"So even if one pursuer can see the prey, it may not know where its teammates are or what they are about to do."

"The team objective is to capture the prey as efficiently and consistently as possible."

---

## Slide 5 — Research questions

### On the slide
Title: **Research questions**
- Can learned communication improve coordination under partial observability?
- Does it help mainly on the training map, or on transfer to unseen maps?
- Are the learned messages interpretable?

### Script

"From that setup, I focused on three main questions."

"First: can the agents learn a useful communication strategy under partial observability?"

"Second: if communication helps, where does it help most? Does it improve performance on the training environment, or does it mainly help when the environment changes?"

"And third: if the agents do learn to communicate, can we interpret those messages in a clean, human-readable way?"

"That last question is important because successful communication is not automatically the same thing as understandable communication. A system may rely on internal signals that are highly functional, but still not easy for humans to interpret."

---

## Slide 6 — Three conditions

### On the slide
Title: **Three conditions**

| Condition | Description |
|-----------|-------------|
| No-Comm | agents act only on local observations |
| Comm-4 | agents send one of 4 symbols each step |
| Comm-16 | agents send one of 16 symbols each step |

Footer: same training framework across all three conditions | 3 independent seeds per condition

### Script

"I compared three experimental conditions."

"The first had no explicit communication. Agents could only use their own observations."

"The second allowed each agent to send one of four possible symbols each step."

"The third expanded that communication channel to sixteen possible symbols."

"Critically, no one tells the agents what the symbols mean. There is no predefined protocol. If a useful communication strategy emerges, the agents figure it out on their own through training."

"I trained three independent random seeds per condition — nine training runs total — to make sure the results are not just one lucky seed."

---

## Slide 7 — Architecture

### On the slide
Title: **How the agents learn**

Diagram:
- Observation → neural network → movement action
- For comm: Observation + received messages → neural network → movement action + outgoing message

Bullets:
- deep Q-learning
- shared parameters across pursuers
- messages received at the next step

### Script

"I used a reinforcement learning method called deep Q-learning."

"Reinforcement learning means the agents learn by trial and error from rewards and penalties. The neural network estimates which actions are likely to lead to better long-term outcomes."

"All three pursuers share the same underlying model — that is called parameter sharing, and it means the agents learn from pooled experience instead of each starting from scratch."

"In the communication conditions, the messages sent by teammates at one time step become part of the receiving agent's input at the next step. So communication becomes one more source of information the agent can use when choosing its next move."

"The communication is end-to-end trained — the agents learn when and what to say by learning what wins the game."

---

## Slide 8 — Environment and evaluation design

### On the slide
Title: **Three maps — train on one, transfer to two**

Show map visuals: easy_open, center_block, split_barrier

- trained on: `easy_open` (open field, no obstacles)
- transfer tested on: `center_block` (central block), `split_barrier` (vertical wall, two gaps)
- greedy evaluation, 200 episodes, 3 matched seeds per condition
- checkpoint selected per run: best greedy capture rate on easy_open

### Script

"All models were trained on the same simple map called easy_open — a clear field with no obstacles."

"Then I evaluated them zero-shot on two unseen transfer maps. Center_block has a central obstacle blocking the direct path. Split_barrier has a full vertical wall with two narrow gaps the agents must navigate through."

"The key point is that the agents never saw these maps during training. Whatever transfer ability they have, they built it on easy_open."

"One important methodological detail: instead of always using the final checkpoint, I swept all checkpoints saved every 500 episodes and selected the best-performing one per run. That was necessary because DQN training can be unstable, and final-checkpoint comparisons turned out to be misleading for some runs."

---

## Slide 9 — Training-map result

### On the slide
Title: **Result 1: all conditions solve the training map**

Show: training curve figure

Main takeaway: **Communication was not necessary for strong in-distribution performance**

### Script

"Here is what training looks like across the three conditions."

"All three reached 100 percent greedy capture on easy_open, with similar average step counts of roughly 9 to 11 steps per episode."

"The comm conditions do converge faster — they reach high capture rates earlier in training — but by the end all three are equivalent."

"That means I cannot honestly claim that communication was necessary to solve the easiest environment. And that is actually useful, because it keeps the interpretation clean."

"If communication is valuable here, the interesting question is: where does it add value beyond the familiar training case?"

---

## Slide 10 — Main result: transfer

### On the slide
Title: **Result 2: communication improves transfer**

| Condition | easy_open | center_block | split_barrier |
|-----------|-----------|--------------|---------------|
| No-Comm | 100% ±0.0 | 77.7% ±28.8 | 65.8% ±26.2 |
| Comm-4 | 100% ±0.0 | 91.0% ±12.7 | 70.7% ±18.3 |
| Comm-16 | 100% ±0.0 | 95.2% ±6.1 | 79.5% ±12.7 |

Highlight: **Comm-16 transfers most reliably** | collision rate: No-Comm up to 38% · Comm-16 under 4%

### Script

"This is the main result."

"On the training map, all three conditions are at 100 percent. No difference."

"On center_block — a map with a central obstacle they have never seen — No-Comm drops to 77.7 percent. Comm-4 holds at 91 percent. Comm-16 reaches 95.2 percent."

"On split_barrier — a map with a vertical wall and narrow gaps — No-Comm falls to 65.8 percent. Comm-4 to 70.7 percent. Comm-16 to 79.5 percent."

"The pattern is consistent: communication improves transfer, and a larger vocabulary improves it further."

"There is a second signal worth noting — collision rates. On the barrier maps, no-comm agents collide with each other on up to 38 percent of steps, because without communication they have no way to coordinate who goes where. The Comm-16 agents' collision rate stays under 4 percent. They are coordinating movement, not just chasing independently."

Pause.

"That is the central empirical finding of the project."

---

## Slide 11 — Seed sensitivity

### On the slide
Title: **Variance tells part of the story too**

- No-Comm center_block: 77.7% ±28.8 → one seed got 100%, one got 37%
- Comm-16 center_block: 95.2% ±6.1

Highlighted line: **Communication doesn't just raise the mean — it raises the floor**

Note: one baseline seed failed training entirely; excluded from primary table; fourth seed confirmed normal

### Script

"I want to be transparent about the variance, because the standard deviations are large."

"On center_block, No-Comm has a standard deviation of plus or minus 28.8 percentage points. That means one seed achieved 100 percent and another achieved 37 percent. That is a genuine difference in behavior — some seeds happen to learn a strategy that transfers, and others do not."

"Comm-16's standard deviation is plus or minus 6.1 — substantially tighter. The communication models are more consistent across seeds."

"We also had one baseline seed that failed outright during training — its Q-values never converged and it relied on exploration rather than a learned policy. We excluded it from the primary table, noted it explicitly, and ran a fourth seed as a sensitivity check, which succeeded normally."

"The variance itself is part of the story: communication does not just raise the mean capture rate. It raises the floor."

---

## Slide 12 — What the transfer result means

### On the slide
Title: **What does this mean?**
- training map can be solved without explicit communication
- unseen maps expose coordination brittleness
- communication helps under distribution shift
- larger channel size helps more than smaller channel size

### Script

"One way to interpret this is that on the training map, agents can learn habits that work well in that one familiar geometry."

"But when the layout changes, those habits become less reliable."

"At that point, teammate information becomes more valuable."

"Communication can help the agents coordinate around obstacles, reduce uncertainty about what teammates are doing, and adapt their pursuit behavior when the environment changes."

"A second important point is that channel size mattered. The sixteen-symbol condition outperformed the four-symbol condition on transfer, which means compact communication did not fully close the gap to the larger channel."

"So the cleanest interpretation is that communication helps generalization, and more channel capacity helps further."

---

## Slide 13 — Transfer GIF demo

### On the slide
Title: **What the transfer policy looks like**

Show: `rollout_comm16_center.gif`

Caption: Comm-16, center_block map (never seen in training) — captured in 61 steps

### Script

"Let me show you what that 95 percent capture rate actually looks like."

"This is Comm-16 on center_block — a map it has never trained on. The three agents are red, blue, and green. The colored badges at the bottom show which symbol each agent is currently broadcasting."

"Watch how they spread out and approach the evader from different angles. They navigate around the obstacle. They do not cluster together on the same path."

"In some episodes you will also see what looks like a holding pattern near chokepoints — that is not hesitation, it is containment behavior, where an agent cuts off an escape route while teammates close in."

"A no-comm agent on the same map is far more likely to get stuck on the same side of the obstacle as its teammates, or to converge on the same cell and block each other. That coordination failure is exactly where the 22-point capture rate gap comes from."

---

## Slide 14 — Interpretability

### On the slide
Title: **Did the agents learn an interpretable language?**

Summary:
- Comm-4 entropy: 1.997 / 2.0 bits (near-maximum)
- Comm-16 entropy: 3.985 / 4.0 bits (near-maximum)
- inter-agent role variation: < 1.1%
- temporal structure: < 2% phase shift across episode
- capture vs. escape: not computable — < 5 failed episodes in 500

### Script

"After seeing that communication helped transfer, I asked what kind of communication the agents actually learned."

"I analyzed the message logs from 500-episode evaluation runs."

"Symbol entropy was near-maximum for both conditions. The agents used the full vocabulary rather than collapsing onto one or two signals. That tells us the channel is active."

"There was very little inter-agent specialization — less than 1.1 percent variation across the three agents. They did not settle into clearly different roles like scout and chaser."

"There was also little temporal structure. Messages did not shift meaningfully across early, middle, and late parts of the episode."

"And on the easy training map, the capture-versus-escape comparison was not meaningful — the agents succeed on nearly every episode, so there are fewer than 5 failures in 500 to analyze."

"The clean summary: the communication was functional, but it did not resolve into a simple human-readable symbolic language."

---

## Slide 15 — Why the null interpretability result matters

### On the slide
Title: **Why this still matters**
- useful communication ≠ human-readable semantics
- performance and interpretability are different goals
- learned protocols may be distributed rather than symbolic

### Script

"I want to emphasize that this is not a failed result."

"There is a common intuition that if agents communicate successfully, they must also create a neat symbolic system — something like 'symbol 3 means I see the prey.' But those are different goals."

"The agents are optimizing for task success, not for human interpretability. So it is completely plausible that they develop distributed signaling patterns that are useful internally but do not map neatly onto simple categories."

"That distinction matters for explainability and AI safety. A system can be effective without being naturally transparent."

---

## Slide 16 — Limitations and future work

### On the slide
Title: **Limitations and next steps**

Limitations:
- simplified grid-world, fixed prey policy
- trained on one map only
- DQN instability across seeds and checkpoints
- communication functional but opaque

Future work:
- train on harder maps or use curriculum learning
- scale to more agents (requires architecture change)
- t-SNE of message space conditioned on agent positions

### Script

"Like any study, this one has honest limitations."

"The environment is a simplified grid-world. The prey uses a fixed evasive policy rather than learning adaptively."

"We trained only on easy_open, so we do not know whether the transfer advantage persists if models are trained on harder maps from the start."

"DQN has instability — we saw significant variance across seeds and checkpoints, and our best-checkpoint selection methodology controls for it but does not eliminate it."

"And while communication improved transfer, the learned protocol is still opaque."

"The three natural extensions are: training on harder maps or curriculum sequences, scaling to more agents once the architecture is updated, and using t-SNE or spatial conditioning to probe what the message embeddings actually encode."

---

## Slide 17 — Conclusion

### On the slide
Title: **Conclusion**

Three main points:
- all conditions solved the easy training map equally
- communication improved mean transfer performance and stability
- learned messages were functional but not cleanly interpretable

Bottom line: **Communication mainly improved adaptability, not training-map performance**

### Script

"To conclude, this project asked whether compact learned communication helps coordination in partially observable multi-agent pursuit."

"The first main result was that communication was not necessary for strong performance on the easiest training map."

"The second, and more important, result was that communication improved transfer to unseen environments, with the sixteen-symbol condition performing most reliably and most consistently across seeds."

"And third, the communication channel was clearly functional — near-maximum entropy, full vocabulary usage — but it did not resolve into a simple human-readable language."

"So the key takeaway is this: in this project, communication did not mainly help the agents solve the familiar environment better. It helped the team remain more robust when the world changed."

"In that sense, communication supported adaptability more than memorization."

"Thank you."

---

## Backup slide — Stress-test nuance

### On the slide
Title: **Larger-map zero-shot test**
- evaluated on `large_split`: 20×20 single-barrier map (same structure as split_barrier, larger grid)
- baseline ep004000: 93.5% capture, 88 steps
- Comm-16 ep005000: 91.0% capture, 97 steps
- but Comm-16 selected checkpoint (ep003500): only 37.5%

Finding: in-distribution optimality and out-of-distribution robustness do not necessarily peak at the same checkpoint

### Script

"As an exploratory stress test, I also evaluated both models on a larger 20 by 20 map with the same single-barrier structure as split_barrier."

"The interesting result was not that communication clearly won. It was that transfer performance became highly checkpoint-sensitive for the communication model."

"Using the checkpoint that was best on the training map, Comm-16 only achieved 37.5 percent on the larger map. But a later checkpoint achieved 91 percent — close to the baseline's 93.5."

"That suggests in-distribution optimality and out-of-distribution robustness do not always peak at the same point in training. I flagged it as a known limitation rather than including it in the main results."

---

## Q&A Prep

**"Why not train on all three maps?"**
We wanted to isolate zero-shot transfer as the test. Training on all maps would tell you about multi-task learning, which is a different question. It is a natural next step.

**"How do you know the symbols aren't just noise?"**
The capture rate improvement is real and consistent across 3 seeds. Whatever the messages encode, they are contributing to better generalization. The near-maximum entropy and full vocabulary usage suggest the channel is carrying information rather than being ignored.

**"Why DQN and not PPO or MADDPG?"**
DQN with parameter sharing is a natural baseline for discrete action spaces and standard in the MARL literature. Its instability is a known limitation we acknowledge. PPO would be a stronger foundation for future work.

**"What happened with the failed seed?"**
One of the three baseline seeds never converged — Q-values stayed noisy and the agent relied on epsilon-exploration at training time, but failed completely at greedy evaluation. This is a known DQN failure mode and is exactly why we ran multiple seeds. We excluded it from the primary table, noted it explicitly, and ran a fourth seed as a sensitivity check, which succeeded normally.

**"Could the comm agents be cheating?"**
No. Messages from step t are received at step t+1, so there is no information leakage within a timestep. The protocol is consistent with the emergent communication literature.

**"What does the large map result mean?"**
It means the communication advantage has limits. The learned protocol was calibrated to a 16×16 scale. On a 20×20 map, it degrades more than the baseline unless you happen to pick the right checkpoint. It is an honest finding and I flag it as a limitation.

---

## Quick definitions — use these exact lines when needed

> "Partial observability means each agent only has a local, incomplete view of the environment."

> "Reinforcement learning means learning by trial and error from rewards and penalties."

> "Parameter sharing means the agents use the same underlying model rather than each learning separately from scratch."

> "Transfer means performance on new environments the agents were not trained on."

> "Entropy here basically means how spread out or varied the message usage is — near-maximum entropy means the agents are using the full vocabulary."

---

## What to avoid saying

Avoid:
- "The agents learned a language" — say "learned a communication protocol"
- "Communication improved everything" — say "improved mean transfer performance"
- "This proves it works in the real world" — say "under this experimental setting"
- "The agents understood the environment" — say "developed a coordination capacity"

---

## Timing guide

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:40 |
| 2 | Motivation | 1:00 |
| 3 | Real-world framing | 1:00 |
| 4 | Task setup | 1:00 |
| 5 | Research questions | 0:55 |
| 6 | Three conditions | 0:55 |
| 7 | Architecture | 1:00 |
| 8 | Maps and eval design | 1:00 |
| 9 | Training-map result | 1:00 |
| 10 | Main transfer result + collision | 1:30 |
| 11 | Seed sensitivity | 1:05 |
| 12 | What the transfer result means | 0:55 |
| 13 | Transfer GIF demo | 1:10 |
| 14 | Interpretability | 1:10 |
| 15 | Why null result matters | 0:55 |
| 16 | Limitations and future work | 1:00 |
| 17 | Conclusion | 1:00 |
| **Total** | | **~16:15** |

> Note: running ~1 min over. Trim by cutting Slide 3 (real-world framing) if you need to hit exactly 15:00 — the motivation slide covers the same ground.

Buffer the extra 20 seconds against Slide 10 or 12 depending on how long the GIF runs.
