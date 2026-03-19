# PPT Script Draft 4

## Title

**Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit**

Target length: 15 minutes  
Audience: mixed / bio-heavy / non-specialist-friendly  
Main thesis:

> In this project, communication mattered less for solving the easiest familiar map, and more for helping the team stay effective when the environment changed.

---

## Slide 1 — Title

### On the slide

- Title
- Your name
- Bethune-Cookman University
- one clean pursuit screenshot or grid visual

### Script

“Good [morning/afternoon], everyone. My name is Uchenna Njoku, and today I’ll be presenting my work on emergent communication in multi-agent reinforcement learning.”

“In simple terms, I studied whether a team of artificial agents can learn to communicate with each other in order to coordinate more effectively when none of them can see the full environment.”

“The main result I’ll argue today is this: communication did not matter much on the easiest training environment, but it became more useful when the agents had to generalize to new, unseen environments.”

Pause.

“So the real story is not just whether communication works, but where it helps, and what kind of communication the agents actually learned.”

---

## Slide 2 — Motivation

### On the slide

Title: **Why study communication in multi-agent systems?**

- local information is often incomplete
- team coordination matters under uncertainty
- communication has practical cost
- we want efficient learned protocols

Possible visuals:

- drones
- warehouse robots
- autonomous vehicles
- distributed sensing systems

### Script

“Let’s start with why this matters.”

“In many real-world systems, you do not have one agent with perfect knowledge making all decisions. Instead, you have multiple agents, each with only a local view, trying to act together.”

“Examples include search-and-rescue drones, groups of robots in a warehouse, autonomous vehicles sharing local information, or distributed sensor systems monitoring an environment that is changing over time.”

“In all of these cases, three things are usually true.”

“First, each agent only sees part of the world.”

“Second, success depends on coordination, not just individual intelligence.”

“And third, communication is not free. It has bandwidth limits, delay, and real system costs.”

“So the scientific question becomes: if communication is limited, can agents learn a compact signaling protocol that still improves teamwork?”

---

## Slide 3 — Why this matters beyond the toy setting

### On the slide

Title: **Why this matters beyond a grid world**

- distributed autonomy
- collective sensing
- uncertain and changing environments

Highlighted line:

**Communication becomes more valuable when the environment is unfamiliar or ambiguous**

### Script

“One reason this is a useful research problem is that the interesting real-world environments are rarely fixed and perfectly predictable.”

“Imagine a post-disaster setting. One robot sees one part of the environment because of walls or debris. Another robot sees a different region. No single robot has the full picture.”

“Or imagine a warehouse fleet. Each robot may know its local obstacles, but efficient behavior depends on what teammates are doing elsewhere.”

“So this is not really about a game. It is about coordination under incomplete information.”

“My results suggest that communication is most valuable not when the environment is already easy and familiar, but when the team has to adapt to change.”

---

## Slide 4 — Task setup

### On the slide

Title: **Task setup**

- 3 pursuers
- 1 evasive prey
- each pursuer sees only a 7x7 local window
- goal: capture the prey efficiently

Visual:

- grid world and local field-of-view illustration

### Script

“To study this question in a controlled way, I used a pursuit task in a grid-world environment.”

“There are three pursuer agents and one prey.”

“The prey follows an evasive policy, meaning it tries to move away from nearby pursuers.”

“Each pursuer sees only a local 7-by-7 patch around itself rather than the full map. That is what we mean by partial observability.”

“Partial observability simply means the agent has incomplete state information. It must act from a restricted local view.”

“So one pursuer might know where the prey is relative to itself, but not where its teammates are, or how they are approaching.”

“The team objective is to capture the prey as efficiently and consistently as possible.”

---

## Slide 5 — Research questions

### On the slide

Title: **Research questions**

- Can learned communication improve coordination under partial observability?
- Does it help mainly on the training map, or on transfer to unseen maps?
- Are the learned messages interpretable?

### Script

“From that setup, I focused on three research questions.”

“First: can the agents learn a useful communication strategy under partial observability?”

“Second: if communication helps, where does it help most? On the training map, or when the environment changes?”

“And third: if the agents do learn to communicate, can we interpret those messages in a clean, human-readable way?”

“That last question matters because successful communication is not automatically the same thing as understandable communication.”

---

## Slide 6 — Experimental conditions

### On the slide

Title: **Three experimental conditions**

| Condition | Description |
|-----------|-------------|
| No-Comm | agents act only on local observations |
| Comm-4 | agents send one of 4 symbols |
| Comm-16 | agents send one of 16 symbols |

Footer note:

- same overall training framework across conditions

### Script

“To answer these questions, I compared three conditions.”

“The first had no explicit communication at all.”

“The second allowed each agent to send one of four possible symbols each step.”

“The third expanded that vocabulary to sixteen symbols.”

“No one tells the agents what those symbols mean. There is no hand-designed protocol.”

“If a useful communication strategy emerges, the agents have to discover it for themselves through learning.”

---

## Slide 7 — How the agents learn

### On the slide

Title: **How the agents learn**

Diagram:

- Observation -> neural network -> movement action
- For comm conditions: Observation + received messages -> neural network -> movement action + outgoing message

Bullets:

- deep Q-learning
- shared parameters across pursuers
- messages arrive at the next step

### Script

“I used a reinforcement learning method called deep Q-learning.”

“Reinforcement learning means learning by trial and error from rewards and penalties.”

“The neural network estimates which actions are likely to lead to better long-term outcomes.”

“All three pursuers share the same underlying model. That is called parameter sharing.”

“Parameter sharing means the agents use one shared learned policy rather than each learning completely separately.”

“In the communication conditions, each agent receives its teammates’ messages from the previous step as part of its next input.”

“So communication becomes one more information source the agent can use when deciding how to move.”

---

## Slide 8 — Training and evaluation design

### On the slide

Title: **Training and evaluation design**

- trained only on `easy_open`
- tested zero-shot on `center_block` and `split_barrier`
- greedy evaluation
- 3 matched seeds per condition
- best-checkpoint selection instead of final-checkpoint-only

### Script

“All models were trained on one simple map called `easy_open`.”

“Then I evaluated them on unseen transfer maps called `center_block` and `split_barrier`, with no additional training.”

“That is the transfer test.”

“Evaluation was greedy, meaning the exploration noise used during training was turned off.”

“I also used three matched seeds per condition.”

“One very important detail is that I did not just use the final checkpoint. Instead, I swept checkpoints saved every 500 episodes and selected the best greedy checkpoint on the training map for each run.”

“That turned out to matter, because some runs partially degraded late in training.”

---

## Slide 9 — Result 1: training-map performance

### On the slide

Title: **Result 1: all conditions solve the training map**

Show:

- `comm_training_curves.png` or a small easy-open summary table

Highlighted takeaway:

**Communication was not necessary for strong in-distribution performance**

### Script

“The first major result is that all three conditions solved the training map.”

“With best-checkpoint selection, all three reached 100 percent greedy capture on `easy_open`, with similar step counts, roughly 9 to 11 steps.”

“That means I cannot honestly claim that communication was necessary for success in the easiest familiar environment.”

“And that is actually useful, because it keeps the interpretation honest.”

“Communication is not being given credit where it is not deserved.”

“So if communication matters, the more interesting question becomes: where does it add value beyond the easiest case?”

---

## Slide 10 — Result 2: transfer performance

### On the slide

Title: **Result 2: communication helps on unseen maps**

Use a simplified version of the matched-seed table:

| Condition | easy_open | center_block | split_barrier |
|-----------|-----------|--------------|---------------|
| No-Comm | 83.8% ± 22.9 | 68.5% ± 44.5 | 67.0% ± 45.3 |
| Comm-4 | 100% ± 0 | 91.0% ± 12.7 | 70.7% ± 18.3 |
| Comm-16 | 100% ± 0 | 95.2% ± 6.1 | 79.5% ± 12.7 |

Big highlight:

**Comm-16 transfers most reliably**

### Script

“The clearest benefit of communication appeared not on the training map, but on the unseen transfer maps.”

“On `center_block`, the no-communication baseline was much less stable across seeds, while the communication conditions held up better.”

“On `split_barrier`, which is the harder transfer map, the sixteen-symbol channel again gave the strongest mean performance.”

“So the main value of communication in this project was not higher asymptotic performance on the easiest known map.”

“It was better generalization and better robustness when the layout changed.”

Pause.

“That is the central empirical finding of the project.”

---

## Slide 11 — What the transfer result means

### On the slide

Title: **What does this mean?**

- training map can be solved without explicit communication
- unseen maps expose coordination brittleness
- communication helps under distribution shift
- larger channel size helps more than smaller channel size

### Script

“One way to interpret this is that on the training map, agents can learn habits that work well in that one familiar geometry.”

“But when the layout changes, those habits become less reliable.”

“At that point, teammate information becomes more valuable.”

“Communication can help the agents coordinate around obstacles, reduce uncertainty about what teammates are doing, and adapt their pursuit behavior when the environment changes.”

“A second important point is that channel size mattered.”

“The sixteen-symbol condition outperformed the four-symbol condition on transfer, which means compact communication did not fully close the gap to the larger channel.”

“So the cleanest interpretation is that communication helps generalization, and more channel capacity helps further.”

---

## Slide 12 — Interpretability result

### On the slide

Title: **Did the agents learn an interpretable language?**

Small summary:

- Comm-4 entropy: 1.997 / 2.0 bits
- Comm-16 entropy: 3.985 / 4.0 bits
- no strong role differentiation
- no strong temporal structure
- success/failure comparison not meaningful on the easy map

### Script

“After seeing that communication helped transfer, I asked the next natural question: what are the agents actually saying?”

“Did they create something like a simple symbolic language?”

“The answer was more subtle.”

“The communication channels were definitely used, but they did not organize into a clean human-readable code.”

“Both conditions had near-maximum entropy, which means the agents used nearly the full vocabulary rather than collapsing onto just one or two symbols.”

“There was very little inter-agent specialization, meaning the three agents did not settle into clearly different message roles.”

“There was also little temporal structure across the episode.”

“And on the easiest map, the success-versus-failure comparison was not meaningful because the agents almost never failed in the late training logs.”

“So the clean conclusion is that the communication was functional, but not simply interpretable at the single-symbol level.”

---

## Slide 13 — Why that null result still matters

### On the slide

Title: **Why this “negative” result still matters**

- useful communication does not imply human-readable semantics
- performance and interpretability are different goals
- learned protocols may be distributed rather than symbolic

### Script

“I want to emphasize that this is not a failed result.”

“In fact, it is scientifically valuable.”

“There is a common intuition that if agents communicate successfully, they must also produce a neat symbolic language.”

“But those are different goals.”

“The agents are optimizing for coordination and reward, not for human readability.”

“So it is completely plausible that the system develops distributed signaling patterns that are useful internally but do not map neatly onto simple symbolic categories.”

“That distinction matters for explainability, AI safety, and human-AI interaction.”

“A system can be effective without being naturally transparent.”

---

## Slide 14 — Qualitative behavior and GIFs

### On the slide

Title: **What the behavior looked like**

Show:

- one easy-map GIF
- one transfer-map GIF, ideally `comm16_center`

Caption:

**Agents sometimes use containment behavior at chokepoints rather than pure chase behavior**

### Script

“I also rendered rollout visualizations to understand the policies qualitatively.”

“On the training map, all conditions captured the prey reliably.”

“On obstacle maps, the more interesting behavior appeared at chokepoints.”

“In some successful episodes, the agents did not simply sprint straight at the prey. Instead, they appeared to hold positions and restrict escape routes.”

“At first that can look like hesitation or waiting, but the safer interpretation is containment rather than direct chase behavior.”

“These rollout visuals are useful because they make the coordination story more intuitive than a table of metrics alone.”

---

## Slide 15 — Limitations

### On the slide

Title: **Limitations**

- simplified grid-world environment
- fixed prey policy
- communication remains hard to interpret
- transfer was tested on a limited family of related maps
- larger-map stress tests showed checkpoint sensitivity

### Script

“Like any study, this project has limitations.”

“First, the environment is intentionally simplified. It is a grid-world pursuit task, not a physical robotics system.”

“Second, the prey follows a fixed evasive policy rather than learning adaptively.”

“Third, while communication improved transfer, the learned protocol remained difficult to interpret.”

“Fourth, the transfer maps were still a small family of related layouts rather than a broad environment distribution.”

“And finally, in larger-map exploratory tests, transfer performance could be checkpoint-sensitive, especially for the communication model. So generalization here is real, but not unlimited.”

---

## Slide 16 — Conclusion

### On the slide

Title: **Conclusion**

Three takeaways:

- all conditions solved the easy training map
- communication improved mean transfer performance and robustness
- learned messages were useful but not cleanly interpretable

Bottom line:

**Communication mainly improved adaptability, not just training-map performance**

### Script

“To conclude, this project asked whether compact learned communication helps coordination in partially observable multi-agent pursuit.”

“The first main result was that communication was not necessary for strong performance on the easiest training map.”

“The second, and more important, result was that communication improved transfer to unseen environments, with the sixteen-symbol condition performing most reliably.”

“And third, the communication channel was clearly functional, but it did not reduce to a simple human-readable language.”

“So the key takeaway is this: in this project, communication did not mainly help the agents solve the familiar environment better. It helped the team remain more robust when the world changed.”

“In that sense, communication supported adaptability more than memorization.”

“Thank you.”

---

## Optional backup slide — Larger-map stress test

### On the slide

Title: **Backup: larger-map stress test**

- larger single-barrier map `large_split`
- both baseline and Comm-16 transferred
- best transfer checkpoint differed from best training-map checkpoint

Possible numbers:

- Baseline `ep004000`: 93.5% capture, 88.08 steps
- Comm-16 `ep005000`: 91.0% capture, 97.49 steps

### Script

“As an exploratory extension, I also evaluated the models on a larger single-barrier map.”

“The most interesting finding there was not that communication clearly won, but that transfer became checkpoint-sensitive.”

“For the communication model, the checkpoint that was best on the training map was not the checkpoint that transferred best to the larger map.”

“So this backup result suggests that in-distribution optimality and out-of-distribution robustness do not necessarily peak at the same point in training.”

---

## Timing guide

- Slide 1: 0:40
- Slide 2: 1:05
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

Use these exact lines if needed:

> “Partial observability means each agent only has a local, incomplete view.”

> “Reinforcement learning means learning by trial and error from rewards and penalties.”

> “Parameter sharing means the agents use the same underlying model rather than each learning completely separately.”

> “Transfer means performance on new environments the agents were not trained on.”

> “Entropy here basically means how spread out or varied the message use is.”

---

## Strongest one-sentence thesis

Use this near the start and near the end:

> “The main value of communication in this project was not better performance on the easiest known map, but better robustness when the agents had to generalize.”
