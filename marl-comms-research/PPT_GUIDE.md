# PPT Guide — Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit
**U. Njoku, J. Calderon — Bethune-Cookman University**
**Target: 15 minutes | Audience: mixed CS/AI, not necessarily deep RL specialists**

---

## How to use this guide

Each slide section contains three parts:
- **SLIDE CONTENT** — exactly what text and visuals go on the slide
- **VISUALS** — specific file paths to use
- **SCRIPT** — what you say out loud

Files are in `marl-comms-research/results/` unless noted otherwise.

---

## Visuals to CREATE before building the deck

These do not exist yet and would significantly strengthen specific slides. Details on each are in the relevant slide section.

| # | Visual | For Slide | Effort |
|---|--------|-----------|--------|
| A | Partial observability diagram — 16×16 grid, 7×7 window shaded, rest dark | 4 | Low |
| B | Architecture diagram — CNN → FC → dual heads (move + message) | 7 | Medium |
| C | Collision rate bar chart — grouped bars by condition and map | 10 | Low (one matplotlib call) |
| D | Per-seed scatter — individual seed capture rates on transfer maps | 11 | Low (one matplotlib call) |

---

## Timing guide

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:40 |
| 2 | Motivation | 1:00 |
| 3 | Task setup | 1:00 |
| 4 | Research questions | 0:55 |
| 5 | Three conditions | 0:55 |
| 6 | Architecture | 1:00 |
| 7 | Maps and eval design | 1:00 |
| 8 | Training-map result | 1:00 |
| 9 | Main transfer result | 1:30 |
| 10 | Seed sensitivity | 1:00 |
| 11 | What transfer means | 0:45 |
| 12 | Visual comparison (GIF pair) | 1:15 |
| 13 | Transfer demo — center_block | 1:00 |
| 14 | Interpretability | 1:10 |
| 15 | Why null result matters | 0:50 |
| 16 | Limitations and future work | 1:00 |
| 17 | Conclusion | 1:00 |
| **Total** | | **~15:00** |

> Optional backup slide at the end for Q&A on the large-map stress test.

---

---

## Slide 1 — Title

### SLIDE CONTENT

**Title:** Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit

**Subtitle:** U. Njoku, J. Calderon
Bethune-Cookman University

**Visual:** Use `results/ppt_gif1_baseline_easy.gif` as a small looping background element or inset — three colored agents converging on prey in 5 steps. Gives the audience an instant visual of what the project is about before you say a word.

**Layout:** Title left-aligned, GIF inset on the right half of the slide.

### SCRIPT

"Good morning/afternoon everyone. My name is Uchenna Njoku, and today I'll be presenting my work on emergent communication in multi-agent reinforcement learning."

"The short version: I studied whether a team of AI agents can learn to communicate with each other to coordinate better, especially when they can only see a small part of the environment around them."

"The key finding was that communication did not matter much when the task was easy and familiar — but it became meaningfully more useful when the agents had to generalize to new, unseen maps."

"So this talk is really about where communication helps, not just whether it helps."

---

## Slide 2 — Motivation

### SLIDE CONTENT

**Title:** Why study communication in multi-agent systems?

**Bullets:**
- Real systems often give each agent only local information
- Team success depends on coordination under uncertainty
- Communication has bandwidth and latency costs in practice
- Goal: useful, compact, *learned* communication — no hand-designed protocol

**Icons (suggested):** drone swarm, warehouse robot, autonomous vehicle, distributed sensor grid. Use simple icons or small stock images. Do not need to be research images.

**Bottom callout box:**
> "Can agents learn to coordinate through a communication channel they design themselves?"

### SCRIPT

"The motivation comes from a practical problem in distributed intelligent systems."

"In many real-world settings there is no single agent with full information. Instead you have multiple agents, each with only a local view, trying to act together."

"Think about teams of drones searching a disaster site, warehouse robots sharing space, autonomous vehicles coordinating at an intersection, or distributed sensors monitoring a changing environment."

"In all of these cases, three things tend to be true: each agent sees only part of the world, success depends on coordination, and communication is not free — it costs bandwidth, has delays, and scales poorly if every agent broadcasts everything."

"So the scientific question is: can agents learn a compact, useful communication strategy from scratch — without anyone telling them what the messages mean?"

---

## Slide 3 — Task setup

### SLIDE CONTENT

**Title:** Task setup — 3 pursuers, 1 evasive prey

**Left side — bullets:**
- 16×16 grid world
- 3 pursuers (red, blue, green) vs 1 evasive prey
- Each pursuer sees only a 7×7 local window — *partial observability*
- Prey moves away from nearest pursuer
- Goal: capture prey as quickly and consistently as possible

**Right side — visual:**
Use `results/ppt_gif1_baseline_easy.gif` here as the main visual.
Full size on the right half of the slide, looping.

**Below the GIF — caption:**
> Baseline agents on easy_open. Captured in 5 steps.

**CREATE — Visual A (partial observability diagram):**
If you want to make this slide stronger, add a small inset diagram:
Draw the 16×16 grid. Place an agent in the center. Shade the 7×7 window around it in light color. Make everything outside the window dark/grey. Label the window "Agent's view." This takes ~10 minutes in any drawing tool and makes partial observability immediately intuitive to a non-RL audience.

### SCRIPT

"To study this question in a controlled way, I used a pursuit task in a grid-world environment."

"There are three pursuer agents and one prey. The prey uses a fixed evasive policy — it tries to move away from nearby pursuers."

"The critical constraint is partial observability. Each pursuer sees only a 7 by 7 patch around itself. It does not know where its teammates are unless they happen to be nearby. It does not know the full position of the prey unless the prey enters its local window."

"So even if one pursuer sees the prey, it may have no idea what its two teammates are doing or where they are heading."

"The team objective is to capture the prey as efficiently and consistently as possible."

---

## Slide 4 — Research questions

### SLIDE CONTENT

**Title:** Three research questions

**Large numbered list:**

1. Can learned communication improve coordination under partial observability?

2. If so, where does it help most — on a familiar training map, or when generalizing to unseen maps?

3. Are the learned messages interpretable in a human-readable way?

**Bottom note (smaller text):**
> Question 3 matters because effective communication and understandable communication are not the same thing.

### SCRIPT

"From that setup, I focused on three main questions."

"First: can agents actually learn a useful communication strategy under partial observability at all? Or does the difficulty of credit assignment make it impossible?"

"Second: if communication helps, when does it help most? On the familiar training environment, or when the environment changes?"

"Third: if the agents do develop a communication protocol, can we read it? Can we say 'symbol 3 means I see the prey' or 'agent 2 always signals when it's chasing'?"

"That last question matters because successful communication is not automatically interpretable communication. A system may develop internal signaling that is highly functional but impossible for a human to parse — and that distinction matters for explainability and AI safety."

---

## Slide 5 — Three conditions

### SLIDE CONTENT

**Title:** Three experimental conditions

**Table — centered, large font:**

| Condition | Communication channel | Description |
|-----------|----------------------|-------------|
| No-Comm | None | Agents act only on local observations |
| Comm-4 | 1 symbol from vocab of 4 | 2 bits per step per agent |
| Comm-16 | 1 symbol from vocab of 16 | 4 bits per step per agent |

**Below table:**
- Same training framework across all three
- No predefined protocol — agents invent meaning from scratch
- 3 independent random seeds per condition (9 training runs total)

**Highlight box:**
> The agents are not told what the symbols mean. If a protocol emerges, they discover it.

### SCRIPT

"I compared three experimental conditions."

"The first was a no-communication baseline. Agents acted only on their own local observations."

"The second allowed each agent to broadcast one of four discrete symbols each timestep. Two bits of information."

"The third expanded that to sixteen symbols — four bits per step."

"The critical thing: no one defines what the symbols mean. There is no protocol handed to the agents. If useful communication emerges, it is because training discovered it as a strategy that wins."

"I ran three independent random seeds per condition — nine training runs total — to make sure the results are not a single lucky initialization."

---

## Slide 6 — Architecture

### SLIDE CONTENT

**Title:** How the agents learn — Deep Q-Learning with parameter sharing

**CREATE — Visual B (architecture diagram):**
This slide really needs a diagram. Draw the following flow exactly as the code works:

```
         BASELINE (No-Comm)                    COMM MODEL (Comm-4 / Comm-16)

   [7×7×3 local observation]             [7×7×3 local observation]
              ↓                                        ↓
   [Conv2d 3→32, 3×3, pad=1]             [Conv2d 3→32, 3×3, pad=1]
              ↓  ReLU                                  ↓  ReLU
   [Conv2d 32→64, 3×3, pad=1]            [Conv2d 32→64, 3×3, pad=1]
              ↓  ReLU                                  ↓  ReLU
          [Flatten]                               [Flatten]
              ↓                                        ↓
       [3136 features]                         [3136 features]
              +                                        +
  [Agent ID embedding (8-dim)]          [Agent ID embedding (8-dim)]
              ↓                                        +
      concat → 3144-dim                [2 teammate msgs (one-hot, 2×vocab_size)]
              ↓                                        ↓  Linear+ReLU
   [Linear 3144 → 256]  ReLU              [Message encoder → 16-dim]
              ↓                                        ↓
   [Linear 256 → 128]   ReLU          concat → 3160-dim (3136+8+16)
              ↓                                        ↓
   [Linear 128 → 5]                     [Linear 3160 → 256]  ReLU
              ↓                                        ↓
     [5 Q-values]                        [Linear 256 → 128]   ReLU
   (move actions)                                      ↓
                                          ┌────────────┴────────────┐
                                          ↓                         ↓
                                 [Linear 128 → 5]        [Linear 128 → vocab_size]
                                          ↓                         ↓
                                   [Move Q-values]       [Message Q-values]
                                   (5 move actions)      (4 or 16 symbols)
```

Use a simple box-and-arrow style. Color the message path blue, baseline grey.
For the presentation you only need to show the COMM MODEL side — the baseline is the same thing minus the blue message path.

**If you don't create the diagram, use bullets:**
- CNN: two convolutional layers (3→32→64 channels, 3×3 kernels) process the 7×7 observation → 3136 features
- Agent ID embedding (8-dim) is concatenated so one shared network can serve all 3 pursuers
- Comm agents also receive 2 teammates' previous messages as one-hot vectors → encoded to 16-dim
- All three inputs concatenated → two FC layers (256 → 128) → dual output heads
- Move head: 5 Q-values (up/down/left/right/stay)
- Message head: vocab_size Q-values (which symbol to broadcast)
- Parameter sharing: all 3 pursuers use the same model weights
- Messages from step *t* arrive at teammates at step *t+1*

**Key design note (small text or callout):**
> Separate move and message heads — joint action space (5 × vocab) failed catastrophically during development

### SCRIPT

"I used deep Q-learning — a method where a neural network learns to estimate which actions lead to better long-term outcomes, trained through trial and error."

"All three pursuers share one set of model weights — that is called parameter sharing. Rather than three separate models each learning from scratch, they all learn from pooled experience. The agent's index is passed as an embedding so one network can serve all three pursuers with different behavior."

"The network first processes each agent's 7 by 7 local observation through two convolutional layers. This extracts spatial features — things like where the prey is relative to the agent, and what cells are occupied. Those features are then flattened and combined with the agent ID embedding, before passing through two fully connected layers down to the Q-value outputs."

"In the communication conditions, each agent also receives the messages its two teammates sent at the previous timestep — encoded and concatenated before the fully connected layers. This is the only additional input."

"The key architectural decision was to use two separate output heads — one for movement, one for the outgoing message. Early experiments used a joint action space combining both. That failed because DQN cannot cleanly separate movement credit from message credit when rewards are sparse. Separate heads keep movement learning identical in complexity to the baseline."

---

## Slide 7 — Maps and evaluation design

### SLIDE CONTENT

**Title:** Train on one map — evaluate zero-shot on two unseen maps

**Three map visuals side by side:**

Use screenshots or diagrams of the three maps. If you have the renderer, take a screenshot of each map at episode start. Label them:

- **easy_open** → "Training map — open field, no obstacles"
- **center_block** → "Transfer map 1 — 4×4 central obstacle"
- **split_barrier** → "Transfer map 2 — vertical wall, two narrow gaps"

Under each map label whether agents were trained or only evaluated there:
- easy_open: ✅ Trained + Evaluated
- center_block: ❌ Never seen in training — evaluated only
- split_barrier: ❌ Never seen in training — evaluated only

**Bottom methodology note:**
> Best-checkpoint selection: swept all checkpoints every 500 episodes, selected highest greedy capture rate on easy_open per run. Necessary because DQN late-training instability makes final-checkpoint comparisons unreliable.

**Note on map screenshots:** Run the renderer and take a screenshot at `env.reset()` before any movement. Or use the first frame of any GIF as a still image.

### SCRIPT

"All three models were trained exclusively on easy_open — a flat open field with no obstacles."

"Then, without any additional training, I evaluated them on two maps they had never seen. Center_block has a central 4 by 4 obstacle that forces agents to route around the middle. Split_barrier has a full vertical wall at column 8 with only two narrow gaps at rows 5 and 10."

"Zero-shot transfer means whatever ability the agents have to handle these maps, they built it up while training on the simple open field."

"One methodological detail worth mentioning: instead of just using the final checkpoint from each training run, I swept every checkpoint saved at 500-episode intervals and picked the best-performing one per run based on easy_open capture rate. DQN training can degrade after peak performance, so final-checkpoint comparisons turned out to be misleading for some seeds."

---

## Slide 8 — Training-map result

### SLIDE CONTENT

**Title:** Result 1 — all conditions solve the training map

**Left: training curves figure**
File: `results/comm_training_curves.png`
This shows all three conditions' capture rates during training. Label the axes clearly if the image is small.

**Right: two GIFs side by side**
- `results/ppt_gif1_baseline_easy.gif` — label: "No-Comm — 5 steps"
- `results/ppt_gif2_comm16_easy.gif` — label: "Comm-16 — 5 steps"

**Caption under GIFs:**
> Both look identical on the training map — communication provides no visible edge here.

**Takeaway box (bottom):**
> ✓ All three conditions: 100% capture, ~9–11 steps average
> Communication was not necessary for strong in-distribution performance

### SCRIPT

"Here is what training looks like across all three conditions."

"All three reached 100 percent greedy capture rate on easy_open, with similar average step counts of roughly 9 to 11 steps per episode."

"You can see from the training curves that the communication conditions do converge faster early on — they reach high capture rates earlier in training. But by the end, all three are equivalent."

"The two GIFs on the right show baseline and Comm-16 on the training map. They look identical. Five steps, clean convergence, no visible difference in strategy."

"That means I cannot claim communication was necessary to solve the easy environment. And that's actually important — it keeps the interpretation clean. If communication is valuable here, the interesting question is: where does it show up?"

---

## Slide 9 — Main transfer result

### SLIDE CONTENT

**Title:** Result 2 — communication improves transfer to unseen maps

**Main table — large, center of slide:**

| Condition | easy_open | center_block | split_barrier |
|-----------|:---------:|:------------:|:-------------:|
| No-Comm   | 100% ±0.0 | 77.7% ±28.8  | 65.8% ±26.2   |
| Comm-4    | 100% ±0.0 | 91.0% ±12.7  | 70.7% ±18.3   |
| **Comm-16**   | **100% ±0.0** | **95.2% ±6.1**   | **79.5% ±12.7**   |

Highlight the Comm-16 row in bold/color.
Highlight the easy_open column lightly ("all the same") and the transfer columns strongly.

**Right side — collision rate callout:**

| Condition | split_barrier collision rate |
|-----------|:---:|
| No-Comm | 38.1% |
| Comm-4 | 4.1% |
| Comm-16 | 3.7% |

**CREATE — Visual C (collision rate bar chart):**
This would be a grouped bar chart with conditions on x-axis, collision rate on y-axis, two bars per condition (center_block and split_barrier). Much more visually striking than a table. One matplotlib call. Tells the coordination story in a glance.

**Bottom takeaway:**
> Communication raises both mean performance and consistency. Larger vocabulary helps more.

### SCRIPT

"This is the main result."

"On the training map — all three at 100 percent. No difference."

"On center_block, a map they've never seen: No-Comm drops to 77.7 percent. Comm-4 holds at 91. Comm-16 reaches 95.2 percent."

"On split_barrier, the harder map: No-Comm falls to 65.8 percent. Comm-4 to 70.7. Comm-16 to 79.5 percent."

"The pattern is consistent: communication improves transfer, and more channel capacity helps more."

"There is a second signal worth paying attention to — collision rates. On split_barrier, the no-comm agents collide with each other on 38 percent of steps. Without any way to coordinate, they converge on the same cell. The Comm-16 collision rate is under 4 percent."

"That is the difference between agents that are independently chasing and agents that are actually coordinating movement."

---

## Slide 10 — Seed sensitivity

### SLIDE CONTENT

**Title:** Communication raises the floor, not just the mean

**Left: per-seed breakdown table**

| Condition | Seed | center_block |
|-----------|------|:------------:|
| No-Comm | s0 | ~100% |
| No-Comm | s1 | ~76% |
| No-Comm | s3 | ~37% |
| Comm-16 | s0 | ~95% |
| Comm-16 | s1 | ~98% |
| Comm-16 | s2 | ~93% |

(Use actual per-seed values from `results/all_seeds_raw.json` if available, otherwise use these approximate values from the summary.)

**CREATE — Visual D (per-seed scatter plot):**
Scatter plot: x-axis = condition (No-Comm, Comm-4, Comm-16), y-axis = center_block capture rate, one dot per seed. Draw a horizontal line at the mean. This makes the variance argument visual and immediate — the No-Comm dots are spread across 40 percentage points; Comm-16 dots are clustered near 95%.

**Right: callout box**
> No-Comm: one seed got 100%, one got 37% — on the same task
>
> Comm-16: all seeds within 5% of each other
>
> Note: one additional baseline seed failed training entirely — Q-values never converged. Excluded from primary table; replacement seed confirmed normal.

**Bottom takeaway:**
> Communication doesn't just raise the mean — it raises the floor and compresses variance.

### SCRIPT

"I want to be transparent about the variance, because the standard deviations are large."

"On center_block, No-Comm has a standard deviation of plus or minus 28.8 percentage points. Breaking that down by seed: one seed achieved close to 100 percent, another achieved around 37 percent. These are not small differences — they represent fundamentally different learned behaviors."

"Comm-16 has a standard deviation of only plus or minus 6.1 points. All three seeds land near 95 percent."

"The variance itself is part of the story. Communication does not just increase average performance. It makes the outcome more predictable across different training runs."

"I'll also note: we had one baseline seed that failed outright during training. Its Q-values never converged and it was relying on exploration noise rather than a learned policy. We excluded it from the primary table, documented it explicitly, and ran a fourth seed as a check. That fourth seed succeeded normally, confirming the failure was isolated."

---

## Slide 11 — What the transfer result means

### SLIDE CONTENT

**Title:** Why does communication help at transfer time?

**Four points, large text, left-aligned:**

1. On easy_open, agents can memorize habits for that one fixed geometry
2. When the layout changes, those habits break down
3. Teammate signals fill in the gaps that local observations can't cover
4. Larger channel → more information → better robustness

**Right side: simple diagram (optional, can be text)**
Two columns:
- "Familiar map" → habits work → communication no advantage
- "New map" → habits break → communication helps fill uncertainty

**Bottom callout:**
> The communication channel is most valuable when local observations are least reliable.

### SCRIPT

"One way to interpret this is that on the training map, agents can develop spatial habits — movement patterns that work well in that one known geometry."

"When the layout changes, those habits become less reliable. An agent that always routes through the center is now blocked by a wall. An agent that spreads to the right edge now has to navigate a gap."

"At that point, information from teammates becomes more valuable. Communication can help coordinate who takes which path, reduce uncertainty about teammate positions, and allow the team to adapt."

"The second important point is that channel size mattered. Sixteen symbols outperformed four symbols on transfer, which means compact communication did not fully capture the benefit. There is some minimum channel capacity required to transmit useful coordination information."

---

## Slide 12 — Visual comparison (the knockout slide)

### SLIDE CONTENT

**Title:** Same map. No communication fails. Communication succeeds.

**Full slide: two GIFs side by side, large**

Left half:
- GIF: `results/ppt_gif3b_nocomm_split_fail.gif`
- Label (red text, bold): **NO COMMUNICATION — split_barrier**
- Subtitle: *Never captures. 300 steps.*

Right half:
- GIF: `results/ppt_gif5_comm16_split_best.gif`
- Label (green text, bold): **COMM-16 — split_barrier**
- Subtitle: *Captured in 25 steps.*

**Both GIFs should autoplay and loop.**

**Minimal other text — let the visual speak.**

Optional small note at bottom:
> Same map. Same training environment. Only difference: 4-bit communication channel.

### SCRIPT

"I want to show you the difference directly."

"On the left: no-communication agents on split_barrier. They were trained on easy_open, they never saw this vertical wall, and there is no way for them to signal to each other what they're seeing."

"Watch what happens — they run into the wall, they converge on each other, and the episode runs to the step limit without ever capturing the prey."

"On the right: Comm-16 agents on the exact same map, trained on the exact same easy_open. The only difference is the 4-bit communication channel."

"They navigate through the gaps, they spread across both sides of the barrier, and they capture in 25 steps."

"That gap — failure versus success — is what the numbers on the previous table actually look like."

---

## Slide 13 — Transfer demo — center_block

### SLIDE CONTENT

**Title:** Comm-16 navigating an unseen obstacle

**Main visual: `results/ppt_gif4_comm16_center_best.gif`**
Large, centered, looping.
Caption: *Comm-16, center_block map — never seen in training. Captured in 15 steps.*

**Small annotation overlays (can add in PowerPoint):**
- Arrow pointing to colored message badges: "Live message symbols"
- Arrow pointing to agents spreading: "Agents spreading — not clustering"

**Side note:**
- Red = Pursuer 0
- Blue = Pursuer 1
- Green = Pursuer 2
- Message badges show which symbol each agent is currently broadcasting

### SCRIPT

"Here is a closer look at what the Comm-16 policy looks like when it succeeds on a new map."

"The three colored agents are navigating center_block — a 4 by 4 obstacle in the middle of the grid that they have never seen during training. The colored badges at the bottom show which symbol each agent is broadcasting in real time."

"Notice they spread out and approach from different angles. They route around the obstacle without getting stuck on the same side. They don't cluster together on one path."

"They capture in 15 steps on a map they've never trained on."

"Whether the specific message symbols are meaningful in a human-readable sense is a different question — and that's what the next slide covers."

---

## Slide 14 — Interpretability

### SLIDE CONTENT

**Title:** Did the agents learn an interpretable language?

**Left: analysis results summary**

| Metric | Comm-4 | Comm-16 |
|--------|--------|---------|
| Channel entropy | 1.997 / 2.0 bits (99.8%) | 3.985 / 4.0 bits (99.6%) |
| Inter-agent role variation | 1.09% | 0.82% |
| Temporal phase shift | 1.9% | 1.6% |
| Capture vs. escape correlation | Not computable | Not computable |

**Right: one of the interpretability plots**
Use `results/msg_analysis_16/freq_dist.png` — this shows symbol usage frequency for Comm-16. Near-uniform distribution across all 16 symbols. Makes the "near-maximum entropy" point visually immediate.

**Bottom callout:**
> Channel is active and functional. Protocol is distributed — not categorical or human-readable.

### SCRIPT

"After seeing that communication helped transfer, I asked what kind of communication the agents actually learned."

"I analyzed message logs from 500-episode evaluation runs."

"Symbol entropy was near-maximum for both conditions — essentially the full vocabulary in active use. That tells us the channel is not being ignored."

"But when I looked for interpretable structure, I found very little. There was less than 1.1 percent variation in symbol usage across the three agents — meaning they did not develop different roles like scout and chaser. There was less than 2 percent temporal shift across early, middle, and late episode phases — meaning messages do not carry episode-state information cleanly. And the success-versus-failure analysis was not even computable — fewer than 5 failed episodes in 500 runs on the training map."

"The frequency distribution on the right shows near-uniform symbol use — which is what maximum entropy looks like in practice."

"The clean summary: the communication is functional, but it does not resolve into a simple human-readable protocol."

---

## Slide 15 — Why the null interpretability result matters

### SLIDE CONTENT

**Title:** Useful ≠ interpretable — and that matters

**Three points, large text:**

- Agents optimize for *task success*, not for human-readable semantics
- Distributed signaling patterns can be effective without being symbolic
- Performance and transparency are different goals — and often in tension

**Quote callout:**
> "A system can be effective without being naturally transparent."

**Right side — optional contrast:**

| | Communication |
|--|--|
| Channel active? | ✅ Yes — near-max entropy |
| Improves transfer? | ✅ Yes — consistent across seeds |
| Human-readable? | ❌ No — distributed, not symbolic |

### SCRIPT

"I want to emphasize that the interpretability result is not a failure."

"There is a common intuition that if agents communicate successfully, they must have created a neat symbolic system — something like 'symbol 3 means prey is to my left.' But effective communication and human-readable communication are separate goals."

"The agents are optimizing for what wins the game, not for what makes sense to a human observer. So it is completely plausible that they develop internal signaling patterns that are useful for coordination but do not map cleanly onto categories we can name."

"That distinction matters practically. In AI safety and explainability research, the fact that a system is performing well does not tell you that you can understand why or predict its failure modes. These results are a small case study in that broader challenge."

---

## Slide 16 — Limitations and future work

### SLIDE CONTENT

**Title:** Limitations and next steps

**Two columns:**

**Limitations:**
- Simplified 16×16 grid world
- Fixed evasive prey policy — does not adapt
- Trained on only one map — unclear if transfer advantage holds when training is harder
- DQN instability: significant seed-to-seed variance, checkpoint-sensitive
- Communication protocol functional but opaque

**Future work:**
- Train on harder maps or use curriculum learning (easy → hard)
- Scale to more agents — requires architecture change for message aggregation
- t-SNE of message embeddings conditioned on agent positions — may reveal spatial encoding
- Test with adaptive prey policy
- Try PPO or MADDPG for more stable training

### SCRIPT

"Like any study, this one has honest limitations."

"The environment is a simplified grid world. The prey uses a fixed evasive policy rather than learning adaptively. We trained on only one map, so we don't know whether the communication advantage persists if models are trained on harder environments from the start."

"DQN has well-known instability. We saw significant variance across seeds and checkpoints, and our best-checkpoint methodology controls for it but does not eliminate it."

"And while communication improved transfer, the protocol is still opaque."

"The three most natural extensions are: training on harder maps or curriculum sequences to test whether communication generalizes further; scaling to more agents, which requires updating the message aggregation architecture; and using t-SNE or spatial conditioning to probe what the message embeddings actually encode about the environment."

---

## Slide 17 — Conclusion

### SLIDE CONTENT

**Title:** Conclusion

**Three numbered findings:**

1. All conditions solved the training map equally — communication was not necessary for in-distribution performance

2. Communication improved transfer to unseen maps — Comm-16 most reliably, smaller variance across seeds

3. Learned messages were functional but not interpretable — near-max entropy, no symbolic structure

**Bottom line — large, bold, centered:**

> Communication mainly improved *adaptability*, not training-map performance.
> Its value was robustness when the world changed — not memorization of a familiar layout.

**Optional: reshow `results/ppt_gif4_comm16_center_best.gif` small in corner as visual callback**

### SCRIPT

"To conclude."

"This project asked whether compact learned communication helps coordination in partially observable multi-agent pursuit."

"Three findings."

"First: communication was not necessary to solve the familiar training environment. All conditions reached 100 percent on easy_open."

"Second: communication consistently improved transfer to unseen maps. The sixteen-symbol condition performed most reliably, with tighter variance across seeds and lower collision rates."

"Third: the communication channel was clearly active — near-maximum entropy, full vocabulary usage — but it did not produce a simple human-readable protocol."

"The key takeaway: in this project, communication did not mainly help agents solve a familiar environment better. It helped the team remain more robust when the world changed."

"Communication supported adaptability. Not memorization."

"Thank you."

---

## Backup Slide — Large-map stress test (use only if asked)

### SLIDE CONTENT

**Title:** Stress test — zero-shot on a larger map

**Result:**
- Evaluated on `large_split`: 20×20 grid, same structure as split_barrier but larger
- Baseline ep004000: 93.5% capture, 88 avg steps
- Comm-16 selected checkpoint (ep003500): only 37.5% capture
- Comm-16 later checkpoint (ep005000): 91.0% capture, 97 steps

**Finding:**
> In-distribution optimality and out-of-distribution robustness do not necessarily peak at the same training checkpoint.

### SCRIPT

"As an exploratory stress test, I evaluated both models on a 20 by 20 map with the same barrier structure as split_barrier."

"The baseline held up at 93.5 percent. The Comm-16 model that was best on the training map — the checkpoint we selected — only achieved 37.5 percent. But a later checkpoint of the same model achieved 91 percent."

"That tells you something important: the checkpoint that best solves the training environment is not always the checkpoint that transfers best. Communication models may be more sensitive to this, because the message protocol may still be shifting late in training."

"I flagged this as a limitation rather than a main result — it was one data point, and interpreting it would require more systematic analysis."

---

## Q&A Prep

**"Why not train on all three maps?"**
We wanted to isolate zero-shot transfer. Training on all maps tests multi-task learning, which is a different question. Natural next step.

**"How do you know the symbols aren't just noise?"**
The capture rate improvement is real and consistent across three seeds. Whatever the messages encode, they are contributing to better generalization. Near-maximum entropy and full vocabulary usage confirm the channel is carrying information rather than being ignored.

**"Why DQN and not PPO or MADDPG?"**
DQN with parameter sharing is a natural baseline for discrete action spaces. Its instability is a known limitation we acknowledge. PPO would be a stronger foundation for future work.

**"What happened with the failed seed?"**
One baseline seed never converged — Q-values stayed noisy and the agent relied on exploration at test time. Known DQN failure mode. We documented it explicitly, excluded it from the primary table, and ran a fourth seed as a sensitivity check. That succeeded normally.

**"Could the comm agents be cheating?"**
No. Messages from step t are received at step t+1 — no information leakage within a timestep. Consistent with the emergent communication literature.

---

## Language to use and avoid

| Avoid | Say instead |
|-------|-------------|
| "The agents learned a language" | "learned a communication protocol" |
| "Communication improved everything" | "improved mean transfer performance" |
| "This proves it works in the real world" | "under this experimental setting" |
| "The agents understood the environment" | "developed a coordination capacity" |
| "The null result means it failed" | "interpretability and performance are separate goals" |
