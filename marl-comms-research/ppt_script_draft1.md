# Presentation Script — Draft 1
**"Emergent Communication Protocols for Coordination in Partially Observable Multi-Agent Pursuit"**
U. Njoku, J. Calderon — Bethune-Cookman University
**Target: 15 minutes**

---

## Slide 1 — Title (0:00–0:30)

Good morning / afternoon everyone. My name is Uchenna Njoku, and today I'm presenting our work on emergent communication in multi-agent reinforcement learning — specifically, whether agents that can talk to each other learn to coordinate better than agents that can't, and whether that advantage holds up when you put them in environments they've never seen before.

---

## Slide 2 — Motivation (0:30–1:30)

Let's start with why this matters.

When you have multiple autonomous agents working toward a shared goal — think drone swarms, search and rescue robots, or autonomous vehicles merging on a highway — they face a fundamental problem: each agent only sees a small piece of the world, and none of them can directly observe what the others know.

The natural question is: if you give these agents a communication channel, do they actually use it to coordinate? And more importantly — does whatever they learn in training still work when the environment changes?

That's the question we set out to answer.

---

## Slide 3 — Setup: The Environment (1:30–2:45)

We used the PettingZoo Pursuit environment, a standard benchmark for cooperative multi-agent reinforcement learning. The setup is:

- A **16x16 grid world**
- **3 pursuer agents** — these are the ones we control and train
- **1 evader** that moves randomly
- Each agent sees only a **7x7 local window** centered on itself — so no agent can see the full map
- Capture happens when any pursuer reaches the evader's cell

**Show map diagram or easy_open screenshot**

We designed three maps. The first — *easy_open* — is a clear field with no obstacles. This is where all three conditions are trained. The second — *center_block* — has a central obstacle blocking the direct path. The third — *split_barrier* — has a vertical wall with two narrow gaps, forcing agents to navigate around it.

The key point: **agents are trained only on easy_open, then tested on center_block and split_barrier without any additional training.** That's the transfer test.

---

## Slide 4 — Three Conditions (2:45–3:30)

We train three conditions:

**No-Comm** — a standard DQN baseline. Each agent has its own Q-network. No communication at all.

**Comm-4** — agents can send a discrete symbol drawn from a vocabulary of 4 symbols each step. Those symbols are fed as input to every other agent's network the next step.

**Comm-16** — same architecture, but vocabulary size is 16. More expressive.

Critically, **no one tells the agents what the symbols mean.** There's no predefined protocol. If a meaningful communication language emerges, the agents figure it out on their own through training.

We trained 3 independent random seeds per condition — 9 training runs total — to make sure our results aren't just one lucky seed.

---

## Slide 5 — Architecture (3:30–4:15)

**Show architecture diagram if you have one, otherwise describe briefly**

The core model is a DQN with parameter sharing — all three pursuers share one network. The input is the 7x7x3 local observation: one channel for walls, one for ally positions, one for evader positions.

For the comm models, there are two output heads: a **move head** that selects one of 5 actions (up, down, left, right, stay), and a **message head** that selects a symbol to broadcast. The messages from the previous step are concatenated to the observation of each receiving agent before it makes its decision.

The communication is **differentiable and end-to-end trained** — the agents learn when and what to say by learning what wins the game.

---

## Slide 6 — Training Curves (4:15–5:15)

**Show comm_training_curves.png**

This is what training looks like across the three conditions, using seed 0 as representative.

A few things to notice. First, **Comm-4 and Comm-16 converge faster than the baseline** — they reach high capture rates earlier in training. Second, all three conditions eventually reach near-perfect performance on easy_open, which is the training map — so all models learned the task.

Third — and this is important for the next part — we did **not** just use the final checkpoint. Because DQN training can be unstable, we swept all checkpoints saved every 500 episodes and selected the best-performing one on easy_open for each run. This gives a fair comparison that isn't sensitive to which exact episode we happened to stop training.

---

## Slide 7 — In-Distribution Results (5:15–6:00)

**Show rollout GIFs: baseline_easy, comm4_easy, comm16_easy side by side**

Here are representative rollout GIFs of all three conditions on easy_open — the training map.

All three capture the evader successfully and quickly. Numerically, all three conditions achieve **100% capture rate** with average episode lengths around 9–10 steps. There is no meaningful difference in performance on the training map.

This is important — it tells us all three conditions have learned a capable policy. The question is what happens when the map changes.

---

## Slide 8 — Transfer Results (6:00–7:30)

**Show sweep_summary table**

This is the main result. Each number is the mean capture rate across 3 independent training seeds, evaluated on 200 episodes.

On **easy_open** — the training map — all three conditions are at 100%. No difference.

On **center_block** — a map with a central obstacle they've never seen — No-Comm drops to **77.7%**. Comm-4 holds at **91%**. Comm-16 reaches **95.2%**.

On **split_barrier** — a map with a full vertical wall and narrow gaps — No-Comm falls to **65.8%**. Comm-4 to **70.7%**. Comm-16 to **79.5%**.

The pattern is consistent: **communication improves transfer performance, and larger vocabulary improves it further.** The communication advantage doesn't appear in training — it appears when the environment changes.

Notice also the collision rates. The no-comm agents collide with each other frequently on the barrier maps — up to 38% of steps — because they have no way to coordinate who goes where. The comm agents' collision rates are dramatically lower, under 5%, because they're coordinating movement.

---

## Slide 9 — Transfer GIF Demo (7:30–8:30)

**Show rollout_comm16_center.gif**

Let me show you what that 95% capture rate actually looks like.

This is Comm-16 on center_block — a map it has never trained on. Watch how the three agents — red, blue, green — spread out and approach the evader from different angles. The colored badges at the bottom show which symbol each agent is currently broadcasting.

They navigate around the obstacle, they don't cluster together, and they complete the capture in about 60 steps.

**Show baseline on center_block if available, or describe verbally**

A no-comm agent on the same map is more likely to get stuck on the same side of the obstacle as its teammates, or bounce between the same two cells waiting for the evader to come to it. That's where the 22-point capture rate gap comes from.

---

## Slide 10 — Seed Sensitivity (8:30–9:15)

I want to be transparent about the variance here, because the standard deviations are large.

On center_block, No-Comm has a standard deviation of **±28.8 percentage points**. That means one seed achieved 100%, another achieved 37%. That's not a small difference — it tells you the baseline policy is genuinely brittle to map changes. Some seeds learn a strategy that happens to transfer; others don't.

Comm-16's standard deviation is **±6.1** — substantially more consistent across seeds.

We also had one baseline seed that failed outright during training — its Q-values never converged, and it only achieved 51.5% capture even on easy_open. We excluded it from the primary table and noted it honestly. We ran a fourth baseline seed as a sensitivity check, which succeeded normally.

The variance itself is part of the story: **communication doesn't just raise the mean, it raises the floor.**

---

## Slide 11 — Message Interpretability (9:15–10:15)

**Show agent_freq.png or message_analysis.md summary**

A natural question is: what are the agents actually saying? Do the symbols mean anything?

We analyzed the message logs from a 500-episode evaluation run with Comm-16.

**Symbol entropy** is nearly maximal — agents use all 16 symbols with roughly equal frequency, which means they're not just defaulting to one signal. The full vocabulary is being used.

**Role differentiation** is minimal — the three agents use symbols with less than 1.1% variation between them. They're not developing specialized roles where one agent is always the "scout" and another the "chaser."

**Temporal structure** is also near-zero — there's no clear pattern of agents switching symbols as the episode progresses.

**Capture vs. escape correlation** — we couldn't compute this meaningfully because the models succeed on easy_open in nearly 100% of episodes. With fewer than 5 failed episodes in 500, there's no statistical signal.

The honest interpretation: the agents are using the communication channel, the vocabulary is active, but we cannot yet identify what specific environmental signals the symbols encode. That's a known challenge in emergent communication research and points directly to future work.

---

## Slide 12 — Discussion (10:15–11:15)

So what does this add up to?

Our core finding is that **the benefit of communication is a generalization benefit, not an in-distribution benefit.** If you only look at the training map, you'd conclude communication is unnecessary — all three conditions perform equally. The communication advantage only appears when you test on new maps.

This makes intuitive sense. On easy_open — a clean field with one evader — three agents can independently converge on the target without coordinating. On center_block or split_barrier, the agents need to navigate around obstacles, and that requires some implicit division of approach directions. The comm agents developed the capacity to signal information that helps with that — even without being told that transfer maps would exist.

The Comm-16 advantage over Comm-4 suggests that **more expressive communication is better for transfer**, even though Comm-4 is sufficient for the training task. The extra symbols provide headroom for richer coordination.

---

## Slide 13 — Limitations (11:15–12:00)

A few honest limitations.

First, **we trained only on easy_open.** We don't know whether the transfer advantage would persist if models were trained on harder maps from the start. It's possible the communication protocol is specifically calibrated to easy_open's geometry.

Second, **the interpretability analysis is limited** by the near-perfect capture rate on the evaluation set. We need harder environments that produce more failures to separate captured from escaped message patterns.

Third, **DQN is unstable.** We saw significant variance across seeds and checkpoints within seeds. Our best-checkpoint selection methodology controls for this, but it's worth noting that the results are not as clean as they would be with a more stable algorithm like PPO.

Fourth, **the communication protocol is emergent but opaque.** We know the symbols are used and they correlate with better transfer performance, but we cannot yet say what they encode.

---

## Slide 14 — Future Work (12:00–12:45)

There are three natural extensions.

**Training on harder maps** — training directly on center_block or split_barrier, or using curriculum learning, would let us test whether communication is even more valuable when coordination is required during training.

**More agents** — scaling to 4 or 5 pursuers would require more coordination by design, and would make the message-passing more interesting to analyze.

**Interpretability** — applying t-SNE to the message embedding space, or analyzing messages conditioned on relative agent positions, could reveal what spatial information the symbols encode.

---

## Slide 15 — Conclusion (12:45–13:30)

To summarize:

We trained three conditions of multi-agent DQN — no communication, 4-symbol communication, and 16-symbol communication — on a cooperative pursuit task, and evaluated zero-shot transfer to unseen maps.

All three conditions achieve 100% capture on the training map. On transfer maps, Comm-16 outperforms Comm-4 outperforms No-Comm — by up to 13 percentage points in capture rate and with dramatically lower collision rates.

The communication benefit is not about learning faster. It's about **building a coordination capacity that generalizes when the environment changes.** Agents that can communicate develop a more robust shared strategy, even without knowing in advance that they'll need it.

Thank you. I'm happy to take questions.

---

## Q&A Prep (13:30–15:00)

**"Why not train on all three maps?"**
That's a great next step. We wanted to isolate zero-shot transfer as the test — training on all maps would tell you about multi-task learning, which is a different question. Future work.

**"How do you know the symbols aren't just noise?"**
The capture rate improvement is real and consistent across 3 seeds. Whatever the messages encode, they're contributing to better generalization. That the symbols are non-trivially distributed (near-max entropy, full vocabulary usage) suggests they're carrying information rather than being ignored.

**"Why DQN and not PPO or MADDPG?"**
DQN with parameter sharing is a natural baseline for discrete action spaces and a standard in the MARL literature. Its instability is a known limitation we acknowledge. PPO would be a stronger foundation for future work.

**"What does baseline_s2 failure mean?"**
One of three baseline seeds never converged — Q-values stayed noisy and the agent relied on epsilon-exploration to capture at training time, but failed completely at greedy evaluation. This is a known DQN failure mode and is why we ran multiple seeds. We excluded it from the primary table and noted it explicitly.

**"Could the comm agents be cheating somehow?"**
No. The messages from step t are received at step t+1, so there's no information leakage. The communication protocol is the same as described in the emergent communication literature.
