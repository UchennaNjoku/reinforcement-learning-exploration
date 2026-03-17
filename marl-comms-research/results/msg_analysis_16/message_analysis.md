# Comm-16 Message Interpretability Analysis

**Source:** `results/comm16_v2/comm16_msg_log_seed0.json`  
**Total messages analyzed:** 36,315  
**Vocabulary size:** 16 symbols

---

## 1. Symbol Frequency Distribution

Entropy: **3.9847 bits** (max possible = 4.0 bits, utilization = **99.6%**)  

Symbol usage percentages:

- Symbol 0: 6.88%
- Symbol 1: 5.55%
- Symbol 2: 5.98%
- Symbol 3: 6.33%
- Symbol 4: 5.65%
- Symbol 5: 5.77%
- Symbol 6: 6.18%
- Symbol 7: 6.5%
- Symbol 8: 5.72%
- Symbol 9: 5.39%
- Symbol 10: 8.53%
- Symbol 11: 5.8%
- Symbol 12: 5.93%
- Symbol 13: 6.75%
- Symbol 14: 8.16%
- Symbol 15: 4.89%


**Interpretation:** The vocabulary is used nearly uniformly (99.6% of max entropy). No symbol dominates — agents are distributing meaning across all 16 symbols.

---

## 2. Per-Agent Symbol Usage

Average inter-agent variation (std across agents per symbol): **0.82%**

Per-agent channel entropy (bits):

- P0: 3.9676 bits
- P1: 3.96 bits
- P2: 3.98 bits


**Interpretation:** All three agents use symbols in very similar proportions — no role differentiation is evident. The shared policy has not specialized different agents to send different message types.

---

## 3. Temporal Analysis (Symbol Usage by Episode Phase)

Maximum phase shift per symbol (early vs late usage %):

- Symbol 0: Δ1.2%
- Symbol 1: Δ1.4%
- Symbol 2: Δ0.9%
- Symbol 3: Δ0.7%
- Symbol 4: Δ1.0%
- Symbol 5: Δ0.4%
- Symbol 6: Δ0.5%
- Symbol 7: Δ0.4%
- Symbol 8: Δ0.4%
- Symbol 9: Δ0.4%
- Symbol 10: Δ1.6%
- Symbol 11: Δ0.9%
- Symbol 12: Δ0.7%
- Symbol 13: Δ0.6%
- Symbol 14: Δ1.3%
- Symbol 15: Δ0.1%


**Interpretation:** Symbol usage is nearly uniform across episode phases. Messages do not encode time-into-episode information.

---

## 4. Capture vs Escape Correlation

**Not computed** — only 1 escaped episode(s) out of 500 total (minimum required: 20).

The model almost never fails on this map/checkpoint combination, so there is insufficient data to compare message distributions between captured and escaped episodes. Re-run on a harder map or a weaker checkpoint to obtain a meaningful sample of failures.

---

## 5. Summary

| Metric | Value |
|--------|-------|
| Vocab size | 16 |
| Channel entropy | 3.9847 / 4.0 bits (99.6%) |
| Dominant symbol | 10 (8.5%) |
| Inter-agent variation | 0.82% avg std |
| Max temporal shift | 1.6% |
| Max capture/escape divergence | not computed (1 escaped ep < 20 required) |
