# Comm-4 Message Interpretability Analysis

**Source:** `results/comm4_v2/comm4_msg_log_seed0.json`  
**Total messages analyzed:** 37,026  
**Vocabulary size:** 4 symbols

---

## 1. Symbol Frequency Distribution

Entropy: **1.9967 bits** (max possible = 2.0 bits, utilization = **99.8%**)  

Symbol usage percentages:

- Symbol 0: 26.95%
- Symbol 1: 23.45%
- Symbol 2: 26.39%
- Symbol 3: 23.21%


**Interpretation:** The vocabulary is used nearly uniformly (99.8% of max entropy). No symbol dominates — agents are distributing meaning across all 4 symbols.

---

## 2. Per-Agent Symbol Usage

Average inter-agent variation (std across agents per symbol): **1.09%**

Per-agent channel entropy (bits):

- P0: 1.9948 bits
- P1: 1.995 bits
- P2: 1.9962 bits


**Interpretation:** All three agents use symbols in very similar proportions — no role differentiation is evident. The shared policy has not specialized different agents to send different message types.

---

## 3. Temporal Analysis (Symbol Usage by Episode Phase)

Maximum phase shift per symbol (early vs late usage %):

- Symbol 0: Δ1.4%
- Symbol 1: Δ1.9%
- Symbol 2: Δ1.5%
- Symbol 3: Δ1.2%


**Interpretation:** Symbol usage is nearly uniform across episode phases. Messages do not encode time-into-episode information.

---

## 4. Capture vs Escape Correlation

**Not computed** — only 2 escaped episode(s) out of 500 total (minimum required: 20).

The model almost never fails on this map/checkpoint combination, so there is insufficient data to compare message distributions between captured and escaped episodes. Re-run on a harder map or a weaker checkpoint to obtain a meaningful sample of failures.

---

## 5. Summary

| Metric | Value |
|--------|-------|
| Vocab size | 4 |
| Channel entropy | 1.9967 / 2.0 bits (99.8%) |
| Dominant symbol | 0 (26.9%) |
| Inter-agent variation | 1.09% avg std |
| Max temporal shift | 1.9% |
| Max capture/escape divergence | not computed (2 escaped ep < 20 required) |
