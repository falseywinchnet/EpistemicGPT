Reasoned thus:

# Minimum-Surprise Byte Pair Encoding (MS-BPE)

*A capacity-aware, information-theoretic refinement of BPE-style subword tokenization*

---

## Section 1 – Concept and Motivation

Modern subword tokenizers such as Byte Pair Encoding (BPE) and unigram LM segment text into variable-length units that balance vocabulary size, compression, and open-vocabulary coverage.([arXiv][1]) These schemes optimize primarily for **frequency-based compression** and downstream loss, with few structural constraints. They often:

* Merge across punctuation and other “regime change” symbols.([ACL Anthology][2])
* Produce tokens that hide important internal structure (e.g. decimals, affixes).
* Ignore the finite information-processing capacity of the model that consumes the tokens.

Meanwhile:

* Written English has an entropy on the order of **~0.6–1.1 bits per character**.([mattmahoney.net][3])
* GPT-style transformers have an empirical memorization capacity of about **3.6 bits per parameter**.([arXiv][4])
* Compute/data-optimal scaling laws suggest training at about **20 tokens per parameter** (Chinchilla-style).([Epoch AI][5])

These facts together imply a **finite information budget per training token**. If tokenization creates highly surprising, heterogeneous tokens, the model is effectively asked to encode more local information than its capacity allows, increasing effective perplexity.

**Minimum-Surprise Byte Pair Encoding (MS-BPE)** is a variant of BPE that explicitly **constrains local surprise** (information density) at the token level, using:

* A **per-character surprise cost** derived from corpus statistics.
* A **rolling surprise limit** when building composite tokens.
* **Static caps** on token self-information, calibrated to model bits/parameter and tokens/parameter scaling.

The goal is not maximal compression, but **capacity-compatible information density**: tokens that are easy for finite models to learn and reason over, while still providing strong compression and open-vocabulary coverage.

---

## Section 2 – Background

### 2.1 Subword tokenization

BPE and related methods were introduced to handle rare words and open vocabularies by composing words from subword units.([arXiv][1])

* **BPE**: start from characters, iteratively merge the most frequent adjacent pair until the desired vocab size is reached.([arXiv][1])
* **Unigram LM**: start from a large candidate vocabulary, then prune tokens based on a unigram language model and ML segmentation.([arXiv][6])

Empirical work has shown that:

* Unigram LM tokenization tends to align better with morphology and can improve LM pretraining performance over vanilla BPE.([ACL Anthology][2])
* BPE often absorbs common affixes and punctuation into neighbors, creating semantically incoherent units.([ACL Anthology][2])

MS-BPE takes BPE’s simple merge mechanism but adds **explicit information-theoretic constraints** to avoid the worst of these pathologies.

### 2.2 Entropy of text

Classical and modern estimates place the entropy rate of written English at approximately **0.6–1.3 bits per character**, with refined estimates suggesting **≤1.1 bpc**.([mattmahoney.net][3])

If a tokenizer produces on average ~4 characters per token, that corresponds to **~2.4–4.4 bits of inherent self-information per token** in the raw source, before modeling.

### 2.3 Model capacity: bits per parameter and tokens per parameter

Recent work measuring memorization capacity of GPT-style transformers finds:

* Capacity ≈ **3.6 bits per parameter** when generalization is removed and only memorization is measured.([arXiv][4])

Chinchilla-like scaling suggests:

* Data/parameter ratio ≈ **20 tokens per parameter** for compute-optimal pretraining of large LMs (1.4T tokens for a 70B-param model).([Epoch AI][5])

Naively spreading memorization capacity across data gives:

[
\text{Capacity per training token}
\approx \frac{3.6\text{ bits/param}}{20\text{ tokens/param}}
\approx 0.18\text{ bits/token}.
]

So each training token carries **far more inherent information (3–5 bits)** than the model could possibly memorize idiosyncratically; the majority must be captured via **shared structure** rather than token-specific special casing.

MS-BPE is explicitly designed to **respect this imbalance**, by preventing individual atomic tokens from concentrating more self-information than a finite model can reasonably absorb.

---

## Section 3 – Formal Setting

Let:

* Σ be the base alphabet (bytes or Unicode code points).
* A corpus (C = x_1,\dots,x_n) with (x_i \in \Sigma).
* A tokenizer (T) maps characters to tokens: (T(C) = z_1,\dots,z_m) from vocabulary (V).
* Each token (z \in V) is a string over Σ, decodable to its character sequence.

Define the **empirical token distribution**:

[
p(t) = \frac{\text{count}(t)}{\sum_{t' \in V} \text{count}(t')}.
]

and **self-information** (“surprise”) of a token type:

[
I(t) = -\log_2 p(t).
]

For character-level modeling, define:

* Character frequency (p(c)), bigram and trigram statistics (p(c_i, c_{i+1}), p(c_{i-1}, c_i, c_{i+1})), etc.

The objective of MS-BPE is to construct (V) and (T) such that:

1. **Reversibility** is preserved (as in normal BPE).
2. The **local information density** per token is bounded by capacity-consistent thresholds.
3. The resulting tokenization supports **low effective bits-per-character** for finite-capacity transformers, under realistic data/compute regimes.

---

## Section 4 – Per-Character Surprise Cost

MS-BPE starts with a **per-character cost function** that approximates local information density and “surprisefulness.”

For each character (c \in \Sigma):

1. **Rarity term**
   Let (p(c)) be the empirical character frequency. Define:

   [
   \text{cost}_\text{freq}(c) = \alpha \cdot (-\log_2 p(c)),
   ]

   with (\alpha > 0). Rare characters get higher cost.

2. **Context imbalance term**
   Estimate directional trigram distributions:

   * (T_\text{prev}(xy \mid c) = p(x,y \mid c)): distribution over 2-character prefixes ending at c.
   * (T_\text{next}(yz \mid c) = p(y,z \mid c)): distribution over 2-character suffixes starting at c.

   Compute either:

   * Entropy difference:
     [
     H_\text{prev}(c) = -\sum_{xy} T_\text{prev}(xy \mid c)\log_2 T_\text{prev}(xy \mid c)
     ]
     [
     H_\text{next}(c) = -\sum_{yz} T_\text{next}(yz \mid c)\log_2 T_\text{next}(yz \mid c)
     ]
     and set (\Delta H(c) = |H_\text{prev}(c)-H_\text{next}(c)|); or

   * Symmetrized divergence:
     [
     D(c) = D_\text{KL}(T_\text{prev} \Vert T_\text{next})
     + D_\text{KL}(T_\text{next} \Vert T_\text{prev}).
     ]

   Then define:

   [
   \text{cost}_\text{imb}(c) = \beta \cdot g(c),
   ]

   where (g(c)) is either (\Delta H(c)) or (D(c)), and (\beta > 0).

Characters with **many possible predecessors but few restricted successors** (punctuation, decimal points, currency symbols, brackets) will typically exhibit high imbalance and thus high cost.

3. **Total per-character cost**

[
\text{cost}(c) = \text{cost}*\text{freq}(c) + \text{cost}*\text{imb}(c).
]

Intuition:

* Common, well-balanced letters (e.g. vowels, frequent consonants) have low cost and are freely mergeable.
* Rare or context-asymmetric characters (digits in odd contexts, punctuation, symbols) have high cost and resist being buried in large tokens.

---

## Section 5 – Surprise Constraints

MS-BPE imposes two levels of surprise constraint:

### 5.1 Rolling surprise limit (per composite token)

When building a composite token from a character sequence (c_1,\dots,c_k), MS-BPE maintains a **rolling surprise sum**:

[
C = \sum_{i=1}^{k} \text{cost}(c_i).
]

A token is only valid if:

[
C \le \tau_\text{inst},
]

for some **instantaneous surprise limit** (\tau_\text{inst}).

Operationally, when proposing merges during vocabulary construction:

* Any candidate whose constituent characters would yield (C > \tau_\text{inst}) is forbidden.
* This prevents creating tokens whose internal information density is too high: they are forced to be represented as multiple tokens.

This rolling limit replaces ad hoc rules like “no composite immediately after a punctuation mark” with a single, unified constraint.

### 5.2 Static surprise cap on token types

Independently of internal character structure, MS-BPE places a static cap on **token-level self-information**:

[
I(t) = -\log_2 p(t) \le \tau_\text{type}.
]

Any token type whose empirical surprisal would exceed (\tau_\text{type}) in the corpus is not allowed into the vocabulary; its character sequence must be decomposed into smaller sub-tokens with lower individual surprisal.

This prevents extremely rare, long tokens (e.g. entire URLs, long numeric strings, or exotic symbols+text bundles) from entering the codebook as atomic items.

### 5.3 Capacity grounding for τ parameters

We can link (\tau_\text{type}) and (\tau_\text{inst}) to **model capacity**:

* Total model capacity:
  (B_\text{param} \approx 3.6 N) bits for N parameters.([arXiv][4])

* Data volume at Chinchilla-like scaling:
  (D \approx 20N) training tokens.([Epoch AI][5])

* Average capacity per training token:
  (B_\text{token} \approx 0.18) bits.

Now choose a fraction (f) of total parameter capacity that may be devoted to **idiosyncratic modeling of rare tokens** (e.g. (f \in [0.05, 0.2])):

[
B_\text{rare} = f \cdot B_\text{param}.
]

Let (I(t) = -\log_2 p(t)), and (\text{count}(t)) be occurrences. Define:

[
S(\tau_\text{type}) = \sum_{{t : I(t) > \tau_\text{type}}} \text{count}(t)\cdot I(t).
]

Choose (\tau_\text{type}) such that:

[
S(\tau_\text{type}) \le B_\text{rare}.
]

This ensures that the total information mass in high-surprise tokens is compatible with the model’s memorization budget; the rest must be captured compositionally.

For (\tau_\text{inst}), use a similar argument over windows (e.g., sequences within which merges are allowed) to ensure that no **single composite span** encodes more bits of self-information than the model’s effective window capacity can handle, given training exposure.

---

## Section 6 – MS-BPE Algorithm

MS-BPE is a constrained variant of standard BPE, which typically does:

1. Start from character vocabulary (V_0 = \Sigma).
2. Repeatedly merge the most frequent adjacent pair until reaching desired vocab size.([ACL Anthology][2])

MS-BPE modifies both vocabulary construction and tokenization:

### 6.1 Precomputation

1. Collect character-level statistics from corpus:

   * (p(c)), bigrams, trigrams.
2. Compute (\text{cost}(c)) for all (c \in \Sigma).
3. Decide (\tau_\text{inst}, \tau_\text{type}) from capacity arguments (Section 5.3).

### 6.2 Constrained vocabulary construction

Initialize:

* (V = \Sigma).
* For each character (c), define trivial token type with cost (\text{cost}(c)).

Loop until |V| reaches target size:

1. Count all adjacent pairs of current tokens over corpus.

2. For each candidate pair ((u,v)):

   * Let s be the concatenated character sequence of u followed by v.
   * Compute rolling surprise cost (C(s) = \sum_{c \in s} \text{cost}(c)).
   * Compute approximate token probability (p(s)) by how often (u,v) appear contiguous; derive (I(s) = -\log_2 p(s)) or a smoothed estimate.

3. **Feasibility filters**:

   * Reject s if (C(s) > \tau_\text{inst}).
   * Reject s if (I(s) > \tau_\text{type}).

4. Among remaining feasible candidates, select a merge according to a standard BPE-like criterion (e.g. highest frequency, or frequency times length gain).

5. Add s as new token to V; update corpus representation by replacing each occurrence of (u,v) with s.

Result:

* A vocabulary where all tokens meet local surprise constraints.

### 6.3 Tokenization procedure

Given new text:

1. Pre-tokenize minimally (e.g. bytes or characters; optional simple rules for whitespace).
2. Apply standard BPE-style greedy merges in the same order they were learned.
3. Optionally enforce rolling surprise at runtime: if a merge would cause (C > \tau_\text{inst}) on the fly, skip that merge even if it is in the learned merge list.

This guarantees both **static** and **instantaneous** surprise bounds for any tokenization of any string.

---

## Section 7 – What MS-BPE Buys

### 7.1 Capacity-aware information density

By construction, MS-BPE ensures that:

* No token type carries more self-information than allowed by (\tau_\text{type}), which is calibrated to model bits/parameter and data/parameter scaling.([arXiv][4])
* No composite token encodes a span whose internal character-level surprise exceeds (\tau_\text{inst}).

This aligns local information density with **what a finite transformer can actually store and reuse**, rather than with the theoretical entropy of the source alone.

### 7.2 Better handling of punctuation, numerals, and symbols

Because punctuation and special symbols typically:

* Are frequent but highly context-imbalanced (very different distributions before vs after),([ACL Anthology][2])
* Or are simply rare,

their cost is high, so:

* They tend to stay as **atomic tokens** or form only very short composites.
* Decimal points, currency signs, quotes, brackets etc. **do not get buried inside long tokens**, preserving visible structure for numeric and syntactic reasoning.

Examples:

* “9.11” → tokens like “9”, “.”, “11” or “9”, “.”, “1”, “1”, not a single opaque token.
* “$bananas” → “$”, “ba”, “na”, “nas” (or similar), keeping “$” separate from the word stem.

### 7.3 Morphology and word structure

MS-BPE shares some goals with unigram LM segmentation:

* Frequent, balanced character sequences (morpheme-like) remain mergeable and form relatively long, reusable tokens.([ACL Anthology][2])
* Tokens that cross high-surprise boundaries (e.g. affix+punctuation) are disfavored or forbidden.

This tends to produce vocabularies that:

* Align better with morphology and word-internal structure.([ACL Anthology][2])
* Avoid BPE’s tendency to absorb common suffixes and punctuation into neighbors crudely.([ACL Anthology][2])

### 7.4 Stabilizing effective perplexity

Empirical work shows that:

* Tokenization choice impacts LM pretraining efficiency and downstream performance; e.g. unigram LM can outperform BPE, especially in typologically diverse languages.([ACL Anthology][2])

MS-BPE aims to further stabilize **character-level bits-per-character** by:

* Preventing extremely high-surprise tokens from ever appearing as atomic units.
* Forcing rare structures to be represented compositionally across lower-surprise tokens.
* Letting the model exploit its capacity more on **structure** and less on memorizing idiosyncratic rare tokens.

The expectation is a smoother scaling curve: fewer tokenization-induced spikes in loss and better utilization of parameters at a given model size.

---

## Section 8 – Necessary Truths and Assumptions

MS-BPE rests on several assumptions made explicit:

1. **Finite, measurable model capacity**

   * Transformers have a finite memorization capacity measured in bits per parameter (~3.6).([arXiv][4])
   * This capacity is meaningfully filled at realistic data scales; beyond that, further data must be “compressed” into shared structure rather than memorization.

2. **Data/parameter scaling regime**

   * Training is near a data/compute-optimal regime (e.g. ~20 tokens/parameter).([Epoch AI][5])
   * Under such regimes, the effective **capacity per training token is small** (~0.1–0.2 bits), so token-level self-information must be constrained.

3. **Stable text entropy regime**

   * For a given domain (e.g. English web text), character-level entropy is approximately constant (around 1 bpc).([mattmahoney.net][3])
   * Tokenization cannot change this entropy, only redistribute it across tokens.

4. **Tokenization as lossy quantization**

   * BPE-style merging is fundamentally a **vector quantization** of sequences into reusable units; it trades off dynamic range of representation against sequence length.
   * MS-BPE explicitly acknowledges this and constrains the quantizer to respect model capacity.

5. **Approximate mapping from cost to self-information**

   * Per-character cost from rarity and context imbalance is an approximation to local information density, not a perfect theoretical quantity.
   * It is still grounded in observable statistics and biased toward characters that cause “epistemic regime shifts.”

Under these assumptions, a surprise-constrained tokenizer is not arbitrary but **necessary**: without such a constraint, unconstrained BPE can (and empirically does) create tokens whose behavior cannot be adequately learned by a finite model, wasting capacity and hurting performance.([ACL Anthology][2])

---

## Section 9 – Limitations and Open Questions

1. **Domain and language dependence**

   * Character distributions and context imbalance differ across domains (code vs prose vs math) and languages (Latin vs non-Latin scripts).
   * MS-BPE likely needs **domain-specific tuning** of (\alpha, \beta, \tau_\text{inst}, \tau_\text{type}).

2. **Choosing budget fraction (f)**

   * The fraction of capacity allocated to rare-token idiosyncrasy is a design choice, not dictated by theory.
   * It must be tuned empirically per architecture/task family.

3. **Interaction with scaling laws**

   * While Chinchilla-style scaling laws define a smooth relationship between N, D, and loss,([Epoch AI][5])  the constants in those laws depend on tokenization.
   * Systematic experiments are required to measure how MS-BPE shifts these constants compared to standard BPE/unigram.

4. **Implementation complexity**

   * Computing trigram-based imbalance and enforcing rolling costs adds overhead to vocabulary construction.
   * However, construction is offline; runtime tokenization remains BPE-like and fast.

5. **Theoretical optimality**

   * MS-BPE is **capacity-aligned**, but not necessarily **optimal** under any single closed-form objective.
   * A full theory would relate MS-BPE to rate–distortion bounds where distortion is measured in **model loss** rather than reconstruction error.

---

## Section 10 – Summary

Minimum-Surprise Byte Pair Encoding (MS-BPE) reframes tokenization as **capacity-aware, information-density shaping**:

* It uses character-level statistics (rarity + context asymmetry) to assign surprise costs.
* It imposes **rolling** and **static** surprise limits per token, calibrated to empirical bits/parameter and tokens/parameter laws.([arXiv][4])
* It thereby prevents individual tokens from concentrating more self-information than a realistic transformer can metabolize, while still allowing efficient compression along smooth, frequent character paths.

Conceptually, MS-BPE is:

* A generalization of BPE where **“do not surprise the model more than its capacity per step”** is a first-class constraint.
* A bridge between classical information theory (entropy, capacity) and the practical realities of finite LLMs and their scaling laws.

What emerges is not just “another tokenizer,” but a **principled recipe** for aligning tokenization with model capacity: a minimum-surprise, capacity-compatible subword code.

[1]: https://arxiv.org/abs/1508.07909 "[1508.07909] Neural Machine Translation of Rare Words with Subword Units"
[2]: https://aclanthology.org/2020.findings-emnlp.414.pdf "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
[3]: https://mattmahoney.net/dc/entropy1.html "
Refining the Estimated Entropy of English by Shannon Game Simulation
"
[4]: https://arxiv.org/abs/2505.24832 "[2505.24832] How much do language models memorize?"
[5]: https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt "Chinchilla scaling: A replication attempt | Epoch AI"
[6]: https://arxiv.org/abs/1804.10959?utm_source=chatgpt.com "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates"
