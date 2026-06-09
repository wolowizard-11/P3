# IMC Prosperity 3 Algorithmic Trading Framework

This repository contains the full, production-ready trading framework built for the IMC Prosperity market-making competition. The system is designed as a growing, multi-asset portfolio optimizer that scales as new financial instruments, derivatives, and alternative data sources are introduced across rounds.

For model calibration, diagnostic tracking, and backtest evaluation, all execution telemetry and platform logs can be explored through the custom dashboard @ [wolowizard-11.github.io/Viz](https://wolowizard-11.github.io/Viz/)

---

## Analytics, Telemetry & Log Visualizer

At the core of our iteration pipeline is a custom-built **Prosperity Log Visualizer**. Since raw platform outputs can be hard to interpret at scale, this frontend utility parses, decompresses, and maps execution telemetry into readable charts.

- **Dual-Source Streaming:** Accepts live production state files pasted directly from the online platform, or JSON/CSV data from local backtest runs.
- **Granular Microstructure Charts:** Tracks inventory delta vs. mid-price divergence, bid-ask spread capture performance, and per-asset PnL trajectories tick-by-tick.
- **Arbitrage Tracking:** Plots synthetic ETF/basket fair values against the market price of the composite instrument, highlighting optimal entry bands and execution slippage.

---

## 🛠️ Round-by-Round Strategy Architecture

### Round 1: Stationary Anchoring & Microstructure Tracking

**Products:** `RAINFOREST_RESIN`, `KELP`

#### RAINFOREST_RESIN

- **Thesis:** A strongly stationary, highly liquid asset with tight mean-reversion characteristics anchored around an equilibrium price of `10,000`.
- **Implementation:** Uses a pure electronic market-making framework. Rather than chasing short-term trends, the system continuously layers bids and asks around fair value. The pricing engine applies structural skew parameters, leaning quotes deeper or shallower based on net inventory to aggressively trigger unwinds near position limits.

#### KELP

- **Thesis:** A non-stationary asset driven by structural order-book flows and persistent micro-trends.
- **Implementation:** To avoid the adverse selection common in simpler market-making setups, we built a volume-filtered institutional VWAP engine. The module screens out thin retail orders (sizes < 15) to isolate genuine institutional liquidity and dynamically adjusts our internal fair-value anchor. Orders are placed conditionally on book width to stay on the right side of short-term momentum.

---

### Round 2: Cross-Sectional Index Arbitrage

**Products Added:** `PICNIC_BASKET1`, `PICNIC_BASKET2`

**Components:** `CROISSANTS`, `JAMS`, `DJEMBES`

#### The Strategy

- **Thesis:** Structural pricing gaps frequently appear between composite index tokens (`PICNIC_BASKET1` / `PICNIC_BASKET2`) and their underlying physical components, caused by fragmented liquidity pools and latency mismatches.
- **Implementation:** The arbitrage engine continuously monitors order books across all constituent assets to calculate a live synthetic fair value.
- **Dynamic Sizing:** Instead of fixed-unit arbitrage, the engine applies a variable scaling factor. Entry size grows dynamically based on how far the market premium or discount stretches beyond our threshold (`40` ticks), maximizing size during dislocation events while keeping exposure tight on marginal spreads.

---

### Round 3: Options Pricing & Delta Neutralization

**Products Added:** `VOLCANIC_ROCK`, `COUPON_9500`, `COUPON_9750`, `COUPON_10000`

#### The Strategy

- **Thesis:** Options contracts carry non-linear directional risk tied to the price dynamics of their underlying asset (`VOLCANIC_ROCK`).
- **Implementation:** Introduced a mathematical derivatives engine that aggregates total directional sensitivity across all active option strikes. Using a portfolio sizing scale factor, it calculates the net portfolio Delta.
- **The Loop:** A dynamic delta-hedging routine sweeps the order book of the underlying `VOLCANIC_ROCK`. If the options portfolio generates positive net delta, the engine automatically shorts the underlying to pull net exposure back to zero, isolating pure volatility premium.

---

### Round 4: Volatility Regime Switching

**Products Updated:** `VOLCANIC_ROCK`, `COUPON_9500`, `COUPON_9750`, `COUPON_10000`

#### Strategy Pivot

In Round 4, structural shifts in market volatility revealed that a static Delta framework underperformed during sustained directional breakouts. We overhauled the `VOLCANIC_ROCK` handling and its associated vouchers into a state-dependent **Regime Switching Model**:

```
[ Calculate 50-Tick Rolling Vol (Std Dev) ]
                    |
    +---------------+---------------+
    |                               |
[ Vol > Threshold ]         [ Vol <= Threshold ]
        |                               |
(REGIME 1: HIGH VOL)         (REGIME 2: LOW VOL)
        |                               |
- Calculate EMA Trendline     - Filter retail noise (<15)
- Measure Deviation           - Establish institutional anchor
- Aggressive Momentum         - Tight Passive Market Making
  Snapback
```

#### Regime 1: High Volatility (Mean Reversion Snapback)

- Triggered when rolling standard deviation crosses asset-specific parameters (`VOL_THRES`).
- The system computes an EMA with an alpha decay factor to establish an equilibrium trendline. When prices break past a standard deviation threshold multiplier, the model crosses the spread aggressively to catch rapid snapbacks.

#### Regime 2: Low Volatility (Tight Liquidity Provision)

- Triggered during low-volatility consolidation phases.
- Directional indicators are discarded entirely. The system operates as a high-frequency passive market maker, placing quotes tightly inside the filtered order book to safely capture the bid-ask spread.

---

### Round 5: Counterparty Flow & Toxicity Tracking

**Products Added:** `SQUID_INK`

#### The Strategy

- **Thesis:** The final round introduces well-capitalized, informed counterparty actors (e.g., the "Olivia" entity) whose transaction scale visibly shifts market trends. Standard statistical metrics break down when facing large, concentrated inventory accumulations or distribution waves.
- **Implementation:** Built an aggressive Order Flow Toxicity tracker that monitors the `market_trades` feed in real time.
- **Execution:** The moment the ledger identifies high-conviction buying, the algorithm immediately bypasses passive quoting, sweeps available ask depth up to position limits (`50`), and rides the directional move. On high-conviction selling, it shorts into the bids — shifting fully from mathematical abstraction to flow trading.

---

### Production Rules

- **No External Dependencies:** Standardized to run on native `numpy` and core Python types only.
- **State Serialization:** Persists critical pricing arrays and sliding window statistics across engine iterations using platform-compliant serialization.
- **Log Budget Compliance:** Includes an automated truncation and compression logger to keep deep trace logs safely within the platform's 3,750-character limit.
