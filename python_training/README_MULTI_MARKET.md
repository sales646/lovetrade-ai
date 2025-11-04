# 🌍 Multi-Market Training Guide

## Cross-Market Transfer Learning för RL Trading

Detta system tränar **en enda policy** på både aktier (NASDAQ) och krypto (Binance) samtidigt för att skapa en robust agent som hanterar olika volatilitetsregimer.

---

## 📊 Varför Multi-Market Training?

### Fördelar

✅ **Volatilitetsrobusthet**: Lär sig hantera både låg (aktier) och hög (krypto) volatilitet  
✅ **Bättre generalisering**: Lär sig underliggande marknadsmekanismer, inte bara specifika instrument  
✅ **24/7 anpassning**: Förstår skillnaden mellan sessions-baserad (aktier) och kontinuerlig (krypto) handel  
✅ **Transfer learning**: Samma princip som multi-språk träning för LLMs  
✅ **Robustare policies**: Överreagerar inte på spikes, vet när man **inte** ska handla  

### Teknisk Grund

PPO och BC är **inte bundna till specifika marknader** — de lär sig:
- Momentum och mean-reversion patterns
- Volymkluster och likviditetsskiften
- Risk-reward balans över olika regimer

---

## 🔧 Teknisk Implementation

### 1️⃣ Market Type Encoding

Varje observation innehåller en **market_type** feature:

```python
state[50] = 0  # Stock (NASDAQ)
state[51] = 1  # Crypto (Binance)
```

Detta blir ett "context token" som nätverket använder för att skilja mellan marknader.

---

### 2️⃣ Normalisering (KRITISKT!)

Eftersom krypto rör sig 5–10× mer än aktier **måste** all data normaliseras:

#### Log-Returns
```python
log_return = ln(P_t / P_{t-1})
```

#### Z-Score Normalisering per Symbol
```python
z_t = (log_return - μ_symbol) / σ_symbol
```

Detta ger **jämförbara state-distributioner** mellan marknader.

#### Implementation
- `_compute_symbol_stats()`: Beräknar mean, std, ATR per symbol vid startup
- `_get_observation()`: Applicerar z-score normalisering på log-returns
- `step()`: Använder Sharpe-normaliserad reward

---

### 3️⃣ Sharpe-Normaliserad Reward

För att undvika att krypto dominerar träningen (pga större rörelser):

```python
reward_t = (log_return / σ_symbol) * position_size * 100
```

Detta gör att agenten bedömer **riskjusterad vinst**, inte bara "störst vinst = bäst".

---

### 4️⃣ Data Blending

Rekommenderade förhållanden:

| Ratio | Användningsfall |
|-------|----------------|
| **70/30 Crypto:Stock** | PPO-träning (mer variation → snabbare inlärning) |
| **50/50** | BC pretrain (balanserad förståelse) |
| **30/70 Crypto:Stock** | Finetune för börshandel (mer stabilitet) |

---

## 🚀 Användning

### Steg 1: Hämta Data

```bash
# Hämta krypto-data (1-min, 3 år, 17 marknader)
python fetch_binance_data.py

# Uppdatera cache med både stocks + crypto
python preload_data.py
```

### Steg 2: Träna Multi-Market Policy

```bash
# Quick training med mixad data
python quick_train.py
```

Environment kommer automatiskt att:
1. Ladda både stock och crypto bars
2. Klassificera symbols (aktier vs krypto)
3. Beräkna symbol-specifik statistik
4. Applicera normalisering och Sharpe-reward

---

## 📈 Träningsschema

| Fas | Data | Syfte | Epochs |
|-----|------|-------|--------|
| **BC Pretrain** | 50/50 stock + crypto | Lär basmönster | 5k–10k |
| **PPO Train** | 70/30 crypto:stock | Lär risk/avkastning | 10M–20M steps |
| **Finetune** | 100% aktier | Anpassa för börsregler | 2k–5k |

---

## 🔍 Features i State Vector

```python
# Standard features (0-49)
- OHLCV, technical indicators, position state, momentum

# Multi-market features (50-51)
state[50] = market_type        # 0 = stock, 1 = crypto
state[51] = normalized_volatility  # ATR / avg_price
```

Total state dim: **52 features**

---

## 💡 Advanced Tricks

### Domain Randomization
Applicera små slumpade distorsioner:
- Fees: ±0.1%
- Slippage: ±0.05%
- Delays: ±1 bar

→ Starkare generalisering

### Trading Hours Mask
Lägg till en signal (0–1) för när marknaden är öppen:
```python
state[52] = is_market_open  # 0 under börs-stängning, 1 annars
```

→ Modellen lär sig sluta handla utanför öppettider

### Multi-Agent Shared Policy (Advanced)
För specialisering med gemensam encoder:
- Shared feature extraction layers
- Separate action heads per marknad

---

## 📊 Exempel: Symbol Stats

Efter att ha kört `_compute_symbol_stats()`:

```
AAPL:  mean=0.000012, std=0.0234, atr=1.23
BTCUSDT: mean=0.000045, std=0.0789, atr=234.56
```

→ Krypto har 3–4× högre std → reward normaliseras för rättvis jämförelse

---

## 🎯 Resultat

En multi-market tränad agent:
- Handlar konservativt på aktier (lägre volatilitet)
- Reagerar snabbt på krypto (högre volatilitet)
- Vet när man **inte** ska handla (viktigt!)
- Generalizar till nya instrument utan omträning

---

## 📚 Referenser

- **PPO**: Proximal Policy Optimization (OpenAI)
- **BC**: Behavior Cloning (Imitation Learning)
- **Cross-Domain Transfer**: Samma princip som BERT/GPT multi-språk träning

---

**Tips**: Börja med 70/30 crypto:stock för PPO, sedan fintune 100% aktier om du primärt handlar NASDAQ.
