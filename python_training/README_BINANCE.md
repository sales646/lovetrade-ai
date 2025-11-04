# Binance Crypto Data Integration

## Overview
Hämtar 1-minuters historisk crypto-data från Binance för att komplettera aktiedata.

## Crypto-par (17 olika marknader)
- **Majors**: BTC, ETH
- **Layer-1**: SOL, ADA, DOT, AVAX
- **DeFi**: UNI, AAVE, LINK
- **Layer-2**: MATIC, ARB
- **Exchange**: BNB
- **Meme**: DOGE, SHIB
- **Stablecoin pairs**: BTCBUSD, ETHBUSD

## Specifikationer
- **Timeframe**: 1 minut (optimal för RL-träning)
- **Historik**: 3 år
- **Datamängd**: ~2.6 miljoner candles per par
- **Total storlek**: ~45 miljoner bars (~5-8 GB)
- **Källa**: Binance Data Vision (officiellt arkiv)

## Användning

### Steg 1: Hämta Binance-data
```bash
cd python_training
python fetch_binance_data.py
```

Detta kommer:
- Ladda ner månadsarkiv från Binance
- Beräkna tekniska indikatorer
- Lägga till data i `historical_bars` och `technical_indicators`
- **INTE radera** befintlig aktiedata

### Steg 2: Uppdatera cache
```bash
python preload_data.py
```

Detta skapar en ny cache-fil som innehåller både aktier OCH crypto.

### Steg 3: Träna med mixad data
```bash
python quick_train.py
```

Träningen kommer nu använda både:
- Stock data (5m timeframe)
- Crypto data (1m timeframe)

## Fördelar med 1-minuters data
✅ Mer granulär - fångar snabba marknadsrörelser
✅ Fler träningsexempel - ~400k candles per år vs 100k för 5m
✅ Bättre för daytrading-strategier
✅ Högre volatilitet = mer RL-signaler

## Datakvalitet
- **Källa**: Binance officiellt arkiv (https://data.binance.vision)
- **Verifierad**: Matchar Binance trading-data
- **Komplett**: Inga luckor i datan
- **Uppdaterad**: Dagliga arkiv tillgängliga

## Tekniska indikatorer
Samma som för stocks:
- RSI-14
- ATR-14
- EMA-20, EMA-50
- VWAP
- Volume Z-score

## Tips
💡 **Mix markets**: Träna på både stocks (5m) och crypto (1m) för robustare policies
💡 **Start small**: Testa först med 5-10 crypto-par innan full dataset
💡 **Monitor GPU**: 1m data = mer compute, se till att GPU används effektivt
