# 🚀 Backtesting Framework

Ein umfassendes, professionelles Backtesting-System für technische Indikatoren auf Cryptocurrency-Märkten mit **23 Indikatoren**.

## 📋 Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Features](#features)
- [Struktur](#struktur)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Verfügbare Indikatoren](#verfügbare-indikatoren)
- [Ergebnisse & Reports](#ergebnisse--reports)
- [Architektur](#architektur)
- [Beispiele](#beispiele)

---

## 🎯 Übersicht

Das **Integrated Backtesting Framework** ist ein modulares, erweiterbares System zum systematischen Testen von technischen Indikatoren auf historischen Kryptowährungsdaten. Es unterstützt **23 verschiedene Indikatoren** und bietet umfassende Performance-Analysen.

### Hauptmerkmale

- ✅ **23 technische Indikatoren** (18 Single-Parameter + 5 Matrix-Indikatoren)
- ✅ **Multi-Kategorie Support**: Majors (BTC, ETH), Alts (UNI, AAVE), Memes (DOGE, SHIB)
- ✅ **Long-Only Strategien** mit klaren Entry/Exit-Regeln
- ✅ **Umfassende Performance-Metriken**: Sharpe Ratio, Sortino Ratio, Omega Ratio, Calmar Ratio
- ✅ **Matrix-Analysen** für Multi-Parameter-Optimierung
- ✅ **Visualisierungen**: Heatmaps, Charts, Performance-Vergleiche
- ✅ **Universal Backtesting System**: Teste alle Indikatoren auf einmal
- ✅ **Detaillierte CSV-Exports** für weitere Analysen

---

## ⚡ Features

### Performance-Metriken

Das Framework berechnet automatisch:

- **Sharpe Ratio**: Risiko-adjustierte Rendite
- **Sortino Ratio**: Downside-fokussierte Risikoadjustierung
- **Omega Ratio**: Gewinn-zu-Verlust-Verhältnis
- **Calmar Ratio**: Return vs. Maximum Drawdown
- **Win Rate**: Prozentsatz gewinnender Trades
- **Total Return**: Gesamtrendite der Strategie
- **Maximum Drawdown**: Größter Wertverlust vom Höchststand

### Visualisierungen

- 📊 **Heatmaps** für Matrix-Parameter-Optimierung
- 📈 **Performance-Charts** mit Equity Curves
- 🎯 **Scatter-Plots** für Return vs. Drawdown
- 🏆 **Ranking-Charts** für Indikator-Vergleiche

### Flexible Konfiguration

- 🎛️ **Quick Mode**: Schnelle Tests mit reduzierten Ranges
- 🔥 **Full Mode**: Komplette Parameter-Ranges (5-150)
- 📂 **Kategorie-basiert**: Separate Tests für Majors/Alts/Memes
- ⚙️ **Anpassbare Schwellenwerte** für jeden Indikator

---

## 📁 Struktur

```
indicator_backtesting_github/
│
├── README.md                          # Diese Datei
├── __init__.py                        # Framework-Initialisierung & Info
├── _backtesting_base_.py              # Basis-Klasse mit gemeinsamen Funktionen
├── run_all_backtests.py               # Universal Backtesting System (alle 20 Indikatoren)
│
├── backtesting_majors.csv             # Asset-Liste: BTC, ETH, BNB, SOL, etc.
├── backtesting_alts.csv               # Asset-Liste: UNI, AAVE, XMR, etc.
├── backtesting_memes.csv              # Asset-Liste: DOGE, SHIB, PEPE, BONK
│
├── price_data/                        # Preisdaten für alle Assets
│   ├── majors/                        # BTC, ETH, BNB, SOL, etc. (btc_1d.csv, eth_1d.csv, ...)
│   ├── alts/                          # UNI, AAVE, XMR, etc. (uni_1d.csv, aave_1d.csv, ...)
│   ├── memes/                         # DOGE, SHIB, PEPE, BONK (doge_1d.csv, shib_1d.csv, ...)
│   ├── indices/                       # Index-Daten
│   └── stables/                       # Stablecoin-Daten
│
├── backtest/                          # Indikator-spezifische Backtesting-Systeme
│   ├── backtest_ema.py                # Exponential Moving Average
│   ├── backtest_rsi.py                # Relative Strength Index
│   ├── backtest_macd.py               # Moving Average Convergence Divergence
│   ├── backtest_di.py                 # Directional Indicators (+DI/-DI)
│   ├── backtest_adx.py                # Average Directional Index
│   ├── backtest_cci.py                # Commodity Channel Index
│   ├── backtest_aroon.py              # Aroon Oscillator
│   ├── backtest_cmo.py                # Chande Momentum Oscillator
│   ├── backtest_mfi.py                # Money Flow Index
│   ├── backtest_willr.py              # Williams %R
│   ├── backtest_mom.py                # Momentum
│   ├── backtest_trix.py               # Triple Exponential Moving Average
│   ├── backtest_apo.py                # Absolute Price Oscillator
│   ├── backtest_ppo.py                # Percentage Price Oscillator
│   ├── backtest_adosc.py              # Chaikin A/D Oscillator
│   ├── backtest_vidya.py              # Variable Index Dynamic Average
│   ├── backtest_trendcont.py          # Trend Continuation (Dual HMA)
│   ├── backtest_hullsuite.py          # Hull Suite
│   ├── backtest_fsvzo.py              # Fourier-Smoothed VZO
│   ├── backtest_bbpct.py              # Bollinger Bands Percentile
│   ├── backtest_frama.py              # Fractal Adaptive Moving Average
│   ├── backtest_supertrend.py         # Supertrend
│   └── backtest_mpt.py                # Multi Pivot Trend
│
├── details/                           # Indikator-spezifische Detailergebnisse
│   ├── ema_backtesting_results/       # EMA: Equity Curves, Reports, CSVs
│   ├── rsi_backtesting_results/       # RSI: Equity Curves, Reports, CSVs
│   ├── macd_backtesting_results/      # MACD: Matrix-Heatmaps, Top-Kombinationen
│   └── ...                            # (für jeden Indikator)
│
└── universal_backtesting_results/     # Gesamt-Analyse aller Indikatoren
    ├── universal_backtesting_summary.csv              # Zusammenfassung aller Indikatoren
    ├── universal_backtesting_report.txt               # Detaillierter Text-Report
    ├── best_calibrations_summary.csv                  # Beste Parameter pro Indikator
    ├── top_10_calibrations_detailed_report.txt        # Top 10 Kalibrierungen
    │
    ├── indicators_sharpe_comparison.png               # Sharpe Ratio Ranking
    ├── return_vs_drawdown_scatter.png                 # Return vs. Drawdown Scatter
    ├── performance_heatmap.png                        # Performance-Matrix
    │
    ├── ema_detailed_results.csv                       # Alle EMA-Ergebnisse
    ├── ema_top_10_calibrations.csv                    # Top 10 EMA-Parameter
    ├── rsi_detailed_results.csv                       # Alle RSI-Ergebnisse
    ├── rsi_top_10_calibrations.csv                    # Top 10 RSI-Parameter
    └── ...                                            # (für jeden Indikator)
```

---

## 🔧 Installation

### Voraussetzungen

```bash
Python 3.8+
pandas
numpy
matplotlib
seaborn
talib  # TA-Lib für technische Indikatoren
```

### Installation von TA-Lib

**Windows:**
```bash
# Download TA-Lib Wheel von https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

**Linux/Mac:**
```bash
# Installiere TA-Lib C-Library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Installiere Python-Wrapper
pip install TA-Lib
```

### Framework-Installation

```bash
# Navigiere zum Projekt-Ordner
cd indicator_backtesting_github

# Installiere Python-Dependencies
pip install -r requirements.txt  # (falls vorhanden)
```

---

## 🚀 Verwendung

### 1. Universal Backtesting System (Alle Indikatoren)

```bash
python run_all_backtests.py
```

**Interaktive Auswahl:**
1. Wähle Asset-Kategorie: Majors / Alts / Memes
2. Wähle Modus: Quick Mode (schnell) oder Full Mode (komplett)

**Ergebnis:**
- Testet alle 23 Indikatoren nacheinander
- Erstellt umfassende Vergleichsberichte
- Generiert Visualisierungen
- Speichert alle Ergebnisse in `universal_backtesting_results/`

### 2. Einzelner Indikator

```python
from backtest.backtest_rsi import RSIBacktestingSystem

# Initialisiere System
rsi_system = RSIBacktestingSystem(
    max_assets=20,
    assets_csv="backtesting_majors.csv",
    category="majors"
)

# Führe Backtests durch (RSI-Länge: 5-150)
results_df = rsi_system.run_rsi_backtests(range(5, 151))

# Generiere Analyse
rsi_system.generate_comprehensive_analysis(
    results_df=results_df,
    length_column='rsi_length'
)
```

### 3. Matrix-Indikator (z.B. MACD)

```python
from backtest.backtest_macd import MACDBacktestingSystem

# Initialisiere System
macd_system = MACDBacktestingSystem(
    max_assets=20,
    assets_csv="backtesting_majors.csv",
    category="majors"
)

# Führe Matrix-Backtests durch
results_df = macd_system.run_macd_backtests(
    fast_period_range=range(5, 26),
    slow_period_range=range(20, 51)
)

# Ergebnisse werden automatisch gespeichert
```

### 4. Framework-Info anzeigen

```python
import __init__

__init__.print_framework_info()
```

---

## 📊 Verfügbare Indikatoren

### 📈 Single-Parameter Indikatoren (18)

| Indikator | Name | Strategie | Typ |
|-----------|------|-----------|-----|
| **EMA** | Exponential Moving Average | Price > EMA = Long | Trend |
| **RSI** | Relative Strength Index | RSI > 50 = Long | Momentum |
| **CCI** | Commodity Channel Index | CCI > 0 = Long | Momentum |
| **ADX** | Average Directional Index | ADX > 25 = Long | Trend Strength |
| **AROON** | Aroon Oscillator | Aroon > 0 = Long | Momentum |
| **CMO** | Chande Momentum Oscillator | CMO > 0 = Long | Momentum |
| **MFI** | Money Flow Index | MFI > 50 = Long | Volume/Momentum |
| **WILLR** | Williams %R | WILLR > -50 = Long | Momentum |
| **MOM** | Momentum | MOM > 0 = Long | Momentum |
| **TRIX** | Triple EMA | TRIX > 0 = Long | Momentum/Trend |
| **VIDYA** | Variable Index Dynamic Average | VIDYA steigend = Long | Adaptive Trend |
| **HULLSUITE** | Hull Suite | MHULL > SHULL = Long | Trend |
| **FSVZO** | Fourier-Smoothed VZO | VZO > Signal = Long | Volume/Momentum |
| **BBPCT** | Bollinger Bands Percentile | Position > 50% = Long | Volatility |
| **FRAMA** | Fractal Adaptive Moving Average | Signal = 1 = Long | Adaptive Trend |
| **MPT** | Multi Pivot Trend | Signal > 0.3 = Long | Trend |

**Parameter-Range**: 5-150 (Einser-Schritte)

### 🔢 Matrix-Indikatoren (5)

| Indikator | Name | Strategie | Parameter |
|-----------|------|-----------|-----------|
| **DI** | Directional Indicators | +DI > -DI = Long | +DI (1-150) × -DI (1-150) |
| **MACD** | MA Convergence Divergence | MACD > Signal = Long | Fast (2-150) × Slow (5-159) |
| **APO** | Absolute Price Oscillator | APO > 0 = Long | Fast (2-150) × Slow (5-159) |
| **PPO** | Percentage Price Oscillator | PPO > 0 = Long | Fast (5-150) × Slow (6-155) |
| **ADOSC** | Chaikin A/D Oscillator | ADOSC > 0 = Long | Fast (2-150) × Slow (5-155) |
| **TRENDCONT** | Trend Continuation | Uptrend = Long | Fast HMA (5-150) × Slow HMA (6-155) |
| **SUPERTREND** | Supertrend | Signal = 1 = Long | ATR Period × Factor |

**Parameter-Ranges**: Validierung Fast < Slow, alle gültigen Kombinationen getestet

---

## 📈 Ergebnisse & Reports

### Universal Backtesting Results

Nach einem kompletten Backtest mit allen **23 Indikatoren** findest du in `universal_backtesting_results/`:

#### 1. **Zusammenfassungen (CSV)**

- `universal_backtesting_summary.csv`: Beste Sharpe Ratio, Return, Drawdown pro Indikator
- `best_calibrations_summary.csv`: Beste Parameter-Kalibrierungen für jeden Indikator

#### 2. **Detaillierte Ergebnisse (CSV)**

Für jeden Indikator:
- `{indikator}_detailed_results.csv`: Alle getesteten Kombinationen mit allen Metriken
- `{indikator}_top_10_calibrations.csv`: Top 10 Parameter-Kalibrierungen (nach Combined Score)

#### 3. **Text-Reports**

- `universal_backtesting_report.txt`: 
  - Ranking aller Indikatoren (nach Sharpe Ratio)
  - Strategische Empfehlungen (TOP 3, Sicherster, Höchste Returns)
  - Detaillierte Indikator-Analyse

- `top_10_calibrations_detailed_report.txt`:
  - Top 10 Kalibrierungen pro Indikator
  - Durchschnitts-Performance pro Parameter
  - Combined Score für optimale Balance

#### 4. **Visualisierungen (PNG)**

- `indicators_sharpe_comparison.png`: Horizontales Bar-Chart mit Sharpe Ratios
- `return_vs_drawdown_scatter.png`: Scatter-Plot (Return vs. Drawdown, Blasengröße = Sharpe)
- `performance_heatmap.png`: Matrix aller Performance-Metriken

### Indikator-spezifische Details

In `details/{indikator}_backtesting_results_{kategorie}/`:

- **Equity Curves**: Visuelle Darstellung der besten Strategien
- **Parameter-Analysen**: Charts für Parameter vs. Performance
- **Top 10 Konfigurationen**: Tabellen und Visualisierungen
- **Matrix-Heatmaps** (nur für Matrix-Indikatoren): 2D-Visualisierung aller Kombinationen

---

## 🏗️ Architektur

### Basis-Klasse: `BaseBacktestingSystem`

Alle Indikatoren erben von dieser Klasse:

```python
class BaseBacktestingSystem:
    """
    Basis-Klasse für alle Backtesting-Systeme
    
    Features:
    - Asset-Daten laden (mit Caching)
    - Backtest durchführen (vectorized)
    - Performance-Metriken berechnen
    - Visualisierungen generieren
    - Reports erstellen
    """
    
    def __init__(self, max_assets, strategy_name, category="majors"):
        # Initialisierung...
        
    def load_asset_data(self, asset_name):
        """Lädt Preis-Daten für ein Asset"""
        
    def calculate_backtest_metrics(self, signals_df):
        """Berechnet Performance-Metriken"""
        
    def run_single_backtest_generic(self, indicator_name, test_range, ...):
        """Generischer Single-Parameter Backtest"""
        
    def generate_comprehensive_analysis(self, results_df, ...):
        """Erstellt umfassende Analyse mit Charts und Reports"""
```

### Indikator-Systeme

Jeder Indikator erweitert die Basis-Klasse:

```python
class RSIBacktestingSystem(BaseBacktestingSystem):
    """RSI-spezifisches Backtesting-System"""
    
    def calculate_rsi_signals(self, data, rsi_length):
        """Berechnet RSI und generiert Signale"""
        
    def run_rsi_backtests(self, rsi_range):
        """Führt Backtests für RSI-Range durch"""
```

### Universal System

Das Universal System orchestriert alle Indikatoren:

```python
class UniversalBacktestingSystem:
    """
    Testet alle 20 Indikatoren und erstellt Vergleiche
    """
    
    def run_all_backtests(self):
        """Startet alle Indikator-Tests nacheinander"""
        
    def generate_universal_report(self):
        """Erstellt umfassenden Vergleichsbericht"""
```

---

## 📝 Beispiele

### Beispiel 1: Quick Test für Meme-Coins

```python
from run_all_backtests import UniversalBacktestingSystem

# Erstelle Universal System für Memes im Quick Mode
system = UniversalBacktestingSystem(
    max_assets=20,
    quick_mode=True,
    category="memes"
)

# Starte alle Backtests
system.run_all_backtests()

# Ergebnisse in: universal_backtesting_results_memes/
```

### Beispiel 2: Detaillierter RSI-Test für Majors

```python
from backtest.backtest_rsi import RSIBacktestingSystem

# Initialisiere RSI-System für Majors
rsi = RSIBacktestingSystem(
    max_assets=20,
    assets_csv="backtesting_majors.csv",
    category="majors"
)

# Teste RSI-Längen 10-50
results = rsi.run_rsi_backtests(range(10, 51))

# Finde beste RSI-Länge
best_sharpe_idx = results['sharpe_ratio'].idxmax()
best_config = results.loc[best_sharpe_idx]

print(f"Beste RSI-Länge: {int(best_config['rsi_length'])}")
print(f"Sharpe Ratio: {best_config['sharpe_ratio']:.3f}")
print(f"Total Return: {best_config['total_return']:.1%}")
print(f"Max Drawdown: {best_config['max_drawdown']:.1%}")
```

### Beispiel 3: MACD Matrix-Optimierung

```python
from backtest.backtest_macd import MACDBacktestingSystem

# Initialisiere MACD-System
macd = MACDBacktestingSystem(max_assets=20, category="majors")

# Teste MACD-Matrix
results = macd.run_macd_backtests(
    fast_period_range=range(8, 21),   # Fast: 8-20
    slow_period_range=range(21, 41)   # Slow: 21-40
)

# Finde beste Kombination
best = results.nlargest(1, 'sharpe_ratio').iloc[0]

print(f"Beste MACD-Kombination:")
print(f"  Fast Period: {int(best['fast_period'])}")
print(f"  Slow Period: {int(best['slow_period'])}")
print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
```

### Beispiel 4: Custom Asset-Liste

```python
import pandas as pd
from backtest.backtest_ema import EMABacktestingSystem

# Erstelle Custom Asset-Liste
custom_assets = pd.DataFrame({
    'asset': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    'category': ['majors', 'majors', 'majors']
})
custom_assets.to_csv('my_custom_assets.csv', index=False)

# Teste mit Custom Assets
ema = EMABacktestingSystem(
    max_assets=3,
    assets_csv='my_custom_assets.csv',
    category='majors'
)

results = ema.run_ema_backtests(range(20, 201))
```

---

## 🎯 Best Practices

### 1. **Parameter-Ranges**

- **Quick Mode**: Nutze für initiale Exploration (Range 5-50)
- **Full Mode**: Für finale Optimierung (Range 5-150)
- **Matrix-Indikatoren**: Beginne mit kleineren Ranges zur Zeitersparnis

### 2. **Asset-Kategorien**

- **Majors**: Stabilere Ergebnisse, gut für Live-Trading
- **Alts**: Höhere Volatilität, potenziell höhere Returns
- **Memes**: Extreme Volatilität, nur für risikobereite Strategien

### 3. **Performance-Bewertung**

- **Sharpe Ratio > 1.0**: Gut
- **Sharpe Ratio > 2.0**: Sehr gut
- **Sharpe Ratio > 3.0**: Ausgezeichnet
- **Max Drawdown < 20%**: Akzeptabel
- **Win Rate > 50%**: Positiv

### 4. **Overfitting vermeiden**

- ✅ Verwende Out-of-Sample Tests
- ✅ Teste auf verschiedenen Zeiträumen
- ✅ Bevorzuge robuste Parameter (konsistent über Assets)
- ✅ Kombiniere multiple Indikatoren für höhere Signifikanz

---

## 🔍 Interpretation der Ergebnisse

### Combined Score

Der Combined Score kombiniert multiple Metriken:

```
Combined Score = (Sharpe × 0.3) + (Sortino × 0.3) + (Return × 0.2) + ((1 - Drawdown) × 0.2)
```

**Interpretation:**
- Balanciert Risk/Reward
- Bevorzugt konsistente Performance über hohe Returns mit hohem Risiko
- Ideal für Parameter-Vergleiche

### Top 10 Kalibrierungen

Die Top 10 Kalibrierungen zeigen:
- **Durchschnitts-Performance** über alle Assets (nicht nur Best-Case)
- **Standardabweichung** für Konsistenz-Bewertung
- **Anzahl Tests** für statistische Signifikanz

---

## 🛠️ Erweiterung & Customization

### Neuen Indikator hinzufügen

1. Erstelle neue Datei in `backtest/backtest_myindicator.py`
2. Erbe von `BaseBacktestingSystem`
3. Implementiere `calculate_myindicator_signals()`
4. Implementiere `run_myindicator_backtests()`
5. Füge zu `__init__.py` und `run_all_backtests.py` hinzu

```python
from _backtesting_base_ import BaseBacktestingSystem
import talib as ta

class MyIndicatorBacktestingSystem(BaseBacktestingSystem):
    def __init__(self, **kwargs):
        super().__init__(strategy_name="MyIndicator", **kwargs)
    
    def calculate_myindicator_signals(self, data, param1):
        # Berechne Indikator
        indicator_values = ta.MYINDICATOR(data['close'], param1)
        
        # Generiere Signale
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = (indicator_values > threshold).astype(int)
        
        return signals
    
    def run_myindicator_backtests(self, param_range):
        return self.run_single_backtest_generic(
            indicator_name="MyIndicator",
            test_range=param_range,
            length_param_name='myindicator_length',
            calculate_signals_func=self.calculate_myindicator_signals
        )
```

---

## 📞 Support & Kontakt

Bei Fragen oder Problemen:
- 💬 GitHub Account
- 📖 Dokumentation: Dieses README


## 🚀 Quick Start

```bash
# 1. Navigiere zum Ordner
cd indicator_backtesting_github

# 2. Starte Universal Backtesting (23 Indikatoren)
python run_all_backtests.py

# 3. Wähle Kategorie (Majors/Alts/Memes)
# 4. Wähle Modus (Quick/Full)
# 5. Warte auf Ergebnisse...
# 6. Analysiere Ergebnisse in universal_backtesting_results/
```

**Happy Backtesting! 📊🚀**
