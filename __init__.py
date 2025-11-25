"""
Integrated Backtesting Framework
=================================

Ein umfassendes Backtesting-System für technische Indikatoren auf Cryptocurrency-Märkten.

Hauptkomponenten:
- BaseBacktestingSystem: Basis-Klasse mit gemeinsamen Funktionen
- Indikator-spezifische Backtesting-Systeme für alle TA-Lib Indikatoren

Features:
- Automatisierte Performance-Analyse (Sharpe, Sortino, Omega, Calmar Ratio)
- Matrix-Analysen für Multi-Parameter Indikatoren
- Heatmap-Visualisierungen
- Umfassende Statistik-Reports
- CSV-Export für weitere Analysen

Verwendung:
    from integrated_backtesting import BaseBacktestingSystem
    from integrated_backtesting.backtest import RSIBacktestingSystem
"""

# Version Info
__version__ = "2.0.0"
__author__ = "AMITPI Backtesting Framework"
__all__ = [
    "BaseBacktestingSystem",
    # Single-Parameter Indikatoren (18)
    "ADXBacktestingSystem",
    "AROONBacktestingSystem",
    "BBPctBacktestingSystem",
    "CCIBacktestingSystem",
    "CMOBacktestingSystem",
    "EMABacktestingSystem",
    "FRAMABacktestingSystem",
    "FSVZOBacktestingSystem",
    "HullSuiteBacktestingSystem",
    "MFIBacktestingSystem",
    "MOMBacktestingSystem",
    "MultiPivotTrendBacktestingSystem",
    "RSIBacktestingSystem",
    "TRIXBacktestingSystem",
    "VIDYABacktestingSystem",
    "WILLRBacktestingSystem",
    # Matrix-Indikatoren (5)
    "ADOSCBacktestingSystem",
    "APOBacktestingSystem",
    "DIBacktestingSystem",
    "MACDBacktestingSystem",
    "PPOBacktestingSystem",
    "SupertrendBacktestingSystem",
    "TrendContinuationBacktestingSystem",
]

# Import der Basis-Klasse
from ._backtesting_base_ import BaseBacktestingSystem

# Import aller Indikator-spezifischen Systeme
try:
    from .backtest.backtest_adosc import ADOSCBacktestingSystem
except ImportError:
    ADOSCBacktestingSystem = None

try:
    from .backtest.backtest_adx import ADXBacktestingSystem
except ImportError:
    ADXBacktestingSystem = None

try:
    from .backtest.backtest_apo import APOBacktestingSystem
except ImportError:
    APOBacktestingSystem = None

try:
    from .backtest.backtest_aroon import AROONBacktestingSystem
except ImportError:
    AROONBacktestingSystem = None

try:
    from .backtest.backtest_bbpct import BBPctBacktestingSystem
except ImportError:
    BBPctBacktestingSystem = None

try:
    from .backtest.backtest_cci import CCIBacktestingSystem
except ImportError:
    CCIBacktestingSystem = None

try:
    from .backtest.backtest_cmo import CMOBacktestingSystem
except ImportError:
    CMOBacktestingSystem = None

try:
    from .backtest.backtest_di import DIBacktestingSystem
except ImportError:
    DIBacktestingSystem = None

try:
    from .backtest.backtest_ema import EMABacktestingSystem
except ImportError:
    EMABacktestingSystem = None

try:
    from .backtest.backtest_fsvzo import FSVZOBacktestingSystem
except ImportError:
    FSVZOBacktestingSystem = None

try:
    from .backtest.backtest_hullsuite import HullSuiteBacktestingSystem
except ImportError:
    HullSuiteBacktestingSystem = None

try:
    from .backtest.backtest_macd import MACDBacktestingSystem
except ImportError:
    MACDBacktestingSystem = None

try:
    from .backtest.backtest_mfi import MFIBacktestingSystem
except ImportError:
    MFIBacktestingSystem = None

try:
    from .backtest.backtest_mom import MOMBacktestingSystem
except ImportError:
    MOMBacktestingSystem = None

try:
    from .backtest.backtest_ppo import PPOBacktestingSystem
except ImportError:
    PPOBacktestingSystem = None

try:
    from .backtest.backtest_rsi import RSIBacktestingSystem
except ImportError:
    RSIBacktestingSystem = None

try:
    from .backtest.backtest_trix import TRIXBacktestingSystem
except ImportError:
    TRIXBacktestingSystem = None

try:
    from .backtest.backtest_vidya import VIDYABacktestingSystem
except ImportError:
    VIDYABacktestingSystem = None

try:
    from .backtest.backtest_trendcont import TrendContinuationBacktestingSystem
except ImportError:
    TrendContinuationBacktestingSystem = None

try:
    from .backtest.backtest_willr import WILLRBacktestingSystem
except ImportError:
    WILLRBacktestingSystem = None

try:
    from .backtest.backtest_frama import FRAMABacktestingSystem
except ImportError:
    FRAMABacktestingSystem = None

try:
    from .backtest.backtest_supertrend import SupertrendBacktestingSystem
except ImportError:
    SupertrendBacktestingSystem = None

try:
    from .backtest.backtest_mpt import MultiPivotTrendBacktestingSystem
except ImportError:
    MultiPivotTrendBacktestingSystem = None


def list_available_indicators():
    """
    Listet alle verfügbaren Indikator-Backtesting-Systeme auf
    
    Returns:
        List[str]: Liste der verfügbaren Indikatoren
    """
    indicators = []
    for name in __all__:
        if name != "BaseBacktestingSystem" and globals().get(name) is not None:
            indicators.append(name.replace("BacktestingSystem", ""))
    return sorted(indicators)


def get_indicator_info(indicator_name: str) -> dict:
    """
    Gibt Informationen über einen spezifischen Indikator zurück
    
    Args:
        indicator_name: Name des Indikators (z.B. "RSI", "MACD", "ADX")
        
    Returns:
        dict: Dictionary mit Indikator-Informationen
    """
    info_map = {
        "ADOSC": {
            "name": "Chaikin A/D Oscillator",
            "type": "Volume/Momentum",
            "parameters": ["fast_period", "slow_period"],
            "threshold": 0.0,
            "strategy": "ADOSC > 0 = Long",
            "description": "Volume-Preis Momentum Indikator"
        },
        "ADX": {
            "name": "Average Directional Index",
            "type": "Trend Strength",
            "parameters": ["period"],
            "threshold": 25.0,
            "strategy": "ADX > 25 = Long",
            "description": "Misst Trendstärke (0-100)"
        },
        "APO": {
            "name": "Absolute Price Oscillator",
            "type": "Momentum",
            "parameters": ["fast_period", "slow_period"],
            "threshold": 0.0,
            "strategy": "APO > 0 = Long",
            "description": "Absolute Differenz zwischen zwei MAs"
        },
        "AROON": {
            "name": "Aroon Oscillator",
            "type": "Momentum",
            "parameters": ["period"],
            "threshold": 0.0,
            "strategy": "Aroon > 0 = Long",
            "description": "Aroon Up - Aroon Down (-100 bis +100)"
        },
        "BBPCT": {
            "name": "Bollinger Bands Percentile",
            "type": "Volatility/Position",
            "parameters": ["length", "factor", "lookback"],
            "threshold": 50.0,
            "strategy": "Position > 50% = Long",
            "description": "Position innerhalb der Bollinger Bands (0-100%)"
        },
        "CCI": {
            "name": "Commodity Channel Index",
            "type": "Momentum",
            "parameters": ["period"],
            "threshold": 0.0,
            "strategy": "CCI > 0 = Long",
            "description": "Preisabweichung vom statistischen Mittel"
        },
        "CMO": {
            "name": "Chande Momentum Oscillator",
            "type": "Momentum",
            "parameters": ["period"],
            "threshold": 0.0,
            "strategy": "CMO > 0 = Long",
            "description": "Momentum basierend auf Up/Down Moves (-100 bis +100)"
        },
        "DI": {
            "name": "Directional Indicators",
            "type": "Trend Direction",
            "parameters": ["plus_di_period", "minus_di_period"],
            "threshold": None,
            "strategy": "+DI > -DI = Long",
            "description": "+DI vs -DI Vergleich für Trendrichtung"
        },
        "EMA": {
            "name": "Exponential Moving Average",
            "type": "Trend",
            "parameters": ["period"],
            "threshold": None,
            "strategy": "Price > EMA = Long",
            "description": "Exponentiell gewichteter gleitender Durchschnitt"
        },
        "FSVZO": {
            "name": "Fourier-Smoothed Volume Zone Oscillator",
            "type": "Volume/Momentum",
            "parameters": ["vzo_length", "signal_length", "smoothing_length"],
            "threshold": 0.0,
            "strategy": "VZO > Signal = Long",
            "description": "Volume-basierter Momentum-Indikator mit Fourier-Glättung"
        },
        "HULLSUITE": {
            "name": "Hull Suite (Hull Moving Average)",
            "type": "Trend",
            "parameters": ["length", "length_mult"],
            "threshold": None,
            "strategy": "MHULL > SHULL = Long",
            "description": "Hull MA mit Shift-Vergleich für Trendwechsel"
        },
        "MACD": {
            "name": "Moving Average Convergence Divergence",
            "type": "Trend/Momentum",
            "parameters": ["fast_period", "slow_period", "signal_period"],
            "threshold": 0.0,
            "strategy": "MACD > Signal = Long",
            "description": "MACD Line vs Signal Line Crossover"
        },
        "MFI": {
            "name": "Money Flow Index",
            "type": "Volume/Momentum",
            "parameters": ["period"],
            "threshold": 50.0,
            "strategy": "MFI > 50 = Long",
            "description": "RSI mit Volumen-Gewichtung (0-100)"
        },
        "MOM": {
            "name": "Momentum",
            "type": "Momentum",
            "parameters": ["period"],
            "threshold": 0.0,
            "strategy": "MOM > 0 = Long",
            "description": "Preisdifferenz über n Perioden"
        },
        "PPO": {
            "name": "Percentage Price Oscillator",
            "type": "Momentum",
            "parameters": ["fast_period", "slow_period"],
            "threshold": 0.0,
            "strategy": "PPO > 0 = Long",
            "description": "Prozentuale Differenz zwischen zwei MAs"
        },
        "RSI": {
            "name": "Relative Strength Index",
            "type": "Momentum",
            "parameters": ["period"],
            "threshold": 50.0,
            "strategy": "RSI > 50 = Long",
            "description": "Überkauft/Überverkauft Indikator (0-100)"
        },
        "TRENDCONT": {
            "name": "Trend Continuation (Dual HMA)",
            "type": "Trend",
            "parameters": ["fast_hma", "slow_hma"],
            "threshold": None,
            "strategy": "Uptrend = Long",
            "description": "Dual Hull MA System mit Neutral-Zone"
        },
        "TRIX": {
            "name": "Triple Exponential Moving Average",
            "type": "Momentum/Trend",
            "parameters": ["period"],
            "threshold": 0.0,
            "strategy": "TRIX > 0 = Long",
            "description": "Rate of Change von dreifach geglätteter EMA"
        },
        "VIDYA": {
            "name": "Variable Index Dynamic Average",
            "type": "Adaptive Trend",
            "parameters": ["vidya_length", "hist_length"],
            "threshold": None,
            "strategy": "VIDYA steigend = Long",
            "description": "Adaptiver MA basierend auf Volatilität"
        },
        "WILLR": {
            "name": "Williams %R",
            "type": "Momentum",
            "parameters": ["period"],
            "threshold": -50.0,
            "strategy": "WILLR > -50 = Long",
            "description": "Überkauft/Überverkauft Indikator (-100 bis 0)"
        },
        "FRAMA": {
            "name": "Fractal Adaptive Moving Average",
            "type": "Adaptive Trend",
            "parameters": ["length"],
            "threshold": None,
            "strategy": "Signal = 1 = Long",
            "description": "Adaptiver MA basierend auf Fraktaler Dimension"
        },
        "SUPERTREND": {
            "name": "Supertrend",
            "type": "Trend/Volatility",
            "parameters": ["atr_period", "factor"],
            "threshold": None,
            "strategy": "Signal = 1 = Long",
            "description": "ATR-basierter Trend-Indikator"
        },
        "MPT": {
            "name": "Multi Pivot Trend",
            "type": "Trend",
            "parameters": ["length"],
            "threshold": 0.3,
            "strategy": "Signal > 0.3 = Long",
            "description": "Multi-Pivot basierter Trend-Indikator"
        }
    }
    
    return info_map.get(indicator_name.upper(), {
        "name": indicator_name,
        "type": "Unknown",
        "parameters": [],
        "threshold": None,
        "strategy": "N/A",
        "description": "Keine Informationen verfügbar"
    })


def print_framework_info():
    """
    Gibt eine Übersicht über das Backtesting-Framework aus
    """
    print("=" * 80)
    print("INTEGRATED BACKTESTING FRAMEWORK - V2.0")
    print("=" * 80)
    print(f"Version: {__version__}")
    print(f"Autor: {__author__}")
    print()
    print("VERFÜGBARE INDIKATOREN:")
    print("-" * 80)
    
    available = list_available_indicators()
    
    # Gruppiere nach Typ
    single_param = []
    matrix_param = []
    
    for indicator in available:
        info = get_indicator_info(indicator)
        if len(info['parameters']) == 1:
            single_param.append((indicator, info))
        else:
            matrix_param.append((indicator, info))
    
    print(f"\n📊 SINGLE-PARAMETER INDIKATOREN ({len(single_param)}):")
    print("-" * 80)
    for indicator, info in sorted(single_param):
        print(f"  • {indicator:12s} - {info['name']:40s} [{info['type']}]")
        print(f"    {'':14s} Strategie: {info['strategy']}")
    
    print(f"\n🔢 MATRIX-INDIKATOREN ({len(matrix_param)}):")
    print("-" * 80)
    for indicator, info in sorted(matrix_param):
        print(f"  • {indicator:12s} - {info['name']:40s} [{info['type']}]")
        print(f"    {'':14s} Strategie: {info['strategy']}")
    
    print()
    print(f"GESAMT: {len(available)} Indikatoren verfügbar")
    print("=" * 80)


# Automatisch Framework-Info anzeigen bei direktem Import
if __name__ == "__main__":
    print_framework_info()
