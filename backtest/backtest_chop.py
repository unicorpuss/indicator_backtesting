"""
Choppiness Index Backtesting System (Optimiert mit Base Class)

Dieses System testet Choppiness Index-basierte Trading-Strategien über verschiedene Assets und Perioden:
- Choppiness Index Längen von 3 bis 150 (Einser-Schritte)
- 8-20 verschiedene Major Crypto Assets  
- Long/Short Strategie: CHOP < 50 = Long Position (Trending), CHOP >= 50 = Short Position (Choppy)
- Umfassende Performance-Metriken (Sharpe, Sortino, Omega, etc.)
- Verwendet gemeinsame Funktionen aus backtesting_base.py

Choppiness Index (CHOP):
- Misst ob der Markt in einem Trend oder in einer Seitwärtsbewegung ist
- Werte zwischen 0 und 100
- CHOP < 38.2: Starker Trend
- CHOP < 50: Moderater Trend (BULLISH - Long-Bias)
- CHOP >= 50: Choppy/Seitwärts (BEARISH - Short-Bias)
- CHOP > 61.8: Sehr choppy/ranging
- Long-Signal: CHOP < 50 (Trending Market - gute Bedingungen für Trend-Following)
- Short-Signal: CHOP >= 50 (Choppy Market - schlechte Bedingungen, defensiv bleiben)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Backtesting-Funktionen
from _backtesting_base_ import BaseBacktestingSystem

def calculate_true_range(high, low, close):
    """
    Berechnet True Range für jeden Bar
    
    True Range = MAX(
        high - low,
        ABS(high - previous_close),
        ABS(low - previous_close)
    )
    """
    n = len(close)
    tr = np.zeros(n)
    
    # Erste Zeile: high - low
    tr[0] = high[0] - low[0]
    
    # Rest: MAX der drei Möglichkeiten
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    return tr


def calculate_choppiness_index(high, low, close, length: int = 14):
    """
    Berechnet den Choppiness Index
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR(1), length) / (MAX(high, length) - MIN(low, length))) / LOG10(length)
    
    Parameters:
    -----------
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices
    length : int
        Lookback-Periode (3-150)
        
    Returns:
    --------
    tuple: (choppiness_values, signals)
        - choppiness_values: Array mit CHOP-Werten
        - signals: Array mit Signals (1=bullish/trending, -1=bearish/choppy)
    """
    # Convert to numpy arrays
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    
    n = len(close)
    choppiness = np.full(n, np.nan)
    
    # Berechne True Range
    true_range = calculate_true_range(high, low, close)
    
    # Berechne Choppiness Index für jeden Punkt
    for i in range(length - 1, n):
        # SUM(ATR(1), length) = Summe der True Ranges
        sum_tr = np.sum(true_range[i - length + 1:i + 1])
        
        # MAX(high, length) - MIN(low, length)
        max_high = np.max(high[i - length + 1:i + 1])
        min_low = np.min(low[i - length + 1:i + 1])
        range_hl = max_high - min_low
        
        # Vermeide Division durch Null
        if range_hl > 0:
            # CHOP = 100 * LOG10(sum_tr / range_hl) / LOG10(length)
            choppiness[i] = 100 * np.log10(sum_tr / range_hl) / np.log10(length)
        else:
            choppiness[i] = 50.0  # Neutral value
    
    # Generate trading signals
    # BULLISH (1): CHOP < 50 (Trending)
    # BEARISH (-1): CHOP >= 50 (Choppy)
    signals = np.where(choppiness < 50, 1, -1)
    
    return choppiness, signals


class ChoppinessIndexBacktestingSystem(BaseBacktestingSystem):
    """
    Choppiness Index-spezifisches Backtesting-System (Long/Short Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        """
        Initialisiert das Choppiness Index Backtesting System
        
        Args:
            max_assets: Maximale Anzahl der Assets
            **kwargs: Zusätzliche Parameter (assets_csv, category, etc.)
        """
        super().__init__(max_assets, "CHOP", **kwargs)
        self.indicator_name = "CHOP"
        self.strategy_description = "CHOP < 50: Long-Position (Trending) | CHOP >= 50: Short-Position (Choppy)"
        self.threshold = 50.0  # Threshold für Bullish/Bearish
    
    def calculate_chop_signals(self, data: pd.DataFrame, chop_length: int = 14) -> pd.DataFrame:
        """
        Berechnet Choppiness Index-Signale für die Long/Short Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            chop_length: CHOP-Periode (3-150)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < chop_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne Choppiness Index
            choppiness, chop_signals = calculate_choppiness_index(
                high=data['high'].values,
                low=data['low'].values,
                close=data['close'].values,
                length=chop_length
            )
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['choppiness'] = choppiness
            signals_df['chop_signal'] = chop_signals
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Choppiness Index Long/Short Signale
            # CHOP < 50 = Trending (Long Position)
            # CHOP >= 50 = Choppy (Short Position)
            signals_df['position'] = signals_df['chop_signal']
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei CHOP-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_chop_backtests(self, chop_range: range = range(3, 151)) -> pd.DataFrame:
        """
        Führt Choppiness Index-Backtests über verschiedene Perioden durch
        
        Args:
            chop_range: Range der CHOP-Perioden zum Testen (3-150)
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=chop_range,
            length_param_name='chop_length',
            calculate_signals_func=self.calculate_chop_signals,
            indicator_name='CHOP'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden Choppiness Index-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit CHOP-spezifischen Parametern
        strategy_description = "Choppiness Index Strategy (CHOP < 50 = Long/Trending, CHOP >= 50 = Short/Choppy)"
        super().generate_comprehensive_report(results_df, 'chop_length', strategy_description)
        
        # Zusätzliche CHOP-spezifische Analyse
        print(f"\n📋 Erstelle CHOP-spezifische Analyse...")
        
        # Verwende die CHOP-spezifische Analyse-Funktion
        self.generate_chop_specific_analysis(results_df)
    
    def generate_chop_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert Choppiness Index-spezifische Analyse
        Fokussiert auf die besten CHOP-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für CHOP-spezifische Analyse")
            return
        
        try:
            # Erstelle CHOP-spezifische Analyse
            chop_report_lines = []
            chop_report_lines.append("=" * 68)
            chop_report_lines.append("CHOPPINESS INDEX - SPEZIFISCHE ANALYSE")
            chop_report_lines.append("=" * 68)
            chop_report_lines.append(f"Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            chop_report_lines.append(f"Getestete CHOP-Perioden: {results_df['chop_length'].min():.0f} bis {results_df['chop_length'].max():.0f}")
            chop_report_lines.append(f"Anzahl Assets: {results_df['asset'].nunique()}")
            chop_report_lines.append(f"Gesamte Backtests: {len(results_df)}")
            chop_report_lines.append("")
            
            # Beste CHOP-Kalibrierung nach verschiedenen Metriken
            best_chop_calibration = self.find_best_average_calibration(results_df, 'chop_length')
            
            chop_report_lines.append(f"🎯 OPTIMALE CHOP-KALIBRIERUNG")
            chop_report_lines.append("-" * 60)
            
            # Beste Metriken
            metric_keys = [
                ('sharpe_ratio', 'Sharpe Ratio'),
                ('sortino_ratio', 'Sortino Ratio'),
                ('omega_ratio', 'Omega Ratio'),
                ('calmar_ratio', 'Calmar Ratio'),
                ('total_return', 'Total Return')
            ]
            
            for best_key, metric_name in metric_keys:
                best_key_name = f'best_chop_{best_key}'
                avg_key = f'avg_{best_key}'
                if best_key_name in best_chop_calibration and avg_key in best_chop_calibration:
                    avg_val = best_chop_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    chop_report_lines.append(f"   📈 Höchste {metric_name}: CHOP-{best_chop_calibration[best_key_name]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_chop_max_drawdown' in best_chop_calibration and 'avg_max_drawdown' in best_chop_calibration:
                avg_dd = best_chop_calibration['avg_max_drawdown']
                chop_report_lines.append(f"   🛡️ Niedrigster Drawdown: CHOP-{best_chop_calibration['best_chop_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            # CHOP-spezifische Trend-Analyse
            chop_report_lines.append(f"\n📊 CHOPPINESS INDEX TREND-ANALYSE")
            chop_report_lines.append("-" * 60)
            chop_report_lines.append("💡 CHOPPINESS INDEX INSIGHTS:")
            chop_report_lines.append(f"   • CHOP misst Market Choppiness vs. Trending Behavior")
            chop_report_lines.append(f"   • CHOP < 38.2: Starker Trend (Excellent for Trend-Following)")
            chop_report_lines.append(f"   • CHOP < 50: Moderater Trend (BULLISH - Long-Bias)")
            chop_report_lines.append(f"   • CHOP >= 50: Choppy/Ranging (BEARISH - Short-Bias)")
            chop_report_lines.append(f"   • CHOP > 61.8: Sehr choppy (Avoid Trend-Following)")
            chop_report_lines.append(f"   • CHOP hilft false breakouts zu vermeiden")
            
            # Trading-Empfehlungen
            chop_report_lines.append(f"\n⚡ CHOPPINESS INDEX TRADING EMPFEHLUNGEN")
            chop_report_lines.append("-" * 60)
            chop_report_lines.append("🔄 LONG/SHORT TRADING PRINZIPIEN:")
            chop_report_lines.append(f"   • Long Entry: CHOP fällt unter 50 (Market wird trending)")
            chop_report_lines.append(f"   • Short Entry: CHOP steigt über 50 (Market wird choppy)")
            chop_report_lines.append(f"   • Kombiniere CHOP mit Trend-Indikatoren für bessere Signale")
            chop_report_lines.append(f"   • Risk Management: Reduziere Position-Size bei CHOP > 61.8")
            
            chop_report_lines.append(f"\n📈 CHOP-LÄNGEN CHARAKTERISTIKA:")
            chop_report_lines.append(f"   • CHOP-3 bis CHOP-10: Sehr kurzfristig, hochfrequent, viele Signale")
            chop_report_lines.append(f"   • CHOP-11 bis CHOP-20: Kurzfristig, gute Reaktionszeit")
            chop_report_lines.append(f"   • CHOP-21 bis CHOP-50: Mittelfristig, balanced Performance")
            chop_report_lines.append(f"   • CHOP-50+: Langfristig, stabile Trend-Identifikation")
            
            chop_report_lines.append(f"\n💼 PRAKTISCHE ANWENDUNG:")
            chop_report_lines.append(f"   • CHOP als Filter: Nur traden wenn CHOP < 50 (Trending)")
            chop_report_lines.append(f"   • CHOP als Signal: CHOP crossing 50 als Entry/Exit")
            chop_report_lines.append(f"   • CHOP als Risk-Manager: Position-Size an CHOP-Wert anpassen")
            chop_report_lines.append(f"   • Kombiniere mit RSI/MACD für Trend-Confirmation")
            
            chop_report_lines.append(f"\n" + "=" * 68)
            chop_report_lines.append(f"🏁 CHOPPINESS INDEX-ANALYSE ABGESCHLOSSEN")
            chop_report_lines.append("=" * 68)
            
            # Speichere CHOP-spezifischen Bericht
            chop_report_text = "\n".join(chop_report_lines)
            chop_report_path = os.path.join(self.results_folder, 'chop_specific_analysis.txt')
            
            with open(chop_report_path, 'w', encoding='utf-8') as f:
                f.write(chop_report_text)
            
            print(f"📄 CHOP-spezifische Analyse gespeichert: {chop_report_path}")
            
        except Exception as e:
            print(f"❌ Fehler bei CHOP-spezifischer Analyse: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    Hauptfunktion für Choppiness Index Backtesting System
    """
    print("🚀 CHOPPINESS INDEX BACKTESTING SYSTEM START (Long/Short)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long/Short (CHOP < 50 = Long, CHOP >= 50 = Short)")
    print("   • CHOP-Range: 3 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • CHOP: Choppiness Index - misst Trend vs. Choppy")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe Choppiness Index Backtesting System aus
        chop_system = ChoppinessIndexBacktestingSystem(max_assets=20)
        
        # Teste verschiedene CHOP-Perioden (Einser-Schritte von 3 bis 150)
        chop_range = range(3, 151)  # 3, 4, 5, 6, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = chop_system.run_chop_backtests(chop_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            chop_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 Choppiness Index-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {chop_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim Choppiness Index-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
