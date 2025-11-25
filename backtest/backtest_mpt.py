"""
Multi Pivot Trend Backtesting System

Dieses System testet Multi Pivot Trend-basierte Trading-Strategien:
- Anzahl der Pivot-Längen: 2 bis 20
- Signal-basierte Long/Short/Neutral Strategie
- MPT Signal = 1 (bullish): Long Position
- MPT Signal = -1 (bearish): Short Position
- MPT Signal = 0 (neutral): Flat/No Position
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Backtesting-Funktionen
from _backtesting_base_ import BaseBacktestingSystem


class MultiPivotTrendBacktestingSystem(BaseBacktestingSystem):
    """
    Multi Pivot Trend-spezifisches Backtesting-System
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        """
        Initialisiert das Multi Pivot Trend Backtesting System
        
        Args:
            max_assets: Maximale Anzahl der Assets
            **kwargs: Zusätzliche Parameter (assets_csv, category, etc.)
        """
        super().__init__(max_assets, "MPT", **kwargs)
        self.indicator_name = "MPT"
        self.strategy_description = "MPT Signal = 1: Long | Signal = -1: Short | Signal = 0: Flat"
        self.threshold = None
    
    def _pivot_high(self, data, length):
        """Calculate pivot highs"""
        n = len(data)
        pivots = np.full(n, np.nan)
        
        for i in range(length, n - length):
            is_pivot = True
            center_val = data[i]
            
            for j in range(i - length, i + length + 1):
                if j != i and data[j] > center_val:
                    is_pivot = False
                    break
            
            if is_pivot:
                pivots[i] = center_val
        
        return pivots
    
    def _pivot_low(self, data, length):
        """Calculate pivot lows"""
        n = len(data)
        pivots = np.full(n, np.nan)
        
        for i in range(length, n - length):
            is_pivot = True
            center_val = data[i]
            
            for j in range(i - length, i + length + 1):
                if j != i and data[j] < center_val:
                    is_pivot = False
                    break
            
            if is_pivot:
                pivots[i] = center_val
        
        return pivots
    
    def calculate_multi_pivot_trend(self, high, low, close, pivot_lengths, mitigation='close', max_bars_back=100):
        """
        Berechnet Multi Pivot Trend direkt (ohne externen Import)
        
        Args:
            high, low, close: Preis-Arrays (können Pandas Series oder NumPy Arrays sein)
            pivot_lengths: Liste der Pivot-Längen
            mitigation: 'close', 'wicks', oder 'hl2'
            max_bars_back: Max Bars für Pivot-Bestätigung
            
        Returns:
            dict mit 'trend_avg', 'signal'
        """
        # Konvertiere zu NumPy Arrays falls nötig
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        n = len(close)
        hl2 = (high + low) / 2
        
        # Determine source
        if mitigation == 'close':
            src_high = close
            src_low = close
        elif mitigation == 'hl2':
            src_high = hl2
            src_low = hl2
        else:  # wicks
            src_high = high
            src_low = low
        
        # Store individual trends
        all_trends = []
        
        for length in pivot_lengths:
            trend = np.zeros(n)
            
            # Calculate pivots
            pivot_highs = self._pivot_high(high, length)
            pivot_lows = self._pivot_low(low, length)
            
            upper = np.full(n, np.nan)
            lower = np.full(n, np.nan)
            
            last_upper = np.nan
            last_lower = np.nan
            last_upper_idx = 0
            last_lower_idx = 0
            
            for i in range(2 * length, n):
                # Update pivot levels
                if not np.isnan(pivot_highs[i - length]):
                    last_upper = pivot_highs[i - length]
                    last_upper_idx = i - length
                    upper[i] = last_upper
                else:
                    upper[i] = last_upper
                
                if not np.isnan(pivot_lows[i - length]):
                    last_lower = pivot_lows[i - length]
                    last_lower_idx = i - length
                    lower[i] = last_lower
                else:
                    lower[i] = last_lower
                
                # Check breakouts
                if i > 0:
                    # Crossover (bullish)
                    if (not np.isnan(upper[i]) and 
                        src_high[i] > upper[i] and 
                        src_high[i-1] <= upper[i] and
                        trend[i-1] <= 0 and
                        i - last_upper_idx < max_bars_back):
                        trend[i] = 1
                    
                    # Crossunder (bearish)
                    elif (not np.isnan(lower[i]) and 
                          src_low[i] < lower[i] and 
                          src_low[i-1] >= lower[i] and
                          trend[i-1] >= 0 and
                          i - last_lower_idx < max_bars_back):
                        trend[i] = -1
                    
                    else:
                        trend[i] = trend[i-1]
            
            all_trends.append(trend)
        
        # Average all trends
        all_trends = np.array(all_trends)
        trend_avg = np.mean(all_trends, axis=0)
        
        # Signal (-1 to 1 scale)
        signal = np.clip(trend_avg, -1, 1)
        
        return {
            'trend_avg': trend_avg,
            'signal': signal
        }
    
    def calculate_mpt_signals(self, data: pd.DataFrame, max_pivot_length: int = 11) -> pd.DataFrame:
        """
        Berechnet Multi Pivot Trend-Signale für die Long/Short Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            max_pivot_length: Maximale Pivot-Länge (erstellt Sequenz von 2 bis max)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < max_pivot_length * 3:
            return pd.DataFrame()
        
        try:
            # Erstelle Pivot-Längen-Liste
            pivot_lengths = list(range(2, max_pivot_length + 1))
            
            # Berechne Multi Pivot Trend direkt
            result = self.calculate_multi_pivot_trend(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                pivot_lengths=pivot_lengths,
                mitigation='close'
            )
            
            signals_df = data.copy()
            signals_df['mpt_trend_avg'] = result['trend_avg']
            signals_df['mpt_signal'] = result['signal']
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna(subset=['mpt_signal'])
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Konvertiere zu discrete signals: >0.3=1, <-0.3=-1, sonst 0
            signals_df['position'] = np.where(signals_df['mpt_signal'] > 0.3, 1,
                                             np.where(signals_df['mpt_signal'] < -0.3, -1, 0))
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei MPT-Berechnung (MaxLength={max_pivot_length}): {e}")
            return pd.DataFrame()
    
    def run_mpt_backtests(self, mpt_range: range = None) -> pd.DataFrame:
        """
        Führt MPT-Backtests über verschiedene maximale Pivot-Längen durch
        
        Args:
            mpt_range: Range der maximalen Pivot-Längen zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        # Standard: 2 bis 25 in Einser-Schritten
        if mpt_range is None:
            mpt_range = range(2, 26)  # 2, 3, 4, ..., 25
        
        return self.run_generic_backtests(
            indicator_range=mpt_range,
            length_param_name='max_pivot_length',
            calculate_signals_func=self.calculate_mpt_signals,
            indicator_name='MPT'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden MPT-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit MPT-spezifischen Parametern
        strategy_description = "Multi Pivot Trend Strategy (Signal-based: 1=Long, 0=Flat, -1=Short)"
        super().generate_comprehensive_report(results_df, 'max_pivot_length', strategy_description)
        
        # Generiere spezifische Analyse (wie bei EMA)
        super().generate_specific_analysis(results_df, 'max_pivot_length',
                                          "Multi Pivot Trend: Composite Trend aus mehreren Pivot-Längen\n"
                                          "   • Long bei starkem Aufwärtstrend (Signal > 0.3)\n"
                                          "   • Short bei starkem Abwärtstrend (Signal < -0.3)\n"
                                          "   • Flat bei schwachem/neutralem Trend (|Signal| ≤ 0.3)\n"
                                          "   • Durchschnitt aus 2 bis max_pivot_length Pivot-Perioden\n"
                                          "   • Robuste Trend-Identifikation durch Multi-Timeframe-Ansatz",
                                          threshold=0.3)


def main():
    """
    Hauptfunktion für Multi Pivot Trend Backtesting System
    """
    print("🚀 MULTI PIVOT TREND BACKTESTING SYSTEM START")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long/Short/Flat basierend auf MPT Signal")
    print("   • Max Pivot Length Range: 2 bis 25 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • MPT: Multi Pivot Trend Indicator")
    print("   • Signal: >0.3=Long, <-0.3=Short, sonst Flat")
    
    try:
        # Erstelle und führe MPT Backtesting System aus
        mpt_system = MultiPivotTrendBacktestingSystem(max_assets=20)
        
        # Teste verschiedene maximale Pivot-Längen
        mpt_range = range(2, 26)  # 2, 3, 4, ..., 25
        
        # Führe Backtests durch
        results_df = mpt_system.run_mpt_backtests(mpt_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            mpt_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 MPT-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {mpt_system.results_folder}/")
        else:
            print("❌ Keine Ergebnisse generiert")
    
    except Exception as e:
        print(f"❌ Fehler beim MPT-Backtesting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
