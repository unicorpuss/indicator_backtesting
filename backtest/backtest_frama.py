"""
FRAMA Channel Backtesting System

Dieses System testet FRAMA Channel-basierte Trading-Strategien:
- FRAMA-Längen von 6 bis 100 (nur gerade Zahlen)
- Signal-basierte Long/Short Strategie
- FRAMA Signal = 1 (bullish): Long Position
- FRAMA Signal = -1 (bearish): Short Position  
- FRAMA Signal = 0 (neutral): Flat/No Position
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


class FRAMABacktestingSystem(BaseBacktestingSystem):
    """
    FRAMA Channel-spezifisches Backtesting-System
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        """
        Initialisiert das FRAMA Backtesting System
        
        Args:
            max_assets: Maximale Anzahl der Assets
            **kwargs: Zusätzliche Parameter (assets_csv, category, etc.)
        """
        super().__init__(max_assets, "FRAMA", **kwargs)
        self.indicator_name = "FRAMA"
        self.strategy_description = "FRAMA Signal = 1: Long | Signal = -1: Short | Signal = 0: Flat"
        self.threshold = None
    
    def calculate_frama(self, high, low, close, length=26, distance=1.5):
        """
        Berechnet FRAMA Channel direkt (ohne externen Import)
        
        Args:
            high, low, close: Preis-Arrays (können Pandas Series oder NumPy Arrays sein)
            length: FRAMA-Periode (muss gerade sein)
            distance: Multiplier für Bands
            
        Returns:
            dict mit 'frama', 'upper_band', 'lower_band', 'signal'
        """
        import talib as ta
        
        # Konvertiere zu NumPy Arrays falls nötig
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        n = len(close)
        hl2 = (high + low) / 2
        
        # Volatilität
        volatility = ta.SMA(high - low, timeperiod=200)
        
        # Initialize arrays
        n1 = np.zeros(n)
        n2 = np.zeros(n)
        n3 = np.zeros(n)
        dimension = np.zeros(n)
        alpha = np.zeros(n)
        frama = np.zeros(n)
        
        half_length = length // 2
        
        # Calculate fractal dimension and FRAMA
        for i in range(length, n):
            # N3: Range over full period
            n3[i] = (np.max(high[i-length+1:i+1]) - np.min(low[i-length+1:i+1])) / length
            
            # N1: Range over first half
            n1[i] = (np.max(high[i-length+1:i-half_length+1]) - np.min(low[i-length+1:i-half_length+1])) / half_length
            
            # N2: Range over second half
            n2[i] = (np.max(high[i-half_length+1:i+1]) - np.min(low[i-half_length+1:i+1])) / half_length
            
            # Calculate fractal dimension
            if n1[i] > 0 and n2[i] > 0 and n3[i] > 0:
                dimension[i] = (np.log(n1[i] + n2[i]) - np.log(n3[i])) / np.log(2)
            
            # Calculate alpha
            alpha[i] = np.exp(-4.6 * (dimension[i] - 1))
            alpha[i] = np.clip(alpha[i], 0.01, 1.0)
            
            # Calculate FRAMA
            if i == length:
                frama[i] = hl2[i]
            else:
                frama[i] = alpha[i] * hl2[i] + (1 - alpha[i]) * frama[i-1]
        
        # Smoothing
        frama_smoothed = ta.SMA(frama, timeperiod=5)
        frama_smoothed = np.where(np.isnan(frama_smoothed), frama, frama_smoothed)
        
        # Channel bands
        upper_band = frama_smoothed + volatility * distance
        lower_band = frama_smoothed - volatility * distance
        
        # Calculate signals
        hlc3 = (high + low + close) / 3
        signal = np.zeros(n)
        
        for i in range(1, n):
            # Breakout up (bullish)
            if hlc3[i] > upper_band[i] and hlc3[i-1] <= upper_band[i-1]:
                signal[i] = 1
            # Breakout down (bearish)
            elif hlc3[i] < lower_band[i] and hlc3[i-1] >= lower_band[i-1]:
                signal[i] = -1
            # Cross FRAMA (neutral)
            elif (close[i] > frama_smoothed[i] and close[i-1] <= frama_smoothed[i-1]) or \
                 (close[i] < frama_smoothed[i] and close[i-1] >= frama_smoothed[i-1]):
                signal[i] = 0
            else:
                signal[i] = signal[i-1]
        
        return {
            'frama': frama_smoothed,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'signal': signal
        }
    
    def calculate_frama_signals(self, data: pd.DataFrame, frama_length: int = 26) -> pd.DataFrame:
        """
        Berechnet FRAMA-Signale für die Long/Short Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            frama_length: FRAMA-Periode (muss gerade sein)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < frama_length + 50:
            return pd.DataFrame()
        
        try:
            # Berechne FRAMA Channel direkt
            result = self.calculate_frama(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                length=frama_length,
                distance=1.5
            )
            
            signals_df = data.copy()
            signals_df['frama'] = result['frama']
            signals_df['frama_upper'] = result['upper_band']
            signals_df['frama_lower'] = result['lower_band']
            signals_df['frama_signal'] = result['signal']
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna(subset=['frama_signal'])
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Position = Signal
            signals_df['position'] = signals_df['frama_signal'].copy()
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei FRAMA-Berechnung (Length={frama_length}): {e}")
            return pd.DataFrame()
    
    def run_frama_backtests(self, frama_range: range = None) -> pd.DataFrame:
        """
        Führt FRAMA-Backtests über verschiedene Perioden durch
        
        Args:
            frama_range: Range der FRAMA-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        # Standard: 2 bis 150 in Einser-Schritten
        if frama_range is None:
            frama_range = range(2, 151)  # 2, 3, 4, ..., 150
        
        return self.run_generic_backtests(
            indicator_range=frama_range,
            length_param_name='frama_length',
            calculate_signals_func=self.calculate_frama_signals,
            indicator_name='FRAMA'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden FRAMA-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit FRAMA-spezifischen Parametern
        strategy_description = "FRAMA Channel Strategy (Signal-based: 1=Long, 0=Flat, -1=Short)"
        super().generate_comprehensive_report(results_df, 'frama_length', strategy_description)
        
        # Generiere spezifische Analyse (wie bei EMA)
        super().generate_specific_analysis(results_df, 'frama_length', 
                                          "FRAMA Channel: Adaptive Moving Average mit Fractal Dimension\n"
                                          "   • Long bei Breakout über Upper Band (Signal = 1)\n"
                                          "   • Short bei Breakout unter Lower Band (Signal = -1)\n"
                                          "   • Flat bei Preis zwischen Bands (Signal = 0)\n"
                                          "   • Bands passen sich an Markt-Volatilität an\n"
                                          "   • Fractal Dimension bestimmt Adaptionsgeschwindigkeit")


def main():
    """
    Hauptfunktion für FRAMA Backtesting System
    """
    print("🚀 FRAMA CHANNEL BACKTESTING SYSTEM START")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long/Short/Flat basierend auf FRAMA Signal")
    print("   • FRAMA-Range: 2 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • FRAMA: Fractal Adaptive Moving Average Channel")
    print("   • Signal: Bullish=1, Neutral=0, Bearish=-1")
    
    try:
        # Erstelle und führe FRAMA Backtesting System aus
        frama_system = FRAMABacktestingSystem(max_assets=20)
        
        # Teste verschiedene FRAMA-Perioden
        frama_range = range(2, 151)  # 2, 3, 4, ..., 150
        
        # Führe Backtests durch
        results_df = frama_system.run_frama_backtests(frama_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            frama_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 FRAMA-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {frama_system.results_folder}/")
        else:
            print("❌ Keine Ergebnisse generiert")
    
    except Exception as e:
        print(f"❌ Fehler beim FRAMA-Backtesting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
