"""
Supertrend Backtesting System (Matrix-Based)

Dieses System testet Supertrend-basierte Trading-Strategien:
- ATR Period: 5 bis 50
- Factor: 0.1 bis 10.0 (in 0.1-Schritten)
- Matrix-Kombination aller Parameter
- Signal-basierte Long/Short Strategie
- Supertrend Signal = 1 (bullish): Long Position
- Supertrend Signal = -1 (bearish): Short Position
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Backtesting-Funktionen
from _backtesting_base_ import BaseBacktestingSystem


class SupertrendBacktestingSystem(BaseBacktestingSystem):
    """
    Supertrend-spezifisches Backtesting-System (Matrix-Based)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        """
        Initialisiert das Supertrend Backtesting System
        
        Args:
            max_assets: Maximale Anzahl der Assets
            **kwargs: Zusätzliche Parameter (assets_csv, category, etc.)
        """
        super().__init__(max_assets, "SUPERTREND", **kwargs)
        self.indicator_name = "SUPERTREND"
        self.strategy_description = "Supertrend Signal = 1: Long | Signal = -1: Short"
        self.threshold = None
    
    def aggregate_metrics(self, asset_results: List[Dict]) -> Dict:
        """
        Aggregiert Metriken über mehrere Assets (für Matrix-Backtesting)
        
        Args:
            asset_results: Liste von Metrik-Dictionaries für verschiedene Assets
            
        Returns:
            Dictionary mit durchschnittlichen Metriken
        """
        if not asset_results:
            return {}
        
        # Extrahiere alle Metrik-Keys (außer 'symbol', 'atr_period', 'factor')
        metric_keys = [k for k in asset_results[0].keys() 
                      if k not in ['symbol', 'atr_period', 'factor']]
        
        # Berechne Durchschnitte
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in asset_results if key in r]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
        
        return avg_metrics
    
    def calculate_supertrend(self, high, low, close, atr_period=10, factor=3.0):
        """
        Berechnet Supertrend direkt (ohne externen Import)
        
        Args:
            high, low, close: Preis-Arrays (können Pandas Series oder NumPy Arrays sein)
            atr_period: ATR-Periode
            factor: ATR-Multiplikator
            
        Returns:
            dict mit 'supertrend', 'signal'
        """
        import talib as ta
        
        # Konvertiere zu NumPy Arrays falls nötig
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        # Calculate ATR
        atr = ta.ATR(high, low, close, timeperiod=atr_period)
        
        # Calculate bands
        hl2 = (high + low) / 2
        upperband = hl2 + (factor * atr)
        lowerband = hl2 - (factor * atr)
        
        # Initialize
        n = len(close)
        supertrend = np.zeros(n)
        direction = np.ones(n)  # 1=downtrend, -1=uptrend
        
        start_idx = atr_period
        
        # Initialize first value
        if close[start_idx] <= upperband[start_idx]:
            supertrend[start_idx] = upperband[start_idx]
            direction[start_idx] = 1
        else:
            supertrend[start_idx] = lowerband[start_idx]
            direction[start_idx] = -1
        
        # Calculate Supertrend
        for i in range(start_idx + 1, n):
            # Update bands
            if lowerband[i] > lowerband[i-1] or close[i-1] < lowerband[i-1]:
                lowerband[i] = lowerband[i]
            else:
                lowerband[i] = lowerband[i-1]
                
            if upperband[i] < upperband[i-1] or close[i-1] > upperband[i-1]:
                upperband[i] = upperband[i]
            else:
                upperband[i] = upperband[i-1]
            
            # Determine direction and supertrend
            if supertrend[i-1] == upperband[i-1] and close[i] <= upperband[i]:
                supertrend[i] = upperband[i]
                direction[i] = 1
            elif supertrend[i-1] == upperband[i-1] and close[i] > upperband[i]:
                supertrend[i] = lowerband[i]
                direction[i] = -1
            elif supertrend[i-1] == lowerband[i-1] and close[i] >= lowerband[i]:
                supertrend[i] = lowerband[i]
                direction[i] = -1
            elif supertrend[i-1] == lowerband[i-1] and close[i] < lowerband[i]:
                supertrend[i] = upperband[i]
                direction[i] = 1
        
        # Signal: flip direction (direction=-1 → signal=1, direction=1 → signal=-1)
        signal = -direction
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'signal': signal  # 1=bullish, -1=bearish
        }
    
    def calculate_supertrend_signals(self, data: pd.DataFrame, atr_period: int = 10, 
                                     factor: float = 3.0) -> pd.DataFrame:
        """
        Berechnet Supertrend-Signale für die Long/Short Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            atr_period: ATR-Periode
            factor: ATR-Multiplikator für Bands
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < atr_period + 10:
            return pd.DataFrame()
        
        try:
            # Berechne Supertrend direkt
            result = self.calculate_supertrend(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                atr_period=atr_period,
                factor=factor
            )
            
            signals_df = data.copy()
            signals_df['supertrend'] = result['supertrend']
            signals_df['supertrend_direction'] = result['direction']
            signals_df['supertrend_signal'] = result['signal']
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna(subset=['supertrend_signal'])
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Position = Signal
            signals_df['position'] = signals_df['supertrend_signal'].copy()
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei Supertrend-Berechnung (ATR={atr_period}, Factor={factor}): {e}")
            return pd.DataFrame()
    
    def run_supertrend_backtests(self, atr_range: range = None, 
                                 factor_range: np.ndarray = None) -> pd.DataFrame:
        """
        Führt Supertrend-Backtests über verschiedene Parameter durch (Matrix)
        
        Args:
            atr_range: Range der ATR-Perioden
            factor_range: Array der Factor-Werte
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        # Standard-Ranges
        if atr_range is None:
            atr_range = range(2, 151)  # 2 bis 150 in Einser-Schritten
        
        if factor_range is None:
            factor_range = np.arange(0.1, 10.1, 0.1)  # 0.1 bis 10.0 in 0.1-Schritten
        
        print(f"📊 Teste {len(atr_range)} ATR-Perioden × {len(factor_range)} Factors = {len(atr_range) * len(factor_range)} Kombinationen")
        
        all_results = []
        total_combinations = len(atr_range) * len(factor_range)
        current = 0
        
        for atr_period in atr_range:
            for factor in factor_range:
                current += 1
                
                # Statische Progress-Anzeige (überschreibt vorherige Zeile)
                print(f"\r   Progress: {current}/{total_combinations} ({current/total_combinations*100:.1f}%) | ATR={atr_period} Factor={factor:.1f}     ", end='', flush=True)
                
                # Teste für alle Assets
                asset_results = []
                
                for symbol in list(self.assets_data.keys())[:self.max_assets]:
                    data = self.assets_data.get(symbol)
                    if data is None or len(data) < 100:
                        continue
                    
                    signals_df = self.calculate_supertrend_signals(data, atr_period, factor)
                    if signals_df.empty:
                        continue
                    
                    # Übergebe nur die strategy_returns Series, nicht das ganze DataFrame
                    metrics = self.calculate_performance_metrics(signals_df['strategy_returns'])
                    if metrics:
                        metrics['symbol'] = symbol
                        metrics['atr_period'] = atr_period
                        metrics['factor'] = round(factor, 1)
                        asset_results.append(metrics)
                
                if asset_results:
                    # Aggregiere über Assets
                    avg_metrics = self.aggregate_metrics(asset_results)
                    avg_metrics['atr_period'] = atr_period
                    avg_metrics['factor'] = round(factor, 1)
                    avg_metrics['num_assets'] = len(asset_results)
                    all_results.append(avg_metrics)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            # Speichere Ergebnisse
            results_path = os.path.join(self.results_folder, 'supertrend_backtest_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"💾 Ergebnisse gespeichert: {results_path}")
            return results_df
        
        return pd.DataFrame()
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden Supertrend-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        print(f"\n📋 Erstelle Supertrend-Bericht...")
        
        # Finde beste Kombination
        best_idx = results_df['avg_sharpe_ratio'].idxmax()
        best_row = results_df.loc[best_idx]
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🎯 SUPERTREND BACKTESTING REPORT (MATRIX-BASED)")
        report_lines.append("=" * 80)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📊 Getestete Kombinationen: {len(results_df)}")
        report_lines.append(f"💰 Assets: {self.max_assets}")
        report_lines.append("")
        
        report_lines.append("🏆 BESTE KALIBRIERUNG (Sharpe Ratio):")
        report_lines.append(f"   • ATR Period: {best_row['atr_period']}")
        report_lines.append(f"   • Factor: {best_row['factor']:.1f}")
        report_lines.append(f"   • Sharpe Ratio: {best_row['avg_sharpe_ratio']:.4f}")
        report_lines.append(f"   • Total Return: {best_row['avg_total_return']:.2%}")
        report_lines.append(f"   • Max Drawdown: {best_row['avg_max_drawdown']:.2%}")
        report_lines.append(f"   • Sortino Ratio: {best_row['avg_sortino_ratio']:.4f}")
        report_lines.append("")
        
        # Top 10 Kombinationen
        report_lines.append("📊 TOP 10 KOMBINATIONEN (nach Sharpe Ratio):")
        top10 = results_df.nlargest(10, 'avg_sharpe_ratio')
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            report_lines.append(f"   {i}. ATR={row['atr_period']}, Factor={row['factor']:.1f} "
                              f"→ Sharpe={row['avg_sharpe_ratio']:.4f}, Return={row['avg_total_return']:.2%}")
        
        report_lines.append("")
        
        # Supertrend-spezifische Strategie-Insights
        report_lines.append("💡 SUPERTREND STRATEGIE EMPFEHLUNGEN")
        report_lines.append("-" * 60)
        report_lines.append("📋 SUPERTREND LONG/SHORT STRATEGIE INSIGHTS:")
        report_lines.append("   • Long bei Uptrend (Preis über Supertrend-Linie, Signal = 1)")
        report_lines.append("   • Short bei Downtrend (Preis unter Supertrend-Linie, Signal = -1)")
        report_lines.append("   • ATR-basierte dynamische Bands passen sich an Volatilität an")
        report_lines.append("   • Factor bestimmt Band-Breite (höher = weniger Signale)")
        report_lines.append("   • Trendfolge-Indikator mit klaren Entry/Exit-Punkten")
        report_lines.append("")
        
        report_lines.append("🎯 PARAMETER-OPTIMIERUNG")
        report_lines.append("-" * 60)
        report_lines.append("📌 ATR PERIOD CHARAKTERISTIKA:")
        report_lines.append("   • ATR 5-15: Kurzfristig, reagiert schnell auf Volatilitätsänderungen")
        report_lines.append("   • ATR 15-30: Mittelfristig, balanced zwischen Reaktivität und Stabilität")
        report_lines.append("   • ATR 30-50: Langfristig, glattere Signale, weniger Whipsaws")
        report_lines.append("")
        report_lines.append("📌 FACTOR CHARAKTERISTIKA:")
        report_lines.append("   • Factor 0.5-2.0: Eng, mehr Trades, sensitiver")
        report_lines.append("   • Factor 2.0-5.0: Medium, ausgewogene Trade-Frequenz")
        report_lines.append("   • Factor 5.0-10.0: Weit, weniger aber qualitativ hochwertige Signale")
        report_lines.append("")
        
        # Analyse der besten Parameter-Bereiche
        # Finde häufigste ATR Perioden in Top 20
        top20 = results_df.nlargest(20, 'avg_sharpe_ratio')
        common_atr = top20['atr_period'].mode()
        common_factor = top20['factor'].mode()
        
        if len(common_atr) > 0:
            report_lines.append(f"📈 HÄUFIGSTE TOP-PARAMETER (Top 20):")
            report_lines.append(f"   • ATR Period: {common_atr.values[0]} (kommt am häufigsten in Top 20 vor)")
            if len(common_factor) > 0:
                report_lines.append(f"   • Factor: {common_factor.values[0]:.1f}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_folder, 'supertrend_specific_analysis.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 Bericht gespeichert: {report_path}")
        print(report_text)


def main():
    """
    Hauptfunktion für Supertrend Backtesting System
    """
    print("🚀 SUPERTREND BACKTESTING SYSTEM START")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long/Short basierend auf Supertrend Signal")
    print("   • ATR Period Range: 2 bis 150 (Einser-Schritte)")
    print("   • Factor Range: 0.5 bis 10.0 (0.5-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Supertrend: ATR-based Trend Indicator")
    print("   • Signal: Bullish=1, Bearish=-1")
    
    try:
        # Erstelle und führe Supertrend Backtesting System aus
        supertrend_system = SupertrendBacktestingSystem(max_assets=20)
        
        # Teste verschiedene Parameter (Matrix)
        atr_range = range(2, 151)  # 2 bis 150
        factor_range = np.arange(0.5, 10.1, 0.5)  # 0.5 bis 10.0 in 0.5-Schritten
        
        # Führe Backtests durch
        results_df = supertrend_system.run_supertrend_backtests(atr_range, factor_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            supertrend_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 Supertrend-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {supertrend_system.results_folder}/")
        else:
            print("❌ Keine Ergebnisse generiert")
    
    except Exception as e:
        print(f"❌ Fehler beim Supertrend-Backtesting: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
