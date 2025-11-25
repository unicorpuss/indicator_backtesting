"""
RSI Backtesting System (Optimiert mit Base Class)

Dieses System testet RSI-basierte Trading-Strategien über verschiedene Assets und Perioden:
- RSI-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: RSI > 50 = Long Position, RSI <= 50 = Cash Position
- Verwendet gemeinsame Funktionen aus backtesting_base.py

RSI (Relative Strength Index):
- Momentum-Oszillator entwickelt von J. Welles Wilder
- Berechnung: 100 - (100 / (1 + RS)), wobei RS = Average Gain / Average Loss
- Werte zwischen 0 und 100
- Long-Signal: RSI > 50 (bullisher Momentum)
- Exit-Signal: RSI <= 50 (neutraler/bearisher Momentum)
"""

import talib as ta
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

class RSIBacktestingSystem(BaseBacktestingSystem):
    """
    RSI-spezifisches Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "RSI", **kwargs)
        self.indicator_name = "RSI"
        self.strategy_description = "RSI > 50: Long-Position | RSI <= 50: Cash-Position"
        self.threshold = 50.0
    
    def calculate_rsi_signals(self, data: pd.DataFrame, rsi_length: int = 14) -> pd.DataFrame:
        """
        Berechnet RSI-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            rsi_length: RSI-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < rsi_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne RSI
            rsi = ta.RSI(data['close'].values, timeperiod=rsi_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['rsi'] = rsi
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # RSI Long-Only Signale (RSI > 50 = Long, RSI <= 50 = Cash)
            signals_df['position'] = np.where(signals_df['rsi'] > 50, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei RSI-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_rsi_backtests(self, rsi_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt RSI-Backtests über verschiedene Perioden durch
        
        Args:
            rsi_range: Range der RSI-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=rsi_range,
            length_param_name='rsi_length',
            calculate_signals_func=self.calculate_rsi_signals,
            indicator_name='RSI'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden RSI-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit RSI-spezifischen Parametern
        strategy_description = "RSI Long-Only Strategy (RSI > 50 = Long, RSI <= 50 = Cash)"
        super().generate_comprehensive_report(results_df, 'rsi_length', strategy_description)
        
        # Zusätzliche RSI-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle RSI-spezifische Analyse...")
        
        # Verwende die RSI-spezifische Analyse-Funktion
        self.generate_rsi_specific_analysis(results_df)
    
    def generate_rsi_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert RSI-spezifische Analyse ähnlich der EMA-Analyse
        Fokussiert auf die besten RSI-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für RSI-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle RSI-spezifische Analyse...")
        
        # Finde beste RSI-Kalibrierungen (nutzt gemeinsame Funktion)
        best_rsi_calibration = self.find_best_average_calibration(results_df, 'rsi_length')
        
        if best_rsi_calibration:
            # Erstelle zusätzlichen RSI-Bericht
            rsi_report_lines = []
            rsi_report_lines.append("=" * 80)
            rsi_report_lines.append("🎯 RSI-SPEZIFISCHE ANALYSE")
            rsi_report_lines.append("=" * 80)
            rsi_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            rsi_report_lines.append("")
            
            # Beste kombinierte RSI-Länge
            if 'best_rsi_combined' in best_rsi_calibration:
                rsi_report_lines.append(
                    f"🥇 Beste Durchschnittliche RSI-Länge (Kombiniert): RSI-{best_rsi_calibration['best_rsi_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte RSI
                combined_keys = ['combined_rsi_sharpe_ratio', 'combined_rsi_total_return', 'combined_rsi_max_drawdown']
                if all(key in best_rsi_calibration for key in combined_keys):
                    rsi_report_lines.append(
                        f"   📊 Avg Sharpe: {best_rsi_calibration['combined_rsi_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_rsi_calibration['combined_rsi_total_return']:.1%} | "
                        f"Avg DD: {best_rsi_calibration['combined_rsi_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_rsi_sortino_ratio' in best_rsi_calibration and 'combined_rsi_win_rate' in best_rsi_calibration:
                    rsi_report_lines.append(
                        f"   📈 Avg Sortino: {best_rsi_calibration['combined_rsi_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_rsi_calibration['combined_rsi_win_rate']:.1%} | "
                        f"Score: {best_rsi_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top RSIs für einzelne Metriken
            rsi_report_lines.append("")
            rsi_report_lines.append("📈 Beste Durchschnitts-RSIs nach Metriken:")
            
            metric_keys = [
                ('best_rsi_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_rsi_total_return', 'avg_total_return', 'Total Return'),
                ('best_rsi_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_rsi_calibration and avg_key in best_rsi_calibration:
                    avg_val = best_rsi_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    rsi_report_lines.append(f"   • {metric_name}: RSI-{best_rsi_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # RSI-spezifische Empfehlungen
            rsi_report_lines.append(f"\n💡 RSI STRATEGIE EMPFEHLUNGEN")
            rsi_report_lines.append("-" * 60)
            rsi_report_lines.append("📋 RSI LONG-ONLY STRATEGIE INSIGHTS:")
            rsi_report_lines.append(f"   • Long-Position wenn RSI > 50 (bullisher Momentum)")
            rsi_report_lines.append(f"   • Cash-Position wenn RSI <= 50 (neutraler/bearisher Momentum)")
            rsi_report_lines.append(f"   • Kürzere RSI-Perioden: Mehr Trades, höhere Sensitivität")
            rsi_report_lines.append(f"   • Längere RSI-Perioden: Weniger Trades, stabilere Signale")
            rsi_report_lines.append(f"   • RSI-50 als Mittellinie für Trend-Bestimmung")
            
            # Parameter-Optimierung
            rsi_report_lines.append(f"\n🎯 RSI PARAMETER-OPTIMIERUNG")
            rsi_report_lines.append("-" * 60)
            rsi_report_lines.append(f"📌 OPTIMALE RSI-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_rsi_combined' in best_rsi_calibration:
                combined_score = best_rsi_calibration.get('avg_combined_score', 0.0)
                rsi_report_lines.append(f"   🥇 Beste Gesamtperformance: RSI-{best_rsi_calibration['best_rsi_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_rsi_calibration and avg_key in best_rsi_calibration:
                    avg_val = best_rsi_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    rsi_report_lines.append(f"   📈 Höchste {metric_name}: RSI-{best_rsi_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_rsi_max_drawdown' in best_rsi_calibration and 'avg_max_drawdown' in best_rsi_calibration:
                avg_dd = best_rsi_calibration['avg_max_drawdown']
                rsi_report_lines.append(f"   🛡️ Niedrigster Drawdown: RSI-{best_rsi_calibration['best_rsi_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            rsi_report_lines.append(f"\n" + "=" * 68)
            rsi_report_lines.append(f"🏁 RSI-ANALYSE ABGESCHLOSSEN")
            rsi_report_lines.append("=" * 68)
            
            # Speichere RSI-spezifischen Bericht
            rsi_report_text = "\n".join(rsi_report_lines)
            rsi_report_path = os.path.join(self.results_folder, 'rsi_specific_analysis.txt')
            
            with open(rsi_report_path, 'w', encoding='utf-8') as f:
                f.write(rsi_report_text)
            
            print(f"📄 RSI-spezifische Analyse gespeichert: {rsi_report_path}")

def main():
    """
    Hauptfunktion für RSI Backtesting System
    """
    print("🚀 RSI BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (RSI > 50)")
    print("   • RSI-Range: 5 bis 150 (Schritte: 5)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe RSI Backtesting System aus
        rsi_system = RSIBacktestingSystem(max_assets=20)
        
        # Teste verschiedene RSI-Perioden (Einser-Schritte wie im Original)
        rsi_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = rsi_system.run_rsi_backtests(rsi_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            rsi_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 RSI-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {rsi_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim RSI-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()