"""
Williams %R Backtesting System (Optimiert mit Base Class)

Dieses System testet Williams %R-basierte Trading-Strategien über verschiedene Assets und Perioden:
- WILLR-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: WILLR > -50 = Long Position, WILLR <= -50 = Cash Position
- Verwendet gemeinsame Funktionen aus backtesting_base.py

Williams %R:
- Momentum-Oszillator entwickelt von Larry Williams
- Berechnung: ((Highest High - Close) / (Highest High - Lowest Low)) * -100
- Werte zwischen -100 und 0
- Long-Signal: WILLR > -50 (über der Mittellinie, bullisher Momentum)
- Exit-Signal: WILLR <= -50 (unter der Mittellinie, bearisher Momentum)
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

class WILLRBacktestingSystem(BaseBacktestingSystem):
    """
    Williams %R-spezifisches Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "WILLR", **kwargs)
        self.indicator_name = "WILLR"
        self.strategy_description = "WILLR > -50: Long-Position | WILLR <= -50: Cash-Position"
        self.threshold = -50.0
    
    def calculate_willr_signals(self, data: pd.DataFrame, willr_length: int = 14) -> pd.DataFrame:
        """
        Berechnet Williams %R-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            willr_length: Williams %R-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < willr_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne Williams %R
            willr = ta.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod=willr_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['willr'] = willr
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Williams %R Long-Only Signale (WILLR > -50 = Long, WILLR <= -50 = Cash)
            signals_df['position'] = np.where(signals_df['willr'] > -50, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei Williams %R-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_willr_backtests(self, willr_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt Williams %R-Backtests über verschiedene Perioden durch
        
        Args:
            willr_range: Range der Williams %R-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=willr_range,
            length_param_name='willr_length',
            calculate_signals_func=self.calculate_willr_signals,
            indicator_name='Williams %R'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden Williams %R-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit WILLR-spezifischen Parametern
        strategy_description = "Williams %R Long-Only Strategy (WILLR > -50 = Long, WILLR <= -50 = Cash)"
        super().generate_comprehensive_report(results_df, 'willr_length', strategy_description)
        
        # Zusätzliche Williams %R-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle WILLR-spezifische Analyse...")
        
        # Verwende die WILLR-spezifische Analyse-Funktion
        self.generate_willr_specific_analysis(results_df)
    
    def generate_willr_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert Williams %R-spezifische Analyse
        Fokussiert auf die besten WILLR-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für WILLR-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle WILLR-spezifische Analyse...")
        
        # Finde beste WILLR-Kalibrierungen (nutzt gemeinsame Funktion)
        best_willr_calibration = self.find_best_average_calibration(results_df, 'willr_length')
        
        if best_willr_calibration:
            # Erstelle zusätzlichen WILLR-Bericht
            willr_report_lines = []
            willr_report_lines.append("=" * 80)
            willr_report_lines.append("🎯 WILLIAMS %R-SPEZIFISCHE ANALYSE")
            willr_report_lines.append("=" * 80)
            willr_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            willr_report_lines.append("")
            
            # Beste kombinierte WILLR-Länge
            if 'best_willr_combined' in best_willr_calibration:
                willr_report_lines.append(
                    f"🥇 Beste Durchschnittliche WILLR-Länge (Kombiniert): WILLR-{best_willr_calibration['best_willr_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte WILLR
                combined_keys = ['combined_willr_sharpe_ratio', 'combined_willr_total_return', 'combined_willr_max_drawdown']
                if all(key in best_willr_calibration for key in combined_keys):
                    willr_report_lines.append(
                        f"   📊 Avg Sharpe: {best_willr_calibration['combined_willr_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_willr_calibration['combined_willr_total_return']:.1%} | "
                        f"Avg DD: {best_willr_calibration['combined_willr_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_willr_sortino_ratio' in best_willr_calibration and 'combined_willr_win_rate' in best_willr_calibration:
                    willr_report_lines.append(
                        f"   📈 Avg Sortino: {best_willr_calibration['combined_willr_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_willr_calibration['combined_willr_win_rate']:.1%} | "
                        f"Score: {best_willr_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top WILLRs für einzelne Metriken
            willr_report_lines.append("")
            willr_report_lines.append("📈 Beste Durchschnitts-WILLRs nach Metriken:")
            
            metric_keys = [
                ('best_willr_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_willr_total_return', 'avg_total_return', 'Total Return'),
                ('best_willr_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_willr_calibration and avg_key in best_willr_calibration:
                    avg_val = best_willr_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    willr_report_lines.append(f"   • {metric_name}: WILLR-{best_willr_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # WILLR-spezifische Empfehlungen
            willr_report_lines.append(f"\n💡 WILLIAMS %R STRATEGIE EMPFEHLUNGEN")
            willr_report_lines.append("-" * 60)
            willr_report_lines.append("📋 WILLR LONG-ONLY STRATEGIE INSIGHTS:")
            willr_report_lines.append(f"   • Long-Position wenn WILLR > -50 (über der Mittellinie)")
            willr_report_lines.append(f"   • Cash-Position wenn WILLR <= -50 (unter der Mittellinie)")
            willr_report_lines.append(f"   • WILLR Werte zwischen -100 und 0")
            willr_report_lines.append(f"   • -50 ist die kritische Mittellinie für Momentum")
            willr_report_lines.append(f"   • Kürzere WILLR-Perioden: Mehr Trades, höhere Sensitivität")
            willr_report_lines.append(f"   • Längere WILLR-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            willr_report_lines.append(f"\n🎯 WILLR PARAMETER-OPTIMIERUNG")
            willr_report_lines.append("-" * 60)
            willr_report_lines.append(f"📌 OPTIMALE WILLR-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_willr_combined' in best_willr_calibration:
                combined_score = best_willr_calibration.get('avg_combined_score', 0.0)
                willr_report_lines.append(f"   🥇 Beste Gesamtperformance: WILLR-{best_willr_calibration['best_willr_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_willr_calibration and avg_key in best_willr_calibration:
                    avg_val = best_willr_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    willr_report_lines.append(f"   📈 Höchste {metric_name}: WILLR-{best_willr_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_willr_max_drawdown' in best_willr_calibration and 'avg_max_drawdown' in best_willr_calibration:
                avg_dd = best_willr_calibration['avg_max_drawdown']
                willr_report_lines.append(f"   🛡️ Niedrigster Drawdown: WILLR-{best_willr_calibration['best_willr_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            willr_report_lines.append(f"\n" + "=" * 68)
            willr_report_lines.append(f"🏁 WILLR-ANALYSE ABGESCHLOSSEN")
            willr_report_lines.append("=" * 68)
            
            # Speichere WILLR-spezifischen Bericht
            willr_report_text = "\n".join(willr_report_lines)
            willr_report_path = os.path.join(self.results_folder, 'willr_specific_analysis.txt')
            
            with open(willr_report_path, 'w', encoding='utf-8') as f:
                f.write(willr_report_text)
            
            print(f"📄 WILLR-spezifische Analyse gespeichert: {willr_report_path}")

def main():
    """
    Hauptfunktion für Williams %R Backtesting System
    """
    print("🚀 WILLIAMS %R BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (WILLR > -50)")
    print("   • WILLR-Range: 5 bis 150 (Schritte: 5)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe Williams %R Backtesting System aus
        willr_system = WILLRBacktestingSystem(max_assets=20)
        
        # Teste verschiedene WILLR-Perioden (Einser-Schritte wie im Original)
        willr_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = willr_system.run_willr_backtests(willr_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            willr_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 WILLR-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {willr_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim WILLR-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()