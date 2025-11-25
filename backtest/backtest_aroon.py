"""
AROON OSCILLATOR Backtesting System (Optimiert mit Base Class)

Dieses System testet AROON OSCILLATOR-basierte Trading-Strategien über verschiedene Assets und Perioden:
- AROON-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: Aroon Oscillator > 0 = Long Position, sonst Cash Position
- Verwendet gemeinsame Funktionen aus backtesting_base.py

AROON OSCILLATOR-Indikator:
- Momentum-Indikator entwickelt von Tushar Chande  
- Berechnung: Aroon Up - Aroon Down
- Aroon Up: ((n - Perioden seit höchstem Hoch) / n) * 100
- Aroon Down: ((n - Perioden seit tiefstem Tief) / n) * 100
- Aroon Oscillator: Aroon Up - Aroon Down
- Werte zwischen -100 und +100 (0 = Mittellinie)
- Long-Signal: Aroon Oscillator > 0 (bullisches Momentum)
- Cash-Signal: Aroon Oscillator <= 0 (bearisches/neutrales Momentum)
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

class AROONBacktestingSystem(BaseBacktestingSystem):
    """
    AROON-spezifisches Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "AROON", **kwargs)
        self.indicator_name = "AROON"
        self.strategy_description = "Aroon Oscillator > 0: Long-Position | Aroon <= 0: Cash-Position"
        self.threshold = 0.0  # Aroon Oscillator Mittellinie (0)
    
    def calculate_aroon_signals(self, data: pd.DataFrame, aroon_length: int = 14) -> pd.DataFrame:
        """
        Berechnet Aroon Oscillator-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            aroon_length: Aroon-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < aroon_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne Aroon Oscillator mit talib (nicht separate Up/Down)
            aroon_osc = ta.AROONOSC(data['high'].values, data['low'].values, timeperiod=aroon_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['aroon'] = aroon_osc
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Aroon Oscillator Long-Only Signale (Aroon > 0 = Long, Aroon <= 0 = Cash)
            # 0 ist die Mittellinie des Aroon Oscillators (-100 bis +100)
            signals_df['position'] = np.where(signals_df['aroon'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei Aroon Oscillator-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_aroon_backtests(self, aroon_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt AROON-Backtests über verschiedene Perioden durch
        
        Args:
            aroon_range: Range der AROON-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=aroon_range,
            length_param_name='aroon_length',
            calculate_signals_func=self.calculate_aroon_signals,
            indicator_name='AROON'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden AROON-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit AROON-spezifischen Parametern
        strategy_description = "Aroon Oscillator Long-Only Strategy (Aroon > 0 = Long, Aroon <= 0 = Cash)"
        super().generate_comprehensive_report(results_df, 'aroon_length', strategy_description)
        
        # Zusätzliche AROON-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle AROON-spezifische Analyse...")
        
        # Verwende die AROON-spezifische Analyse-Funktion
        self.generate_aroon_specific_analysis(results_df)
    
    def generate_aroon_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert AROON-spezifische Analyse
        Fokussiert auf die besten AROON-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für AROON-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle AROON-spezifische Analyse...")
        
        # Finde beste AROON-Kalibrierungen (nutzt gemeinsame Funktion)
        best_aroon_calibration = self.find_best_average_calibration(results_df, 'aroon_length')
        
        if best_aroon_calibration:
            # Erstelle zusätzlichen AROON-Bericht
            aroon_report_lines = []
            aroon_report_lines.append("=" * 80)
            aroon_report_lines.append("🎯 AROON-SPEZIFISCHE ANALYSE")
            aroon_report_lines.append("=" * 80)
            aroon_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            aroon_report_lines.append("")
            
            # Beste kombinierte AROON-Länge
            if 'best_aroon_combined' in best_aroon_calibration:
                aroon_report_lines.append(
                    f"🥇 Beste Durchschnittliche AROON-Länge (Kombiniert): AROON-{best_aroon_calibration['best_aroon_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte AROON
                combined_keys = ['combined_aroon_sharpe_ratio', 'combined_aroon_total_return', 'combined_aroon_max_drawdown']
                if all(key in best_aroon_calibration for key in combined_keys):
                    aroon_report_lines.append(
                        f"   📊 Avg Sharpe: {best_aroon_calibration['combined_aroon_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_aroon_calibration['combined_aroon_total_return']:.1%} | "
                        f"Avg DD: {best_aroon_calibration['combined_aroon_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_aroon_sortino_ratio' in best_aroon_calibration and 'combined_aroon_win_rate' in best_aroon_calibration:
                    aroon_report_lines.append(
                        f"   📈 Avg Sortino: {best_aroon_calibration['combined_aroon_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_aroon_calibration['combined_aroon_win_rate']:.1%} | "
                        f"Score: {best_aroon_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top AROONs für einzelne Metriken
            aroon_report_lines.append("")
            aroon_report_lines.append("📈 Beste Durchschnitts-AROONs nach Metriken:")
            
            metric_keys = [
                ('best_aroon_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_aroon_total_return', 'avg_total_return', 'Total Return'),
                ('best_aroon_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_aroon_calibration and avg_key in best_aroon_calibration:
                    avg_val = best_aroon_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    aroon_report_lines.append(f"   • {metric_name}: AROON-{best_aroon_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # AROON-spezifische Empfehlungen
            aroon_report_lines.append(f"\n💡 AROON STRATEGIE EMPFEHLUNGEN")
            aroon_report_lines.append("-" * 60)
            aroon_report_lines.append("📋 AROON OSCILLATOR LONG-ONLY STRATEGIE INSIGHTS:")
            aroon_report_lines.append(f"   • Long-Position wenn Aroon Oscillator > 0 (bullisches Momentum)")
            aroon_report_lines.append(f"   • Cash-Position wenn Aroon Oscillator <= 0 (bearisches Momentum)")
            aroon_report_lines.append(f"   • Aroon Oscillator = Aroon Up - Aroon Down")
            aroon_report_lines.append(f"   • Range: -100 bis +100 mit 0 als Mittellinie")
            aroon_report_lines.append(f"   • > 0: Aufwärtstrend dominiert über Abwärtstrend")
            aroon_report_lines.append(f"   • < 0: Abwärtstrend dominiert über Aufwärtstrend")
            aroon_report_lines.append(f"   • Kürzere AROON-Perioden: Mehr Trades, höhere Sensitivität")
            aroon_report_lines.append(f"   • Längere AROON-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            aroon_report_lines.append(f"\n🎯 AROON PARAMETER-OPTIMIERUNG")
            aroon_report_lines.append("-" * 60)
            aroon_report_lines.append(f"📌 OPTIMALE AROON-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_aroon_combined' in best_aroon_calibration:
                combined_score = best_aroon_calibration.get('avg_combined_score', 0.0)
                aroon_report_lines.append(f"   🥇 Beste Gesamtperformance: AROON-{best_aroon_calibration['best_aroon_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_aroon_calibration and avg_key in best_aroon_calibration:
                    avg_val = best_aroon_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    aroon_report_lines.append(f"   📈 Höchste {metric_name}: AROON-{best_aroon_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_aroon_max_drawdown' in best_aroon_calibration and 'avg_max_drawdown' in best_aroon_calibration:
                avg_dd = best_aroon_calibration['avg_max_drawdown']
                aroon_report_lines.append(f"   🛡️ Niedrigster Drawdown: AROON-{best_aroon_calibration['best_aroon_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            aroon_report_lines.append(f"\n" + "=" * 68)
            aroon_report_lines.append(f"🏁 AROON-ANALYSE ABGESCHLOSSEN")
            aroon_report_lines.append("=" * 68)
            
            # Speichere AROON-spezifischen Bericht
            aroon_report_text = "\n".join(aroon_report_lines)
            aroon_report_path = os.path.join(self.results_folder, 'aroon_specific_analysis.txt')
            
            with open(aroon_report_path, 'w', encoding='utf-8') as f:
                f.write(aroon_report_text)
            
            print(f"📄 AROON-spezifische Analyse gespeichert: {aroon_report_path}")

def main():
    """
    Hauptfunktion für AROON Backtesting System
    """
    print("🚀 AROON BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (Aroon Oscillator > 0)")
    print("   • AROON-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Aroon Oscillator: -100 bis +100 (0 = Mittellinie)")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe AROON Backtesting System aus
        aroon_system = AROONBacktestingSystem(max_assets=20)
        
        # Teste verschiedene AROON-Perioden (Einser-Schritte wie im Original)
        aroon_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = aroon_system.run_aroon_backtests(aroon_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            aroon_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 AROON-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {aroon_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim AROON-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()