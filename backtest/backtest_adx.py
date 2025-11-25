"""
ADX (Average Directional Index) Backtesting System (Optimiert mit Base Class)

Dieses System testet ADX-basierte Trading-Strategien über verschiedene Assets und Perioden:
- ADX-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: ADX > 25 = Long Position, sonst Cash Position
- Verwendet gemeinsame Funktionen aus backtesting_base.py

ADX-Indikator:
- Trend-Stärke-Indikator entwickelt von J. Welles Wilder
- Berechnung: Glättung der Differenz zwischen +DI und -DI
- Werte zwischen 0 und 100
- 0-25: Schwacher/kein Trend (seitwärts)
- 25-50: Starker Trend
- 50-75: Sehr starker Trend
- 75-100: Extrem starker Trend
- Long-Signal: ADX > 25 (starker Trend, unabhängig von Richtung)
- Cash-Signal: ADX <= 25 (schwacher Trend, seitwärts bewegung)
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

class ADXBacktestingSystem(BaseBacktestingSystem):
    """
    ADX-spezifisches Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "ADX", **kwargs)
        self.indicator_name = "ADX"
        self.strategy_description = "ADX > 25: Long-Position | ADX <= 25: Cash-Position"
        self.threshold = 25.0  # ADX Trend-Stärke Schwellenwert (25)
    
    def calculate_adx_signals(self, data: pd.DataFrame, adx_length: int = 14) -> pd.DataFrame:
        """
        Berechnet ADX-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            adx_length: ADX-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < adx_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne ADX mit talib
            adx = ta.ADX(data['high'].astype(float).values, 
                        data['low'].astype(float).values, 
                        data['close'].astype(float).values, 
                        timeperiod=adx_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['adx'] = adx
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # ADX Long-Only Signale (ADX > 25 = Long, ADX <= 25 = Cash)
            # 25 ist der typische Schwellenwert für starke Trends
            signals_df['position'] = np.where(signals_df['adx'] > 25, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei ADX-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_adx_backtests(self, adx_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt ADX-Backtests über verschiedene Perioden durch
        
        Args:
            adx_range: Range der ADX-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=adx_range,
            length_param_name='adx_length',
            calculate_signals_func=self.calculate_adx_signals,
            indicator_name='ADX'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden ADX-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit ADX-spezifischen Parametern
        strategy_description = "ADX Long-Only Strategy (ADX > 25 = Long, ADX <= 25 = Cash)"
        super().generate_comprehensive_report(results_df, 'adx_length', strategy_description)
        
        # Zusätzliche ADX-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle ADX-spezifische Analyse...")
        
        # Verwende die ADX-spezifische Analyse-Funktion
        self.generate_adx_specific_analysis(results_df)
    
    def generate_adx_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert ADX-spezifische Analyse
        Fokussiert auf die besten ADX-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für ADX-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle ADX-spezifische Analyse...")
        
        # Finde beste ADX-Kalibrierungen (nutzt gemeinsame Funktion)
        best_adx_calibration = self.find_best_average_calibration(results_df, 'adx_length')
        
        if best_adx_calibration:
            # Erstelle zusätzlichen ADX-Bericht
            adx_report_lines = []
            adx_report_lines.append("=" * 80)
            adx_report_lines.append("🎯 ADX-SPEZIFISCHE ANALYSE")
            adx_report_lines.append("=" * 80)
            adx_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            adx_report_lines.append("")
            
            # Beste kombinierte ADX-Länge
            if 'best_adx_combined' in best_adx_calibration:
                adx_report_lines.append(
                    f"🥇 Beste Durchschnittliche ADX-Länge (Kombiniert): ADX-{best_adx_calibration['best_adx_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte ADX
                combined_keys = ['combined_adx_sharpe_ratio', 'combined_adx_total_return', 'combined_adx_max_drawdown']
                if all(key in best_adx_calibration for key in combined_keys):
                    adx_report_lines.append(
                        f"   📊 Avg Sharpe: {best_adx_calibration['combined_adx_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_adx_calibration['combined_adx_total_return']:.1%} | "
                        f"Avg DD: {best_adx_calibration['combined_adx_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_adx_sortino_ratio' in best_adx_calibration and 'combined_adx_win_rate' in best_adx_calibration:
                    adx_report_lines.append(
                        f"   📈 Avg Sortino: {best_adx_calibration['combined_adx_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_adx_calibration['combined_adx_win_rate']:.1%} | "
                        f"Score: {best_adx_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top ADXs für einzelne Metriken
            adx_report_lines.append("")
            adx_report_lines.append("📈 Beste Durchschnitts-ADXs nach Metriken:")
            
            metric_keys = [
                ('best_adx_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_adx_total_return', 'avg_total_return', 'Total Return'),
                ('best_adx_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_adx_calibration and avg_key in best_adx_calibration:
                    avg_val = best_adx_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    adx_report_lines.append(f"   • {metric_name}: ADX-{best_adx_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # ADX-spezifische Empfehlungen
            adx_report_lines.append(f"\n💡 ADX STRATEGIE EMPFEHLUNGEN")
            adx_report_lines.append("-" * 60)
            adx_report_lines.append("📋 ADX LONG-ONLY STRATEGIE INSIGHTS:")
            adx_report_lines.append(f"   • Long-Position wenn ADX > 25 (starker Trend)")
            adx_report_lines.append(f"   • Cash-Position wenn ADX <= 25 (schwacher Trend, Seitwärtsbewegung)")
            adx_report_lines.append(f"   • ADX misst Trend-STÄRKE, nicht Trend-RICHTUNG")
            adx_report_lines.append(f"   • 0-25: Schwacher/kein Trend (seitwärts)")
            adx_report_lines.append(f"   • 25-50: Starker Trend")
            adx_report_lines.append(f"   • 50-75: Sehr starker Trend")
            adx_report_lines.append(f"   • 75-100: Extrem starker Trend")
            adx_report_lines.append(f"   • Kürzere ADX-Perioden: Mehr Trades, höhere Sensitivität")
            adx_report_lines.append(f"   • Längere ADX-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            adx_report_lines.append(f"\n🎯 ADX PARAMETER-OPTIMIERUNG")
            adx_report_lines.append("-" * 60)
            adx_report_lines.append(f"📌 OPTIMALE ADX-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_adx_combined' in best_adx_calibration:
                combined_score = best_adx_calibration.get('avg_combined_score', 0.0)
                adx_report_lines.append(f"   🥇 Beste Gesamtperformance: ADX-{best_adx_calibration['best_adx_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_adx_calibration and avg_key in best_adx_calibration:
                    avg_val = best_adx_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    adx_report_lines.append(f"   📈 Höchste {metric_name}: ADX-{best_adx_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_adx_max_drawdown' in best_adx_calibration and 'avg_max_drawdown' in best_adx_calibration:
                avg_dd = best_adx_calibration['avg_max_drawdown']
                adx_report_lines.append(f"   🛡️ Niedrigster Drawdown: ADX-{best_adx_calibration['best_adx_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            adx_report_lines.append(f"\n" + "=" * 68)
            adx_report_lines.append(f"🏁 ADX-ANALYSE ABGESCHLOSSEN")
            adx_report_lines.append("=" * 68)
            
            # Speichere ADX-spezifischen Bericht
            adx_report_text = "\n".join(adx_report_lines)
            adx_report_path = os.path.join(self.results_folder, 'adx_specific_analysis.txt')
            
            with open(adx_report_path, 'w', encoding='utf-8') as f:
                f.write(adx_report_text)
            
            print(f"📄 ADX-spezifische Analyse gespeichert: {adx_report_path}")

def main():
    """
    Hauptfunktion für ADX Backtesting System
    """
    print("🚀 ADX BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (ADX > 25)")
    print("   • ADX-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • ADX Trend-Stärke: 0-100 (25 = Schwellenwert)")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe ADX Backtesting System aus
        adx_system = ADXBacktestingSystem(max_assets=20)
        
        # Teste verschiedene ADX-Perioden (Einser-Schritte wie im Original)
        adx_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = adx_system.run_adx_backtests(adx_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            adx_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 ADX-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {adx_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim ADX-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()