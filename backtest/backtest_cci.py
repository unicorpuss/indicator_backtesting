"""
CCI Backtesting System Test (Optimiert mit Base Class)

Dieses System testet CCI-basierte Trading-Strategien über verschiedene Assets und Perioden:
- CCI-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: CCI > 0 = Long Position, CCI <= 0 = Cash Position
- Umfassende Performance-Metriken (Sharpe, Sortino, Omega, etc.)
- Verwendet gemeinsame Funktionen aus backtesting_base.py

CCI (Commodity Channel Index):
- Momentum-Indikator entwickelt von Donald Lambert
- Berechnung: (Typical Price - SMA) / (0.015 * Mean Deviation)
- Typical Price = (High + Low + Close) / 3
- Normalerweise zwischen +100 und -100, aber kann darüber/darunter gehen
- Long-Signal: CCI > 0 (bullisches Momentum über Null-Linie)
- Cash-Signal: CCI <= 0 (bearisches/neutrales Momentum unter Null-Linie)
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

class CCIBacktestingSystem(BaseBacktestingSystem):
    """
    CCI-spezifisches Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "CCI", **kwargs)
        self.indicator_name = "CCI"
        self.strategy_description = "CCI > 0: Long-Position | CCI <= 0: Cash-Position"
        self.threshold = 0.0
    
    def calculate_cci_signals(self, data: pd.DataFrame, cci_length: int = 14) -> pd.DataFrame:
        """
        Berechnet CCI-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            cci_length: CCI-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < cci_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne CCI
            cci = ta.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=cci_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['cci'] = cci
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # CCI Long-Only Signale (CCI > 0 = Long, CCI <= 0 = Cash)
            signals_df['position'] = np.where(signals_df['cci'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei CCI-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_cci_backtests(self, cci_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt CCI-Backtests über verschiedene Perioden durch
        
        Args:
            cci_range: Range der CCI-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=cci_range,
            length_param_name='cci_length',
            calculate_signals_func=self.calculate_cci_signals,
            indicator_name='CCI'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden CCI-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit CCI-spezifischen Parametern
        strategy_description = "CCI Long-Only Strategy (CCI > 0 = Long, CCI <= 0 = Cash)"
        super().generate_comprehensive_report(results_df, 'cci_length', strategy_description)
        
        # Zusätzliche CCI-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle CCI-spezifische Analyse...")
        
        # Verwende die CCI-spezifische Analyse-Funktion
        self.generate_cci_specific_analysis(results_df)
    
    def generate_cci_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert CCI-spezifische Analyse ähnlich der EMA-Analyse
        Fokussiert auf die besten CCI-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für CCI-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle CCI-spezifische Analyse...")
        
        # Finde beste CCI-Kalibrierungen (nutzt gemeinsame Funktion)
        best_cci_calibration = self.find_best_average_calibration(results_df, 'cci_length')
        
        if best_cci_calibration:
            # Erstelle zusätzlichen CCI-Bericht
            cci_report_lines = []
            cci_report_lines.append("=" * 80)
            cci_report_lines.append("🎯 CCI-SPEZIFISCHE ANALYSE")
            cci_report_lines.append("=" * 80)
            cci_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            cci_report_lines.append("")
            
            # Beste kombinierte CCI-Länge
            if 'best_cci_combined' in best_cci_calibration:
                cci_report_lines.append(
                    f"🥇 Beste Durchschnittliche CCI-Länge (Kombiniert): CCI-{best_cci_calibration['best_cci_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte CCI
                combined_keys = ['combined_cci_sharpe_ratio', 'combined_cci_total_return', 'combined_cci_max_drawdown']
                if all(key in best_cci_calibration for key in combined_keys):
                    cci_report_lines.append(
                        f"   📊 Avg Sharpe: {best_cci_calibration['combined_cci_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_cci_calibration['combined_cci_total_return']:.1%} | "
                        f"Avg DD: {best_cci_calibration['combined_cci_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_cci_sortino_ratio' in best_cci_calibration and 'combined_cci_win_rate' in best_cci_calibration:
                    cci_report_lines.append(
                        f"   📈 Avg Sortino: {best_cci_calibration['combined_cci_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_cci_calibration['combined_cci_win_rate']:.1%} | "
                        f"Score: {best_cci_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top CCIs für einzelne Metriken
            cci_report_lines.append("")
            cci_report_lines.append("📈 Beste Durchschnitts-CCIs nach Metriken:")
            
            metric_keys = [
                ('best_cci_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_cci_total_return', 'avg_total_return', 'Total Return'),
                ('best_cci_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_cci_calibration and avg_key in best_cci_calibration:
                    avg_val = best_cci_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    cci_report_lines.append(f"   • {metric_name}: CCI-{best_cci_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # CCI-spezifische Empfehlungen
            cci_report_lines.append(f"\n💡 CCI STRATEGIE EMPFEHLUNGEN")
            cci_report_lines.append("-" * 60)
            cci_report_lines.append("📋 CCI LONG-ONLY STRATEGIE INSIGHTS:")
            cci_report_lines.append(f"   • Long-Position wenn CCI > 0 (bullisches Momentum)")
            cci_report_lines.append(f"   • Cash-Position wenn CCI <= 0 (bearisches/neutrales Momentum)")
            cci_report_lines.append(f"   • CCI über +100: Starker Aufwärtstrend, überkauft")
            cci_report_lines.append(f"   • CCI unter -100: Starker Abwärtstrend, überverkauft")
            cci_report_lines.append(f"   • CCI zwischen -100 und +100: Normale Handelsspanne")
            cci_report_lines.append(f"   • Kürzere CCI-Perioden: Mehr Trades, höhere Sensitivität")
            cci_report_lines.append(f"   • Längere CCI-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            cci_report_lines.append(f"\n🎯 CCI PARAMETER-OPTIMIERUNG")
            cci_report_lines.append("-" * 60)
            cci_report_lines.append(f"📌 OPTIMALE CCI-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_cci_combined' in best_cci_calibration:
                combined_score = best_cci_calibration.get('avg_combined_score', 0.0)
                cci_report_lines.append(f"   🥇 Beste Gesamtperformance: CCI-{best_cci_calibration['best_cci_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_cci_calibration and avg_key in best_cci_calibration:
                    avg_val = best_cci_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    cci_report_lines.append(f"   📈 Höchste {metric_name}: CCI-{best_cci_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_cci_max_drawdown' in best_cci_calibration and 'avg_max_drawdown' in best_cci_calibration:
                avg_dd = best_cci_calibration['avg_max_drawdown']
                cci_report_lines.append(f"   🛡️ Niedrigster Drawdown: CCI-{best_cci_calibration['best_cci_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            cci_report_lines.append(f"\n" + "=" * 68)
            cci_report_lines.append(f"🏁 CCI-ANALYSE ABGESCHLOSSEN")
            cci_report_lines.append("=" * 68)
            
            # Speichere CCI-spezifischen Bericht
            cci_report_text = "\n".join(cci_report_lines)
            cci_report_path = os.path.join(self.results_folder, 'cci_specific_analysis.txt')
            
            with open(cci_report_path, 'w', encoding='utf-8') as f:
                f.write(cci_report_text)
            
            print(f"📄 CCI-spezifische Analyse gespeichert: {cci_report_path}")

def main():
    """
    Hauptfunktion für CCI Backtesting System
    """
    print("🚀 CCI BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (CCI > 0)")
    print("   • CCI-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe CCI Backtesting System aus
        cci_system = CCIBacktestingSystem(max_assets=20)
        
        # Teste verschiedene CCI-Perioden (Einser-Schritte wie im Original)
        cci_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = cci_system.run_cci_backtests(cci_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            cci_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 CCI-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {cci_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim CCI-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()