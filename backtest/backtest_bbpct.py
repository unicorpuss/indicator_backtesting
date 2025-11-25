#!/usr/bin/env python3
"""
BBPCT (Bollinger Bands Percentile) BACKTESTING SYSTEM
======================================================

Backtesting-System für BBPct Indikator mit:
- Single-Parameter Analyse (Length)
- Position > 50: Long-Position | Position <= 50: Cash-Position
- Optimierung über verschiedene Length-Kalibrierungen (5-150)
- Umfassende Performance-Analyse
- Integration mit backtesting_base.py für Code-Wiederverwendung

BBPct (Bollinger Bands Percentile):
- Misst Position des Preises zwischen Bollinger Bands (0-100%)
- Basis: Simple Moving Average (SMA)
- Bands: SMA ± (Factor * StdDev)
- Position > 50% = Preis über Mittellinie (Long)
- Position <= 50% = Preis unter Mittellinie (Cash)
- Factor und Lookback sind fixiert (2.0 und 750)

Autor: Enhanced Backtesting Framework
Datum: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import talib as ta
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Backtesting-Funktionen
from _backtesting_base_ import BaseBacktestingSystem

class BBPctBacktestingSystem(BaseBacktestingSystem):
    """
    BBPct Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "BBPCT", **kwargs)
        self.indicator_name = "BBPCT"
        self.strategy_description = "Position > 50: Long-Position | Position <= 50: Cash-Position"
        self.threshold = 50.0  # Mittellinie der Position zwischen Bands
    
    def calculate_bbpct_signals(self, data: pd.DataFrame, bbpct_length: int = 20) -> pd.DataFrame:
        """
        Berechnet BBPct-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            bbpct_length: BBPct-Periode (Length für SMA und StdDev)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        # Feste Parameter
        factor = 2.0
        lookback = 750
        
        min_length = max(bbpct_length, lookback) + 10
        if len(data) < min_length:
            return pd.DataFrame()
        
        try:
            source = data['close'].values
            
            # Berechne Basis (SMA)
            basis = ta.SMA(source, timeperiod=bbpct_length)
            
            # Berechne Standard Deviation
            dev = factor * ta.STDDEV(source, timeperiod=bbpct_length, nbdev=1)
            
            # Berechne Upper und Lower Bands
            upper = basis + dev
            lower = basis - dev
            
            # Berechne Position zwischen Bands (0-100)
            position_between_bands = 100 * (source - lower) / (upper - lower)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['basis'] = basis
            signals_df['upper'] = upper
            signals_df['lower'] = lower
            signals_df['position_between_bands'] = position_between_bands
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # BBPct Signale (Position > 50 = Long, Position <= 50 = Cash)
            # 50 = Mittellinie zwischen den Bands
            signals_df['position'] = np.where(
                signals_df['position_between_bands'] > 50, 1, 0
            )
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei BBPct-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_bbpct_backtests(self, bbpct_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt BBPct-Backtests über verschiedene Perioden durch
        
        Args:
            bbpct_range: Range der BBPct-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=bbpct_range,
            length_param_name='bbpct_length',
            calculate_signals_func=self.calculate_bbpct_signals,
            indicator_name='BBPCT'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden BBPct-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode
        strategy_description = "BBPct Long-Only Strategy (Position > 50 = Long, <= 50 = Cash)"
        super().generate_comprehensive_report(results_df, 'bbpct_length', strategy_description)
        
        # Zusätzliche BBPCT-spezifische Analyse
        print(f"\n📋 Erstelle BBPCT-spezifische Analyse...")
        self.generate_bbpct_specific_analysis(results_df)
    
    def generate_bbpct_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert BBPCT-spezifische Analyse
        Fokussiert auf die besten BBPct-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für BBPCT-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle BBPCT-spezifische Analyse...")
        
        # Finde beste BBPCT-Kalibrierungen
        best_bbpct_calibration = self.find_best_average_calibration(results_df, 'bbpct_length')
        
        if best_bbpct_calibration:
            # Erstelle zusätzlichen BBPCT-Bericht
            bbpct_report_lines = []
            bbpct_report_lines.append("=" * 80)
            bbpct_report_lines.append("🎯 BBPCT-SPEZIFISCHE ANALYSE")
            bbpct_report_lines.append("=" * 80)
            bbpct_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            bbpct_report_lines.append("")
            
            # Beste kombinierte BBPct-Länge
            if 'best_bbpct_combined' in best_bbpct_calibration:
                best_len = best_bbpct_calibration['best_bbpct_combined']
                avg_sharpe = best_bbpct_calibration.get('combined_bbpct_sharpe_ratio', 0)
                avg_return = best_bbpct_calibration.get('combined_bbpct_total_return', 0)
                avg_dd = best_bbpct_calibration.get('combined_bbpct_max_drawdown', 0)
                avg_sortino = best_bbpct_calibration.get('combined_bbpct_sortino_ratio', 0)
                avg_win_rate = best_bbpct_calibration.get('combined_bbpct_win_rate', 0)
                combined_score = best_bbpct_calibration.get('avg_combined_score', 0)
                
                bbpct_report_lines.append(f"🥇 Beste Durchschnittliche BBPct-Länge (Kombiniert): BBPCT-{best_len}")
                bbpct_report_lines.append(f"   📊 Avg Sharpe: {avg_sharpe:.3f} | Avg Return: {avg_return:.1%} | Avg DD: {avg_dd:.1%}")
                bbpct_report_lines.append(f"   📈 Avg Sortino: {avg_sortino:.3f} | Avg Win Rate: {avg_win_rate:.1%} | Score: {combined_score:.3f}")
                bbpct_report_lines.append("")
            
            # Top BBPcts für einzelne Metriken
            bbpct_report_lines.append("📈 Beste Durchschnitts-BBPcts nach Metriken:")
            
            metric_keys = [
                ('best_bbpct_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_bbpct_total_return', 'avg_total_return', 'Total Return'),
                ('best_bbpct_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_bbpct_calibration:
                    best_len = best_bbpct_calibration[best_key]
                    avg_value = best_bbpct_calibration.get(avg_key, 0)
                    if 'return' in avg_key or 'drawdown' in avg_key or 'win_rate' in avg_key:
                        bbpct_report_lines.append(f"   • {metric_name}: BBPCT-{best_len} (Ø {avg_value:.1%})")
                    else:
                        bbpct_report_lines.append(f"   • {metric_name}: BBPCT-{best_len} (Ø {avg_value:.3f})")
            
            # BBPCT-spezifische Empfehlungen
            bbpct_report_lines.append(f"\n💡 BBPCT STRATEGIE EMPFEHLUNGEN")
            bbpct_report_lines.append("-" * 60)
            bbpct_report_lines.append("📋 BBPCT LONG-ONLY STRATEGIE INSIGHTS:")
            bbpct_report_lines.append(f"   • Long-Position wenn Position > 50% (Preis über Mittellinie)")
            bbpct_report_lines.append(f"   • Cash-Position wenn Position <= 50% (Preis unter Mittellinie)")
            bbpct_report_lines.append(f"   • BBPct = 100 * (Price - Lower) / (Upper - Lower)")
            bbpct_report_lines.append(f"   • Bollinger Bands: Basis ± (2.0 * StdDev)")
            bbpct_report_lines.append(f"   • Position 0% = unteres Band, 50% = Mittellinie, 100% = oberes Band")
            bbpct_report_lines.append(f"   • Kürzere BB-Perioden: Engere Bands, mehr Signale")
            bbpct_report_lines.append(f"   • Längere BB-Perioden: Weitere Bands, stabilere Signale")
            bbpct_report_lines.append(f"   • Besonders effektiv in Mean-Reverting und Trend-Following Märkten")
            
            bbpct_report_lines.append("")
            bbpct_report_lines.append("=" * 80)
            
            # Speichere BBPCT-spezifischen Bericht
            bbpct_report_text = "\n".join(bbpct_report_lines)
            bbpct_report_path = os.path.join(self.results_folder, 'bbpct_specific_analysis.txt')
            
            with open(bbpct_report_path, 'w', encoding='utf-8') as f:
                f.write(bbpct_report_text)
            
            print(f"📄 BBPCT-spezifische Analyse gespeichert: {bbpct_report_path}")

def main():
    """
    Hauptfunktion für BBPct Backtesting System
    """
    print("🚀 BBPCT BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (Position > 50%)")
    print("   • BBPct-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • BBPct: Bollinger Bands Position Percentile")
    print("   • Factor: 2.0 (fixiert)")
    print("   • Lookback: 750 (fixiert)")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe BBPct Backtesting System aus
        bbpct_system = BBPctBacktestingSystem(max_assets=20)
        
        # Teste verschiedene BBPct-Perioden
        bbpct_range = range(5, 151)
        
        # Führe Backtests durch
        results_df = bbpct_system.run_bbpct_backtests(bbpct_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            bbpct_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 BBPct-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {bbpct_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim BBPct-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
