#!/usr/bin/env python3
"""
HULL SUITE BACKTESTING SYSTEM
==============================

Backtesting-System für Hull Suite Indikator mit:
- Single-Parameter Analyse (Length mit Multiplier=1.0)
- MHULL > SHULL: Long-Position | MHULL <= SHULL: Cash-Position
- Optimierung über verschiedene Length-Kalibrierungen (5-150)
- Umfassende Performance-Analyse
- Integration mit backtesting_base.py für Code-Wiederverwendung

Hull Suite Indicator:
- Verwendet Hull Moving Average (HMA) mit Shift-Vergleich
- MHULL = HMA[0] (aktueller Wert)
- SHULL = HMA[2] (verschobener Wert)
- Signal: MHULL > SHULL = Long (bullish)
- Signal: MHULL < SHULL = Cash (bearish)

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

class HullSuiteBacktestingSystem(BaseBacktestingSystem):
    """
    Hull Suite Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "HULLSUITE", **kwargs)
        self.indicator_name = "HULLSUITE"
        self.strategy_description = "MHULL > SHULL: Long-Position | MHULL <= SHULL: Cash-Position"
        self.threshold = None
    
    def calculate_hma(self, source: np.ndarray, period: int) -> np.ndarray:
        """
        Berechnet Hull Moving Average (HMA)
        HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        """
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        
        wma_half = ta.WMA(source, timeperiod=half_length)
        wma_full = ta.WMA(source, timeperiod=period)
        
        raw_hma = 2 * wma_half - wma_full
        hma = ta.WMA(raw_hma, timeperiod=sqrt_length)
        
        return hma
    
    def calculate_hullsuite_signals(self, data: pd.DataFrame, hull_length: int = 55) -> pd.DataFrame:
        """
        Berechnet Hull Suite-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            hull_length: Hull-Periode (mit Multiplier=1.0)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        min_length = hull_length + 10
        if len(data) < min_length:
            return pd.DataFrame()
        
        try:
            source = data['close'].values
            
            # Berechne HMA (mit Multiplier=1.0)
            length_mult = 1.0
            len_adjusted = int(hull_length * length_mult)
            
            hull = self.calculate_hma(source, len_adjusted)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['hull'] = hull
            
            # MHULL = HMA[0] (aktueller Wert)
            signals_df['mhull'] = signals_df['hull']
            
            # SHULL = HMA[2] (verschobener Wert um 2 Perioden)
            signals_df['shull'] = signals_df['hull'].shift(2)
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Hull Suite Signale (MHULL > SHULL = Long, MHULL <= SHULL = Cash)
            signals_df['position'] = np.where(
                signals_df['mhull'] > signals_df['shull'], 1, 0
            )
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei Hull Suite-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_hullsuite_backtests(self, hull_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt Hull Suite-Backtests über verschiedene Perioden durch
        
        Args:
            hull_range: Range der Hull-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=hull_range,
            length_param_name='hull_length',
            calculate_signals_func=self.calculate_hullsuite_signals,
            indicator_name='HULLSUITE'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden Hull Suite-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode
        strategy_description = "Hull Suite Long-Only Strategy (MHULL > SHULL = Long)"
        super().generate_comprehensive_report(results_df, 'hull_length', strategy_description)
        
        # Zusätzliche HULLSUITE-spezifische Analyse
        print(f"\n📋 Erstelle HULLSUITE-spezifische Analyse...")
        self.generate_hullsuite_specific_analysis(results_df)
    
    def generate_hullsuite_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert HULLSUITE-spezifische Analyse
        Fokussiert auf die besten Hull-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für HULLSUITE-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle HULLSUITE-spezifische Analyse...")
        
        # Finde beste HULLSUITE-Kalibrierungen
        best_hull_calibration = self.find_best_average_calibration(results_df, 'hull_length')
        
        if best_hull_calibration:
            # Erstelle zusätzlichen HULLSUITE-Bericht
            hull_report_lines = []
            hull_report_lines.append("=" * 80)
            hull_report_lines.append("🎯 HULLSUITE-SPEZIFISCHE ANALYSE")
            hull_report_lines.append("=" * 80)
            hull_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            hull_report_lines.append("")
            
            # Beste kombinierte Hull-Länge
            if 'best_hull_combined' in best_hull_calibration:
                best_len = best_hull_calibration['best_hull_combined']
                avg_sharpe = best_hull_calibration.get('combined_hull_sharpe_ratio', 0)
                avg_return = best_hull_calibration.get('combined_hull_total_return', 0)
                avg_dd = best_hull_calibration.get('combined_hull_max_drawdown', 0)
                avg_sortino = best_hull_calibration.get('combined_hull_sortino_ratio', 0)
                avg_win_rate = best_hull_calibration.get('combined_hull_win_rate', 0)
                combined_score = best_hull_calibration.get('avg_combined_score', 0)
                
                hull_report_lines.append(f"🥇 Beste Durchschnittliche Hull-Länge (Kombiniert): HULL-{best_len}")
                hull_report_lines.append(f"   📊 Avg Sharpe: {avg_sharpe:.3f} | Avg Return: {avg_return:.1%} | Avg DD: {avg_dd:.1%}")
                hull_report_lines.append(f"   📈 Avg Sortino: {avg_sortino:.3f} | Avg Win Rate: {avg_win_rate:.1%} | Score: {combined_score:.3f}")
                hull_report_lines.append("")
            
            # Top Hulls für einzelne Metriken
            hull_report_lines.append("📈 Beste Durchschnitts-Hulls nach Metriken:")
            
            metric_keys = [
                ('best_hull_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_hull_total_return', 'avg_total_return', 'Total Return'),
                ('best_hull_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_hull_calibration:
                    best_len = best_hull_calibration[best_key]
                    avg_value = best_hull_calibration.get(avg_key, 0)
                    if 'return' in avg_key or 'drawdown' in avg_key or 'win_rate' in avg_key:
                        hull_report_lines.append(f"   • {metric_name}: HULL-{best_len} (Ø {avg_value:.1%})")
                    else:
                        hull_report_lines.append(f"   • {metric_name}: HULL-{best_len} (Ø {avg_value:.3f})")
            
            # HULLSUITE-spezifische Empfehlungen
            hull_report_lines.append(f"\n💡 HULLSUITE STRATEGIE EMPFEHLUNGEN")
            hull_report_lines.append("-" * 60)
            hull_report_lines.append("📋 HULLSUITE LONG-ONLY STRATEGIE INSIGHTS:")
            hull_report_lines.append(f"   • Long-Position wenn MHULL > SHULL (aktueller HMA über verzögertem HMA)")
            hull_report_lines.append(f"   • Cash-Position wenn MHULL <= SHULL (aktueller HMA unter verzögertem HMA)")
            hull_report_lines.append(f"   • Hull Moving Average reduziert Lag gegenüber SMA/EMA")
            hull_report_lines.append(f"   • MHULL = HMA[0], SHULL = HMA[2] (2-Perioden Shift)")
            hull_report_lines.append(f"   • HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))")
            hull_report_lines.append(f"   • Kürzere Hull-Perioden: Schnellere Trend-Erkennung, mehr Whipsaws")
            hull_report_lines.append(f"   • Längere Hull-Perioden: Stabilere Trends, weniger Fehlsignale")
            hull_report_lines.append(f"   • Besonders effektiv in klaren Trendphasen mit geringem Noise")
            
            hull_report_lines.append("")
            hull_report_lines.append("=" * 80)
            
            # Speichere HULLSUITE-spezifischen Bericht
            hull_report_text = "\n".join(hull_report_lines)
            hull_report_path = os.path.join(self.results_folder, 'hullsuite_specific_analysis.txt')
            
            with open(hull_report_path, 'w', encoding='utf-8') as f:
                f.write(hull_report_text)
            
            print(f"📄 HULLSUITE-spezifische Analyse gespeichert: {hull_report_path}")

def main():
    """
    Hauptfunktion für Hull Suite Backtesting System
    """
    print("🚀 HULL SUITE BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (MHULL > SHULL)")
    print("   • Hull-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Hull Suite: HMA mit Shift-Vergleich")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe Hull Suite Backtesting System aus
        hull_system = HullSuiteBacktestingSystem(max_assets=20)
        
        # Teste verschiedene Hull-Perioden
        hull_range = range(5, 151)
        
        # Führe Backtests durch
        results_df = hull_system.run_hullsuite_backtests(hull_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            hull_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 Hull Suite-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {hull_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim Hull Suite-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
