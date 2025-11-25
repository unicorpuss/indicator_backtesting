#!/usr/bin/env python3
"""
VIDYA (Variable Index Dynamic Average) BACKTESTING SYSTEM
==========================================================

Backtesting-System für VIDYA Indikator mit:
- Single-Parameter Analyse (Length)
- VIDYA steigend: Long-Position | VIDYA fallend: Cash-Position
- Optimierung über verschiedene Length-Kalibrierungen (5-150)
- Umfassende Performance-Analyse
- Integration mit backtesting_base.py für Code-Wiederverwendung

VIDYA (Variable Index Dynamic Average):
- Adaptiver Moving Average basierend auf Volatilität
- Verwendet Ratio von kurz-/langfristiger Standardabweichung
- Passt Glättung dynamisch an Marktbedingungen an
- Steigend = Bullisches Signal (Long)
- Fallend = Bearisches Signal (Cash)

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

class VIDYABacktestingSystem(BaseBacktestingSystem):
    """
    VIDYA Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "VIDYA", **kwargs)
        self.indicator_name = "VIDYA"
        self.strategy_description = "VIDYA steigend: Long-Position | VIDYA fallend: Cash-Position"
        self.threshold = None  # Kein fester Schwellenwert, nur Richtung
    
    def calculate_vidya_signals(self, data: pd.DataFrame, vidya_length: int = 20) -> pd.DataFrame:
        """
        Berechnet VIDYA-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            vidya_length: VIDYA-Periode (auch für hist_length verwendet)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        # VIDYA benötigt mindestens hist_length Perioden
        hist_length = vidya_length * 2  # Längere Periode für Volatilitätsvergleich
        min_length = hist_length + 10
        
        if len(data) < min_length:
            return pd.DataFrame()
        
        try:
            source = data['close'].values
            
            # Berechne Standardabweichungen
            stdev_short = ta.STDDEV(source, timeperiod=vidya_length, nbdev=1)
            stdev_long = ta.STDDEV(source, timeperiod=hist_length, nbdev=1)
            
            # Berechne k (Volatilitäts-Ratio)
            k = np.where(stdev_long != 0, stdev_short / stdev_long, 0)
            
            # Smoothing Constant
            sc = 2 / (vidya_length + 1)
            
            # Berechne VIDYA rekursiv
            vidya = np.full(len(source), np.nan)
            first_valid_idx = hist_length
            
            if first_valid_idx < len(source):
                vidya[first_valid_idx] = source[first_valid_idx]
            
            for i in range(first_valid_idx + 1, len(source)):
                if not np.isnan(vidya[i-1]) and not np.isnan(k[i]):
                    alpha = k[i] * sc
                    vidya[i] = alpha * source[i] + (1 - alpha) * vidya[i-1]
                else:
                    vidya[i] = source[i] if np.isnan(vidya[i-1]) else vidya[i-1]
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['vidya'] = vidya
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # VIDYA Trendfolge-Signale (steigend = Long, fallend = Cash)
            signals_df['position'] = np.where(
                signals_df['vidya'] > signals_df['vidya'].shift(1), 1, 0
            )
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei VIDYA-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_vidya_backtests(self, vidya_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt VIDYA-Backtests über verschiedene Perioden durch
        
        Args:
            vidya_range: Range der VIDYA-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=vidya_range,
            length_param_name='vidya_length',
            calculate_signals_func=self.calculate_vidya_signals,
            indicator_name='VIDYA'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden VIDYA-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode
        strategy_description = "VIDYA Long-Only Strategy (Steigend = Long, Fallend = Cash)"
        super().generate_comprehensive_report(results_df, 'vidya_length', strategy_description)
        
        # Zusätzliche VIDYA-spezifische Analyse
        print(f"\n📋 Erstelle VIDYA-spezifische Analyse...")
        self.generate_vidya_specific_analysis(results_df)
    
    def generate_vidya_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert VIDYA-spezifische Analyse
        Fokussiert auf die besten VIDYA-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für VIDYA-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle VIDYA-spezifische Analyse...")
        
        # Finde beste VIDYA-Kalibrierungen
        best_vidya_calibration = self.find_best_average_calibration(results_df, 'vidya_length')
        
        if best_vidya_calibration:
            # Erstelle zusätzlichen VIDYA-Bericht
            vidya_report_lines = []
            vidya_report_lines.append("=" * 80)
            vidya_report_lines.append("🎯 VIDYA-SPEZIFISCHE ANALYSE")
            vidya_report_lines.append("=" * 80)
            vidya_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            vidya_report_lines.append("")
            
            # Beste kombinierte VIDYA-Länge
            if 'best_vidya_combined' in best_vidya_calibration:
                best_len = best_vidya_calibration['best_vidya_combined']
                avg_sharpe = best_vidya_calibration.get('combined_vidya_sharpe_ratio', 0)
                avg_return = best_vidya_calibration.get('combined_vidya_total_return', 0)
                avg_dd = best_vidya_calibration.get('combined_vidya_max_drawdown', 0)
                avg_sortino = best_vidya_calibration.get('combined_vidya_sortino_ratio', 0)
                avg_win_rate = best_vidya_calibration.get('combined_vidya_win_rate', 0)
                combined_score = best_vidya_calibration.get('avg_combined_score', 0)
                
                vidya_report_lines.append(f"🥇 Beste Durchschnittliche VIDYA-Länge (Kombiniert): VIDYA-{best_len}")
                vidya_report_lines.append(f"   📊 Avg Sharpe: {avg_sharpe:.3f} | Avg Return: {avg_return:.1%} | Avg DD: {avg_dd:.1%}")
                vidya_report_lines.append(f"   📈 Avg Sortino: {avg_sortino:.3f} | Avg Win Rate: {avg_win_rate:.1%} | Score: {combined_score:.3f}")
                vidya_report_lines.append("")
            
            # Top VIDYAs für einzelne Metriken
            vidya_report_lines.append("📈 Beste Durchschnitts-VIDYAs nach Metriken:")
            
            metric_keys = [
                ('best_vidya_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_vidya_total_return', 'avg_total_return', 'Total Return'),
                ('best_vidya_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_vidya_calibration:
                    best_len = best_vidya_calibration[best_key]
                    avg_value = best_vidya_calibration.get(avg_key, 0)
                    if 'return' in avg_key or 'drawdown' in avg_key or 'win_rate' in avg_key:
                        vidya_report_lines.append(f"   • {metric_name}: VIDYA-{best_len} (Ø {avg_value:.1%})")
                    else:
                        vidya_report_lines.append(f"   • {metric_name}: VIDYA-{best_len} (Ø {avg_value:.3f})")
            
            # VIDYA-spezifische Empfehlungen
            vidya_report_lines.append(f"\n💡 VIDYA STRATEGIE EMPFEHLUNGEN")
            vidya_report_lines.append("-" * 60)
            vidya_report_lines.append("📋 VIDYA LONG-ONLY STRATEGIE INSIGHTS:")
            vidya_report_lines.append(f"   • Long-Position wenn VIDYA steigend (adaptiver Aufwärtstrend)")
            vidya_report_lines.append(f"   • Cash-Position wenn VIDYA fallend (adaptiver Abwärtstrend)")
            vidya_report_lines.append(f"   • VIDYA passt sich automatisch an Volatilität an")
            vidya_report_lines.append(f"   • Volatilitäts-Ratio (k) steuert Glättungsgrad dynamisch")
            vidya_report_lines.append(f"   • Kürzere VIDYA-Perioden: Höhere Sensitivität, mehr Trades")
            vidya_report_lines.append(f"   • Längere VIDYA-Perioden: Stabilere Signale, weniger Whipsaws")
            vidya_report_lines.append(f"   • Besonders effektiv in trendenden Märkten mit variabler Volatilität")
            
            vidya_report_lines.append("")
            vidya_report_lines.append("=" * 80)
            
            # Speichere VIDYA-spezifischen Bericht
            vidya_report_text = "\n".join(vidya_report_lines)
            vidya_report_path = os.path.join(self.results_folder, 'vidya_specific_analysis.txt')
            
            with open(vidya_report_path, 'w', encoding='utf-8') as f:
                f.write(vidya_report_text)
            
            print(f"📄 VIDYA-spezifische Analyse gespeichert: {vidya_report_path}")

def main():
    """
    Hauptfunktion für VIDYA Backtesting System
    """
    print("🚀 VIDYA BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (VIDYA steigend)")
    print("   • VIDYA-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • VIDYA: Adaptiver MA mit Volatilitäts-Anpassung")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe VIDYA Backtesting System aus
        vidya_system = VIDYABacktestingSystem(max_assets=20)
        
        # Teste verschiedene VIDYA-Perioden
        vidya_range = range(5, 151)
        
        # Führe Backtests durch
        results_df = vidya_system.run_vidya_backtests(vidya_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            vidya_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 VIDYA-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {vidya_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim VIDYA-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
