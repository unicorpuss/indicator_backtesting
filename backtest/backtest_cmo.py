#!/usr/bin/env python3
"""
CMO (Chande Momentum Oscillator) BACKTESTING SYSTEM - OPTIMIZED VERSION
========================================================================

Backtesting-System für Chande Momentum Oscillator (CMO) Indikator mit:
- Momentum-basierte Trading-Strategie
- CMO > 0: Long-Position | CMO <= 0: Cash-Position (0 = Mittellinie)
- Umfassende Performance-Analyse
- Automatische Heatmap- und Line Plot-Generierung
- Integration mit backtesting_base.py für Code-Wiederverwendung

CMO (Chande Momentum Oscillator):
- Entwickelt von Tushar Chande
- Berechnet Momentum basierend auf Summe der Aufwärts- vs. Abwärtsbewegungen
- Range: -100 bis +100
- 0 = Mittellinie (neutral)
- > 0 = Bullish Momentum
- < 0 = Bearish Momentum

Autor: Optimized Backtesting Framework
Datum: 2024
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Tuple, Optional
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Base-Klasse
from _backtesting_base_ import BaseBacktestingSystem

class CMOBacktestingSystem(BaseBacktestingSystem):
    """
    Chande Momentum Oscillator (CMO) Backtesting System
    
    Der CMO ist ein momentum-basierter Oszillator von Tushar Chande.
    Strategie: CMO > 0 = Long-Position, CMO <= 0 = Cash-Position (0 = Mittellinie)
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "CMO", **kwargs)
        self.indicator_name = "CMO"
        self.strategy_description = "CMO > 0: Long-Position | CMO <= 0: Cash-Position"
        self.threshold = 0.0

    def calculate_cmo_signals(self, data: pd.DataFrame, cmo_length: int = 14) -> pd.DataFrame:
        """
        Berechnet CMO-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            cmo_length: CMO-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < cmo_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne CMO mit talib
            cmo = ta.CMO(data['close'].values, timeperiod=cmo_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['cmo'] = cmo
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # CMO Long-Only Signale (CMO > 0 = Long, CMO <= 0 = Cash)
            signals_df['position'] = np.where(signals_df['cmo'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei CMO-Berechnung: {e}")
            return pd.DataFrame()

    def run_cmo_backtests(self, cmo_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt CMO-Backtests über verschiedene Perioden durch
        
        Args:
            cmo_range: Range der CMO-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=cmo_range,
            length_param_name='cmo_length',
            calculate_signals_func=self.calculate_cmo_signals,
            indicator_name='CMO'
        )

    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden CMO-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit CMO-spezifischen Parametern
        strategy_description = "CMO Long-Only Strategy (CMO > 0 = Long, CMO <= 0 = Cash)"
        super().generate_comprehensive_report(results_df, 'cmo_length', strategy_description)
        
        # Zusätzliche CMO-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle CMO-spezifische Analyse...")
        
        # Verwende die CMO-spezifische Analyse-Funktion
        self.generate_cmo_specific_analysis(results_df)
    
    def generate_cmo_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert CMO-spezifische Analyse ähnlich der CCI-Analyse
        Fokussiert auf die besten CMO-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für CMO-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle CMO-spezifische Analyse...")
        
        # Finde beste CMO-Kalibrierungen (nutzt gemeinsame Funktion)
        best_cmo_calibration = self.find_best_average_calibration(results_df, 'cmo_length')
        
        if best_cmo_calibration:
            # Erstelle zusätzlichen CMO-Bericht
            cmo_report_lines = []
            cmo_report_lines.append("=" * 80)
            cmo_report_lines.append("🎯 CMO-SPEZIFISCHE ANALYSE")
            cmo_report_lines.append("=" * 80)
            cmo_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            cmo_report_lines.append("")
            
            # Beste kombinierte CMO-Länge
            if 'best_cmo_combined' in best_cmo_calibration:
                cmo_report_lines.append(
                    f"🥇 Beste Durchschnittliche CMO-Länge (Kombiniert): CMO-{best_cmo_calibration['best_cmo_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte CMO
                combined_keys = ['combined_cmo_sharpe_ratio', 'combined_cmo_total_return', 'combined_cmo_max_drawdown']
                if all(key in best_cmo_calibration for key in combined_keys):
                    cmo_report_lines.append(
                        f"   📊 Avg Sharpe: {best_cmo_calibration['combined_cmo_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_cmo_calibration['combined_cmo_total_return']:.1%} | "
                        f"Avg DD: {best_cmo_calibration['combined_cmo_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_cmo_sortino_ratio' in best_cmo_calibration and 'combined_cmo_win_rate' in best_cmo_calibration:
                    cmo_report_lines.append(
                        f"   📈 Avg Sortino: {best_cmo_calibration['combined_cmo_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_cmo_calibration['combined_cmo_win_rate']:.1%} | "
                        f"Score: {best_cmo_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top CMOs für einzelne Metriken
            cmo_report_lines.append("")
            cmo_report_lines.append("📈 Beste Durchschnitts-CMOs nach Metriken:")
            
            metric_keys = [
                ('best_cmo_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_cmo_total_return', 'avg_total_return', 'Total Return'),
                ('best_cmo_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_cmo_calibration and avg_key in best_cmo_calibration:
                    avg_val = best_cmo_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    cmo_report_lines.append(f"   • {metric_name}: CMO-{best_cmo_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # CMO-spezifische Empfehlungen
            cmo_report_lines.append(f"\n💡 CMO STRATEGIE EMPFEHLUNGEN")
            cmo_report_lines.append("-" * 60)
            cmo_report_lines.append("📋 CMO LONG-ONLY STRATEGIE INSIGHTS:")
            cmo_report_lines.append(f"   • Long-Position wenn CMO > 0 (bullisches Momentum)")
            cmo_report_lines.append(f"   • Cash-Position wenn CMO <= 0 (bearisches Momentum)")
            cmo_report_lines.append(f"   • CMO über +50: Starker Aufwärtstrend, mögliche Überkauf")
            cmo_report_lines.append(f"   • CMO unter -50: Starker Abwärtstrend, mögliche Überverkauf")
            cmo_report_lines.append(f"   • CMO nahe 0: Neutraler/seitlicher Markt")
            cmo_report_lines.append(f"   • Kürzere CMO-Perioden: Mehr Trades, höhere Sensitivität")
            cmo_report_lines.append(f"   • Längere CMO-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            cmo_report_lines.append(f"\n🎯 CMO PARAMETER-OPTIMIERUNG")
            cmo_report_lines.append("-" * 60)
            cmo_report_lines.append(f"📌 OPTIMALE CMO-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_cmo_combined' in best_cmo_calibration:
                combined_score = best_cmo_calibration.get('avg_combined_score', 0.0)
                cmo_report_lines.append(f"   🥇 Beste Gesamtperformance: CMO-{best_cmo_calibration['best_cmo_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_cmo_calibration and avg_key in best_cmo_calibration:
                    avg_val = best_cmo_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    cmo_report_lines.append(f"   📈 Höchste {metric_name}: CMO-{best_cmo_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_cmo_max_drawdown' in best_cmo_calibration and 'avg_max_drawdown' in best_cmo_calibration:
                avg_dd = best_cmo_calibration['avg_max_drawdown']
                cmo_report_lines.append(f"   🛡️ Niedrigster Drawdown: CMO-{best_cmo_calibration['best_cmo_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            cmo_report_lines.append(f"\n" + "=" * 68)
            cmo_report_lines.append(f"🏁 CMO-ANALYSE ABGESCHLOSSEN")
            cmo_report_lines.append("=" * 68)
            
            # Speichere CMO-spezifischen Bericht
            cmo_report_text = "\n".join(cmo_report_lines)
            cmo_report_path = os.path.join(self.results_folder, 'cmo_specific_analysis.txt')
            
            with open(cmo_report_path, 'w', encoding='utf-8') as f:
                f.write(cmo_report_text)
            
            print(f"📄 CMO-spezifische Analyse gespeichert: {cmo_report_path}")

def main():
    """
    Hauptfunktion für CMO Backtesting System Demo
    """
    print("🚀 CMO BACKTESTING SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Initialisiere System
        cmo_system = CMOBacktestingSystem()
        
        # Führe Backtests durch (kleiner Range für Demo)
        results_df = cmo_system.run_cmo_backtests(cmo_range=range(10, 31))
        
        if not results_df.empty:
            print(f"✅ Demo erfolgreich: {len(results_df)} Tests durchgeführt")
            
            # Generiere umfassenden Bericht (wie andere Systeme)
            print("\n📊 Generiere umfassenden CMO-Bericht...")
            cmo_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 CMO-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {cmo_system.results_folder}/")
            
        else:
            print("❌ Demo fehlgeschlagen: Keine Ergebnisse")
            
    except Exception as e:
        print(f"❌ Fehler in CMO Demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()