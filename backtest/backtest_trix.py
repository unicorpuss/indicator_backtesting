#!/usr/bin/env python3
"""
TRIX (Triple Exponential Moving Average) BACKTESTING SYSTEM - OPTIMIZED VERSION
===============================================================================

Backtesting-System für TRIX (Triple Exponential Moving Average) Indikator mit:
- Trend-basierte Trading-Strategie
- TRIX > 0: Long-Position | TRIX <= 0: Cash-Position (0 = Mittellinie)
- Umfassende Performance-Analyse
- Automatische Heatmap- und Line Plot-Generierung
- Integration mit backtesting_base.py für Code-Wiederverwendung

TRIX (Triple Exponential Moving Average):
- Entwickelt zur Filterung von Kursschwankungen
- Berechnet Rate of Change einer dreifach geglätteten EMA
- TRIX = ROC von EMA(EMA(EMA(Close)))
- Eliminiert kurzfristige Schwankungen durch dreifache Glättung
- > 0 = Aufwärtstrend (Momentum positiv)
- < 0 = Abwärtstrend (Momentum negativ)
- 0 = Trendwende oder neutraler Punkt

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

class TRIXBacktestingSystem(BaseBacktestingSystem):
    """
    TRIX (Triple Exponential Moving Average) Backtesting System
    
    Der TRIX ist ein Momentum-Indikator basierend auf dreifach geglätteten EMAs.
    Strategie: TRIX > 0 = Long-Position, TRIX <= 0 = Cash-Position (0 = Mittellinie)
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "TRIX", **kwargs)
        self.indicator_name = "TRIX"
        self.strategy_description = "TRIX > 0: Long-Position | TRIX <= 0: Cash-Position"
        self.threshold = 0.0

    def calculate_trix_signals(self, data: pd.DataFrame, trix_length: int = 14) -> pd.DataFrame:
        """
        Berechnet TRIX-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            trix_length: TRIX-Periode für die EMA-Berechnung
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < trix_length * 3 + 1:  # TRIX benötigt mehr Daten wegen dreifacher Glättung
            return pd.DataFrame()
        
        try:
            # Berechne TRIX mit talib
            trix = ta.TRIX(data['close'].values, timeperiod=trix_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['trix'] = trix
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # TRIX Long-Only Signale (TRIX > 0 = Long, TRIX <= 0 = Cash)
            signals_df['position'] = np.where(signals_df['trix'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei TRIX-Berechnung: {e}")
            return pd.DataFrame()

    def run_trix_backtests(self, trix_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt TRIX-Backtests über verschiedene Perioden durch
        
        Args:
            trix_range: Range der TRIX-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=trix_range,
            length_param_name='trix_length',
            calculate_signals_func=self.calculate_trix_signals,
            indicator_name='TRIX'
        )

    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden TRIX-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit TRIX-spezifischen Parametern
        strategy_description = "TRIX Long-Only Strategy (TRIX > 0 = Long, TRIX <= 0 = Cash)"
        super().generate_comprehensive_report(results_df, 'trix_length', strategy_description)
        
        # Zusätzliche TRIX-spezifische Analyse
        print(f"\n📋 Erstelle TRIX-spezifische Analyse...")
        self.generate_trix_specific_analysis(results_df)
    
    def generate_trix_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert TRIX-spezifische Analyse
        Fokussiert auf die besten TRIX-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für TRIX-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle TRIX-spezifische Analyse...")
        
        # Finde beste TRIX-Kalibrierungen
        best_trix_calibration = self.find_best_average_calibration(results_df, 'trix_length')
        
        if best_trix_calibration:
            # Erstelle zusätzlichen TRIX-Bericht
            trix_report_lines = []
            trix_report_lines.append("=" * 80)
            trix_report_lines.append("🎯 TRIX-SPEZIFISCHE ANALYSE")
            trix_report_lines.append("=" * 80)
            trix_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            trix_report_lines.append("")
            
            # Beste kombinierte TRIX-Länge
            if 'best_trix_combined' in best_trix_calibration:
                trix_report_lines.append(
                    f"🥇 Beste Durchschnittliche TRIX-Länge (Kombiniert): TRIX-{best_trix_calibration['best_trix_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte TRIX
                combined_keys = ['combined_trix_sharpe_ratio', 'combined_trix_total_return', 'combined_trix_max_drawdown']
                if all(key in best_trix_calibration for key in combined_keys):
                    trix_report_lines.append(
                        f"   📊 Avg Sharpe: {best_trix_calibration['combined_trix_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_trix_calibration['combined_trix_total_return']:.1%} | "
                        f"Avg DD: {best_trix_calibration['combined_trix_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_trix_sortino_ratio' in best_trix_calibration and 'combined_trix_win_rate' in best_trix_calibration:
                    trix_report_lines.append(
                        f"   📈 Avg Sortino: {best_trix_calibration['combined_trix_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_trix_calibration['combined_trix_win_rate']:.1%} | "
                        f"Score: {best_trix_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top TRIXs für einzelne Metriken
            trix_report_lines.append("")
            trix_report_lines.append("📈 Beste Durchschnitts-TRIXs nach Metriken:")
            
            metric_keys = [
                ('best_trix_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_trix_total_return', 'avg_total_return', 'Total Return'),
                ('best_trix_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_trix_calibration and avg_key in best_trix_calibration:
                    avg_val = best_trix_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    trix_report_lines.append(f"   • {metric_name}: TRIX-{best_trix_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # TRIX-spezifische Empfehlungen
            trix_report_lines.append(f"\n💡 TRIX STRATEGIE EMPFEHLUNGEN")
            trix_report_lines.append("-" * 60)
            trix_report_lines.append("📋 TRIX LONG-ONLY STRATEGIE INSIGHTS:")
            trix_report_lines.append(f"   • Long-Position wenn TRIX > 0 (dreifach geglätteter Aufwärtstrend)")
            trix_report_lines.append(f"   • Cash-Position wenn TRIX <= 0 (dreifach geglätteter Abwärtstrend)")
            trix_report_lines.append(f"   • TRIX filtert kurzfristige Schwankungen durch dreifache Glättung")
            trix_report_lines.append(f"   • Positive TRIX-Werte: Bestätigter Aufwärtstrend")
            trix_report_lines.append(f"   • Negative TRIX-Werte: Bestätigter Abwärtstrend")
            trix_report_lines.append(f"   • TRIX-Nulllinie: Kritischer Trendwendepunkt")
            trix_report_lines.append(f"   • Kürzere TRIX-Perioden: Sensitivere, aber rauschigere Signale")
            trix_report_lines.append(f"   • Längere TRIX-Perioden: Stabilere, aber langsamere Signale")
            trix_report_lines.append(f"   • TRIX eignet sich besonders für Trendfolge-Strategien")
            
            # Parameter-Optimierung
            trix_report_lines.append(f"\n🎯 TRIX PARAMETER-OPTIMIERUNG")
            trix_report_lines.append("-" * 60)
            trix_report_lines.append(f"📌 OPTIMALE TRIX-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_trix_combined' in best_trix_calibration:
                combined_score = best_trix_calibration.get('avg_combined_score', 0.0)
                trix_report_lines.append(f"   🥇 Beste Gesamtperformance: TRIX-{best_trix_calibration['best_trix_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_trix_calibration and avg_key in best_trix_calibration:
                    avg_val = best_trix_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    trix_report_lines.append(f"   📈 Höchste {metric_name}: TRIX-{best_trix_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_trix_max_drawdown' in best_trix_calibration and 'avg_max_drawdown' in best_trix_calibration:
                avg_dd = best_trix_calibration['avg_max_drawdown']
                trix_report_lines.append(f"   🛡️ Niedrigster Drawdown: TRIX-{best_trix_calibration['best_trix_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            # Klassische TRIX-Kalibrierungen
            trix_report_lines.append(f"\n📚 KLASSISCHE TRIX-KALIBRIERUNGEN:")
            trix_report_lines.append("-" * 60)
            classical_periods = [12, 14, 18, 21, 30, 50]
            for period in classical_periods:
                period_data = results_df[results_df['trix_length'] == period]
                if not period_data.empty:
                    avg_sharpe = period_data['sharpe_ratio'].mean()
                    avg_return = period_data['total_return'].mean()
                    avg_dd = period_data['max_drawdown'].mean()
                    trix_report_lines.append(
                        f"   • TRIX-{period}: Sharpe {avg_sharpe:.3f} | Return {avg_return:.1%} | DD {avg_dd:.1%}"
                    )
            
            # TRIX-spezifische Marktbedingungen
            trix_report_lines.append(f"\n📊 TRIX MARKTBEDINGUNGEN:")
            trix_report_lines.append("-" * 60)
            trix_report_lines.append(f"💡 TRIX PERFORMANCE CHARAKTERISTIKA:")
            trix_report_lines.append(f"   • Trendmärkte: TRIX zeigt beste Performance in klaren Trends")
            trix_report_lines.append(f"   • Seitwärtsmärkte: Kann zu wenigen, aber qualitativ guten Signalen führen")
            trix_report_lines.append(f"   • Volatile Märkte: Dreifache Glättung reduziert Fehlsignale")
            trix_report_lines.append(f"   • Langfristige Trends: TRIX ist besonders stark in langfristigen Aufwärtstrends")
            trix_report_lines.append(f"   • Trendwenden: TRIX kann Trendwenden verzögert, aber zuverlässig anzeigen")
            
            # Erweiterte TRIX-Strategien
            trix_report_lines.append(f"\n🔧 ERWEITERTE TRIX-STRATEGIEN:")
            trix_report_lines.append("-" * 60)
            trix_report_lines.append(f"🎯 MÖGLICHE TRIX-VERBESSERUNGEN:")
            trix_report_lines.append(f"   • TRIX-Signal + Signallinie: Zusätzliche EMA der TRIX für Bestätigung")
            trix_report_lines.append(f"   • TRIX-Divergenzen: Erkennung von Preis-TRIX-Divergenzen")
            trix_report_lines.append(f"   • Multi-Timeframe TRIX: Kombination verschiedener TRIX-Perioden")
            trix_report_lines.append(f"   • TRIX-Momentum: Betrachtung der TRIX-Steigung für Momentum")
            trix_report_lines.append(f"   • TRIX-Filter: Kombination mit anderen Indikatoren als Filter")
            
            trix_report_lines.append(f"\n" + "=" * 68)
            trix_report_lines.append(f"🏁 TRIX-ANALYSE ABGESCHLOSSEN")
            trix_report_lines.append("=" * 68)
            
            # Speichere TRIX-spezifischen Bericht
            trix_report_text = "\n".join(trix_report_lines)
            trix_report_path = os.path.join(self.results_folder, 'trix_specific_analysis.txt')
            
            with open(trix_report_path, 'w', encoding='utf-8') as f:
                f.write(trix_report_text)
            
            print(f"📄 TRIX-spezifische Analyse gespeichert: {trix_report_path}")

def main():
    """
    Hauptfunktion für TRIX Backtesting System Demo
    """
    print("🚀 TRIX BACKTESTING SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Initialisiere System
        trix_system = TRIXBacktestingSystem()
        
        # Führe Backtests durch (kleiner Range für Demo)
        results_df = trix_system.run_trix_backtests(trix_range=range(10, 151))
        
        if not results_df.empty:
            print(f"✅ Demo erfolgreich: {len(results_df)} Tests durchgeführt")
            
            # Generiere umfassenden Bericht
            print("\n📊 Generiere umfassenden TRIX-Bericht...")
            trix_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 TRIX-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {trix_system.results_folder}/")
            
        else:
            print("❌ Demo fehlgeschlagen: Keine Ergebnisse")
            
    except Exception as e:
        print(f"❌ Fehler in TRIX Demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()