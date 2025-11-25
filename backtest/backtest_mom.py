#!/usr/bin/env python3
"""
MOM (Momentum) BACKTESTING SYSTEM - OPTIMIZED VERSION
=====================================================

Backtesting-System für Momentum (MOM) Indikator mit:
- Momentum-basierte Trading-Strategie
- MOM > 0: Long-Position | MOM <= 0: Cash-Position (0 = Mittellinie)
- Umfassende Performance-Analyse
- Automatische Heatmap- und Line Plot-Generierung
- Integration mit backtesting_base.py für Code-Wiederverwendung

MOM (Momentum):
- Klassischer Momentum-Indikator
- Berechnet Preisdifferenz zwischen aktueller und vergangener Periode
- MOM = Preis(heute) - Preis(vor n Perioden)
- > 0 = Bullish Momentum (Preis gestiegen)
- < 0 = Bearish Momentum (Preis gefallen)
- 0 = Neutral (kein Momentum)

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

class MOMBacktestingSystem(BaseBacktestingSystem):
    """
    Momentum (MOM) Backtesting System
    
    Der MOM ist ein klassischer Momentum-Indikator.
    Strategie: MOM > 0 = Long-Position, MOM <= 0 = Cash-Position (0 = Mittellinie)
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "MOM", **kwargs)
        self.indicator_name = "MOM"
        self.strategy_description = "MOM > 0: Long-Position | MOM <= 0: Cash-Position"
        self.threshold = 0.0

    def calculate_mom_signals(self, data: pd.DataFrame, mom_length: int = 14) -> pd.DataFrame:
        """
        Berechnet MOM-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            mom_length: MOM-Periode (Lookback-Periode)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < mom_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne MOM mit talib
            mom = ta.MOM(data['close'].values, timeperiod=mom_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['mom'] = mom
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # MOM Long-Only Signale (MOM > 0 = Long, MOM <= 0 = Cash)
            signals_df['position'] = np.where(signals_df['mom'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei MOM-Berechnung: {e}")
            return pd.DataFrame()

    def run_mom_backtests(self, mom_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt MOM-Backtests über verschiedene Perioden durch
        
        Args:
            mom_range: Range der MOM-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=mom_range,
            length_param_name='mom_length',
            calculate_signals_func=self.calculate_mom_signals,
            indicator_name='MOM'
        )

    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden MOM-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit MOM-spezifischen Parametern
        strategy_description = "MOM Long-Only Strategy (MOM > 0 = Long, MOM <= 0 = Cash)"
        super().generate_comprehensive_report(results_df, 'mom_length', strategy_description)
        
        # Zusätzliche MOM-spezifische Analyse
        print(f"\n📋 Erstelle MOM-spezifische Analyse...")
        self.generate_mom_specific_analysis(results_df)
    
    def generate_mom_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert MOM-spezifische Analyse
        Fokussiert auf die besten MOM-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für MOM-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle MOM-spezifische Analyse...")
        
        # Finde beste MOM-Kalibrierungen
        best_mom_calibration = self.find_best_average_calibration(results_df, 'mom_length')
        
        if best_mom_calibration:
            # Erstelle zusätzlichen MOM-Bericht
            mom_report_lines = []
            mom_report_lines.append("=" * 80)
            mom_report_lines.append("🎯 MOM-SPEZIFISCHE ANALYSE")
            mom_report_lines.append("=" * 80)
            mom_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            mom_report_lines.append("")
            
            # Beste kombinierte MOM-Länge
            if 'best_mom_combined' in best_mom_calibration:
                mom_report_lines.append(
                    f"🥇 Beste Durchschnittliche MOM-Länge (Kombiniert): MOM-{best_mom_calibration['best_mom_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte MOM
                combined_keys = ['combined_mom_sharpe_ratio', 'combined_mom_total_return', 'combined_mom_max_drawdown']
                if all(key in best_mom_calibration for key in combined_keys):
                    mom_report_lines.append(
                        f"   📊 Avg Sharpe: {best_mom_calibration['combined_mom_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_mom_calibration['combined_mom_total_return']:.1%} | "
                        f"Avg DD: {best_mom_calibration['combined_mom_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_mom_sortino_ratio' in best_mom_calibration and 'combined_mom_win_rate' in best_mom_calibration:
                    mom_report_lines.append(
                        f"   📈 Avg Sortino: {best_mom_calibration['combined_mom_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_mom_calibration['combined_mom_win_rate']:.1%} | "
                        f"Score: {best_mom_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top MOMs für einzelne Metriken
            mom_report_lines.append("")
            mom_report_lines.append("📈 Beste Durchschnitts-MOMs nach Metriken:")
            
            metric_keys = [
                ('best_mom_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_mom_total_return', 'avg_total_return', 'Total Return'),
                ('best_mom_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_mom_calibration and avg_key in best_mom_calibration:
                    avg_val = best_mom_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    mom_report_lines.append(f"   • {metric_name}: MOM-{best_mom_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # MOM-spezifische Empfehlungen
            mom_report_lines.append(f"\n💡 MOM STRATEGIE EMPFEHLUNGEN")
            mom_report_lines.append("-" * 60)
            mom_report_lines.append("📋 MOM LONG-ONLY STRATEGIE INSIGHTS:")
            mom_report_lines.append(f"   • Long-Position wenn MOM > 0 (Preis über Vergangenheitswert)")
            mom_report_lines.append(f"   • Cash-Position wenn MOM <= 0 (Preis unter/gleich Vergangenheitswert)")
            mom_report_lines.append(f"   • Hohe MOM-Werte: Starkes Momentum, mögliche Fortsetzung")
            mom_report_lines.append(f"   • Niedrige/negative MOM-Werte: Schwaches/negatives Momentum")
            mom_report_lines.append(f"   • MOM = 0: Neutraler Punkt, Preis gleich Vergangenheit")
            mom_report_lines.append(f"   • Kürzere MOM-Perioden: Sensitivere Signale, mehr Trades")
            mom_report_lines.append(f"   • Längere MOM-Perioden: Stabilere Trends, weniger Trades")
            mom_report_lines.append(f"   • MOM reagiert schnell auf Trendwenden (direkter Preisvergleich)")
            
            # Parameter-Optimierung
            mom_report_lines.append(f"\n🎯 MOM PARAMETER-OPTIMIERUNG")
            mom_report_lines.append("-" * 60)
            mom_report_lines.append(f"📌 OPTIMALE MOM-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_mom_combined' in best_mom_calibration:
                combined_score = best_mom_calibration.get('avg_combined_score', 0.0)
                mom_report_lines.append(f"   🥇 Beste Gesamtperformance: MOM-{best_mom_calibration['best_mom_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_mom_calibration and avg_key in best_mom_calibration:
                    avg_val = best_mom_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    mom_report_lines.append(f"   📈 Höchste {metric_name}: MOM-{best_mom_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_mom_max_drawdown' in best_mom_calibration and 'avg_max_drawdown' in best_mom_calibration:
                avg_dd = best_mom_calibration['avg_max_drawdown']
                mom_report_lines.append(f"   🛡️ Niedrigster Drawdown: MOM-{best_mom_calibration['best_mom_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            # Klassische MOM-Kalibrierungen
            mom_report_lines.append(f"\n📚 KLASSISCHE MOM-KALIBRIERUNGEN:")
            mom_report_lines.append("-" * 60)
            classical_periods = [10, 12, 14, 20, 21, 30]
            for period in classical_periods:
                period_data = results_df[results_df['mom_length'] == period]
                if not period_data.empty:
                    avg_sharpe = period_data['sharpe_ratio'].mean()
                    avg_return = period_data['total_return'].mean()
                    avg_dd = period_data['max_drawdown'].mean()
                    mom_report_lines.append(
                        f"   • MOM-{period}: Sharpe {avg_sharpe:.3f} | Return {avg_return:.1%} | DD {avg_dd:.1%}"
                    )
            
            mom_report_lines.append(f"\n" + "=" * 68)
            mom_report_lines.append(f"🏁 MOM-ANALYSE ABGESCHLOSSEN")
            mom_report_lines.append("=" * 68)
            
            # Speichere MOM-spezifischen Bericht
            mom_report_text = "\n".join(mom_report_lines)
            mom_report_path = os.path.join(self.results_folder, 'mom_specific_analysis.txt')
            
            with open(mom_report_path, 'w', encoding='utf-8') as f:
                f.write(mom_report_text)
            
            print(f"📄 MOM-spezifische Analyse gespeichert: {mom_report_path}")

def main():
    """
    Hauptfunktion für MOM Backtesting System Demo
    """
    print("🚀 MOM BACKTESTING SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Initialisiere System
        mom_system = MOMBacktestingSystem()
        
        # Führe Backtests durch (kleiner Range für Demo)
        results_df = mom_system.run_mom_backtests(mom_range=range(10, 151))
        
        if not results_df.empty:
            print(f"✅ Demo erfolgreich: {len(results_df)} Tests durchgeführt")
            
            # Generiere umfassenden Bericht
            print("\n📊 Generiere umfassenden MOM-Bericht...")
            mom_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 MOM-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {mom_system.results_folder}/")
            
        else:
            print("❌ Demo fehlgeschlagen: Keine Ergebnisse")
            
    except Exception as e:
        print(f"❌ Fehler in MOM Demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()