#!/usr/bin/env python3
"""
MFI (Money Flow Index) BACKTESTING SYSTEM - OPTIMIZED VERSION
==============================================================

Backtesting-System für Money Flow Index (MFI) Indikator mit:
- Momentum-basierte Trading-Strategie
- MFI > 50: Long-Position | MFI <= 50: Cash-Position (50 = Mittellinie)
- Umfassende Performance-Analyse
- Automatische Heatmap- und Line Plot-Generierung
- Integration mit backtesting_base.py für Code-Wiederverwendung

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

class MFIBacktestingSystem(BaseBacktestingSystem):
    """
    Money Flow Index (MFI) Backtesting System
    
    Der MFI ist ein momentum-basierter Oszillator, der Preis- und Volumendaten kombiniert.
    Strategie: MFI > 50 = Long-Position, MFI <= 50 = Cash-Position (50 = Mittellinie)
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "MFI", **kwargs)
        self.indicator_name = "MFI"
        self.strategy_description = "MFI > 50: Long-Position | MFI <= 50: Cash-Position"
        self.threshold = 50.0

    def calculate_mfi_signals(self, data: pd.DataFrame, mfi_length: int = 14) -> pd.DataFrame:
        """
        Berechnet MFI-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            mfi_length: MFI-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < mfi_length + 1:
            return pd.DataFrame()
        
        try:
            # Stelle sicher, dass Volumen-Daten verfügbar sind
            data_copy = data.copy()
            if 'volume' not in data_copy.columns:
                data_copy['volume'] = np.ones(len(data_copy)) * 1000000
            
            # Berechne MFI
            mfi = ta.MFI(data_copy['high'].astype(float).values, 
                        data_copy['low'].astype(float).values, 
                        data_copy['close'].astype(float).values, 
                        data_copy['volume'].astype(float).values, 
                        timeperiod=mfi_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['mfi'] = mfi
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # MFI Long-Only Signale (MFI > 50 = Long, MFI <= 50 = Cash)
            signals_df['position'] = np.where(signals_df['mfi'] > 50, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei MFI-Berechnung: {e}")
            return pd.DataFrame()

    def run_mfi_backtests(self, mfi_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt MFI-Backtests über verschiedene Perioden durch
        
        Args:
            mfi_range: Range der MFI-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=mfi_range,
            length_param_name='mfi_length',
            calculate_signals_func=self.calculate_mfi_signals,
            indicator_name='MFI'
        )

    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden MFI-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit MFI-spezifischen Parametern
        strategy_description = "MFI Long-Only Strategy (MFI > 50 = Long, MFI <= 50 = Cash)"
        super().generate_comprehensive_report(results_df, 'mfi_length', strategy_description)
        
        # Zusätzliche MFI-spezifische Analyse (wie in anderen Systemen)
        print(f"\n📋 Erstelle MFI-spezifische Analyse...")
        
        # Verwende die MFI-spezifische Analyse-Funktion
        self.generate_mfi_specific_analysis(results_df)
    
    def generate_mfi_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert MFI-spezifische Analyse ähnlich der CCI-Analyse
        Fokussiert auf die besten MFI-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für MFI-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle MFI-spezifische Analyse...")
        
        # Finde beste MFI-Kalibrierungen (nutzt gemeinsame Funktion)
        best_mfi_calibration = self.find_best_average_calibration(results_df, 'mfi_length')
        
        if best_mfi_calibration:
            # Erstelle zusätzlichen MFI-Bericht
            mfi_report_lines = []
            mfi_report_lines.append("=" * 80)
            mfi_report_lines.append("🎯 MFI-SPEZIFISCHE ANALYSE")
            mfi_report_lines.append("=" * 80)
            mfi_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            mfi_report_lines.append("")
            
            # Beste kombinierte MFI-Länge
            if 'best_mfi_combined' in best_mfi_calibration:
                mfi_report_lines.append(
                    f"🥇 Beste Durchschnittliche MFI-Länge (Kombiniert): MFI-{best_mfi_calibration['best_mfi_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte MFI
                combined_keys = ['combined_mfi_sharpe_ratio', 'combined_mfi_total_return', 'combined_mfi_max_drawdown']
                if all(key in best_mfi_calibration for key in combined_keys):
                    mfi_report_lines.append(
                        f"   📊 Avg Sharpe: {best_mfi_calibration['combined_mfi_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_mfi_calibration['combined_mfi_total_return']:.1%} | "
                        f"Avg DD: {best_mfi_calibration['combined_mfi_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_mfi_sortino_ratio' in best_mfi_calibration and 'combined_mfi_win_rate' in best_mfi_calibration:
                    mfi_report_lines.append(
                        f"   📈 Avg Sortino: {best_mfi_calibration['combined_mfi_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_mfi_calibration['combined_mfi_win_rate']:.1%} | "
                        f"Score: {best_mfi_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top MFIs für einzelne Metriken
            mfi_report_lines.append("")
            mfi_report_lines.append("📈 Beste Durchschnitts-MFIs nach Metriken:")
            
            metric_keys = [
                ('best_mfi_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_mfi_total_return', 'avg_total_return', 'Total Return'),
                ('best_mfi_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_mfi_calibration and avg_key in best_mfi_calibration:
                    avg_val = best_mfi_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    mfi_report_lines.append(f"   • {metric_name}: MFI-{best_mfi_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # MFI-spezifische Empfehlungen
            mfi_report_lines.append(f"\n💡 MFI STRATEGIE EMPFEHLUNGEN")
            mfi_report_lines.append("-" * 60)
            mfi_report_lines.append("📋 MFI LONG-ONLY STRATEGIE INSIGHTS:")
            mfi_report_lines.append(f"   • Long-Position wenn MFI > 50 (bullisher Money Flow)")
            mfi_report_lines.append(f"   • Cash-Position wenn MFI <= 50 (bearisher Money Flow)")
            mfi_report_lines.append(f"   • MFI kombiniert Preis- UND Volumen-Information")
            mfi_report_lines.append(f"   • MFI über 80: Überkauft, mögliche Korrektur")
            mfi_report_lines.append(f"   • MFI unter 20: Überverkauft, mögliche Erholung")
            mfi_report_lines.append(f"   • Kürzere MFI-Perioden: Mehr Trades, höhere Sensitivität")
            mfi_report_lines.append(f"   • Längere MFI-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            mfi_report_lines.append(f"\n🎯 MFI PARAMETER-OPTIMIERUNG")
            mfi_report_lines.append("-" * 60)
            mfi_report_lines.append(f"📌 OPTIMALE MFI-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_mfi_combined' in best_mfi_calibration:
                combined_score = best_mfi_calibration.get('avg_combined_score', 0.0)
                mfi_report_lines.append(f"   🥇 Beste Gesamtperformance: MFI-{best_mfi_calibration['best_mfi_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_mfi_calibration and avg_key in best_mfi_calibration:
                    avg_val = best_mfi_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    mfi_report_lines.append(f"   📈 Höchste {metric_name}: MFI-{best_mfi_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_mfi_max_drawdown' in best_mfi_calibration and 'avg_max_drawdown' in best_mfi_calibration:
                avg_dd = best_mfi_calibration['avg_max_drawdown']
                mfi_report_lines.append(f"   🛡️ Niedrigster Drawdown: MFI-{best_mfi_calibration['best_mfi_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            mfi_report_lines.append(f"\n" + "=" * 68)
            mfi_report_lines.append(f"🏁 MFI-ANALYSE ABGESCHLOSSEN")
            mfi_report_lines.append("=" * 68)
            
            # Speichere MFI-spezifischen Bericht
            mfi_report_text = "\n".join(mfi_report_lines)
            mfi_report_path = os.path.join(self.results_folder, 'mfi_specific_analysis.txt')
            
            with open(mfi_report_path, 'w', encoding='utf-8') as f:
                f.write(mfi_report_text)
            
            print(f"📄 MFI-spezifische Analyse gespeichert: {mfi_report_path}")

def main():
    """
    Hauptfunktion für MFI Backtesting System
    """
    print("� MFI BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (MFI > 50)")
    print("   • MFI-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe MFI Backtesting System aus
        mfi_system = MFIBacktestingSystem(max_assets=20)
        
        # Teste verschiedene MFI-Perioden (Einser-Schritte wie im Original)
        mfi_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = mfi_system.run_mfi_backtests(mfi_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            mfi_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 MFI-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {mfi_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim MFI-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()