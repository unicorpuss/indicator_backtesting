"""
EMA Backtesting System (Optimiert mit Base Class)

Dieses System testet EMA-basierte Trading-Strategien über verschiedene Assets und Perioden:
- EMA-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets  
- Long/Short Strategie: Price > EMA = Long Position, Price <= EMA = Short Position
- Umfassende Performance-Metriken (Sharpe, Sortino, Omega, etc.)
- Verwendet gemeinsame Funktionen aus backtesting_base.py

EMA (Exponential Moving Average):
- Gleitender Durchschnitt mit exponentieller Gewichtung
- Reagiert schneller auf Preisänderungen als Simple Moving Average (SMA)
- Neuere Preise haben höheres Gewicht als ältere
- Long-Signal: Price > EMA (Preis über gleitendem Durchschnitt)
- Short-Signal: Price <= EMA (Preis unter gleitendem Durchschnitt)
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

class EMABacktestingSystem(BaseBacktestingSystem):
    """
    EMA-spezifisches Backtesting-System (Long/Short Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        """
        Initialisiert das EMA Backtesting System
        
        Args:
            max_assets: Maximale Anzahl der Assets
            **kwargs: Zusätzliche Parameter (assets_csv, category, etc.)
        """
        super().__init__(max_assets, "EMA", **kwargs)
        self.indicator_name = "EMA"
        self.strategy_description = "Price > EMA: Long-Position | Price <= EMA: Short-Position"
        self.threshold = None  # EMA ist ein Moving Average, kein fester Schwellenwert
    
    def calculate_ema_signals(self, data: pd.DataFrame, ema_length: int = 14) -> pd.DataFrame:
        """
        Berechnet EMA-Signale für die Long/Short Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            ema_length: EMA-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < ema_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne EMA
            ema = ta.EMA(data['close'].values, timeperiod=ema_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['ema'] = ema
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # EMA Long/Short Signale (Price > EMA = Long, Price <= EMA = Short)
            signals_df['position'] = np.where(signals_df['close'] > signals_df['ema'], 1, -1)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei EMA-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_ema_backtests(self, ema_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt EMA-Backtests über verschiedene Perioden durch
        
        Args:
            ema_range: Range der EMA-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=ema_range,
            length_param_name='ema_length',
            calculate_signals_func=self.calculate_ema_signals,
            indicator_name='EMA'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden EMA-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode mit EMA-spezifischen Parametern
        strategy_description = "EMA Crossover Strategy (Price > EMA = Long, Price <= EMA = Short)"
        super().generate_comprehensive_report(results_df, 'ema_length', strategy_description)
        
        # Zusätzliche EMA-spezifische Analyse (wie in ursprünglichem System)
        print(f"\n📋 Erstelle EMA-spezifische Analyse...")
        
        # Verwende die EMA-spezifische Analyse-Funktion
        self.generate_ema_specific_analysis(results_df)
    
    def generate_ema_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert EMA-spezifische Analyse
        Fokussiert auf die besten EMA-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für EMA-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle EMA-spezifische Analyse...")
        
        # Finde beste EMA-Kalibrierungen (nutzt gemeinsame Funktion)
        best_ema_calibration = self.find_best_average_calibration(results_df, 'ema_length')
        
        if best_ema_calibration:
            # Erstelle zusätzlichen EMA-Bericht
            ema_report_lines = []
            ema_report_lines.append("=" * 80)
            ema_report_lines.append("🎯 EMA-SPEZIFISCHE ANALYSE")
            ema_report_lines.append("=" * 80)
            ema_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ema_report_lines.append("")
            
            # Beste kombinierte EMA-Länge
            if 'best_ema_combined' in best_ema_calibration:
                ema_report_lines.append(
                    f"🥇 Beste Durchschnittliche EMA-Länge (Kombiniert): EMA-{best_ema_calibration['best_ema_combined']:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte EMA
                combined_keys = ['combined_ema_sharpe_ratio', 'combined_ema_total_return', 'combined_ema_max_drawdown']
                if all(key in best_ema_calibration for key in combined_keys):
                    ema_report_lines.append(
                        f"   📊 Avg Sharpe: {best_ema_calibration['combined_ema_sharpe_ratio']:.3f} | "
                        f"Avg Return: {best_ema_calibration['combined_ema_total_return']:.1%} | "
                        f"Avg DD: {best_ema_calibration['combined_ema_max_drawdown']:.1%}"
                    )
                
                # Sortino und Win Rate
                if 'combined_ema_sortino_ratio' in best_ema_calibration and 'combined_ema_win_rate' in best_ema_calibration:
                    ema_report_lines.append(
                        f"   📈 Avg Sortino: {best_ema_calibration['combined_ema_sortino_ratio']:.3f} | "
                        f"Avg Win Rate: {best_ema_calibration['combined_ema_win_rate']:.1%} | "
                        f"Score: {best_ema_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top EMAs für einzelne Metriken
            ema_report_lines.append("")
            ema_report_lines.append("📈 Beste Durchschnitts-EMAs nach Metriken:")
            
            metric_keys = [
                ('best_ema_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_ema_total_return', 'avg_total_return', 'Total Return'),
                ('best_ema_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_ema_calibration and avg_key in best_ema_calibration:
                    avg_val = best_ema_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    ema_report_lines.append(f"   • {metric_name}: EMA-{best_ema_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # EMA-spezifische Empfehlungen
            ema_report_lines.append(f"\n💡 EMA STRATEGIE EMPFEHLUNGEN")
            ema_report_lines.append("-" * 60)
            ema_report_lines.append("📋 EMA CROSSOVER LONG/SHORT STRATEGIE INSIGHTS:")
            ema_report_lines.append(f"   • Long-Position wenn Preis > EMA (bullisher Trend)")
            ema_report_lines.append(f"   • Short-Position wenn Preis <= EMA (bearisher Trend)")
            ema_report_lines.append(f"   • EMA reagiert schneller auf Preisänderungen als SMA")
            ema_report_lines.append(f"   • Exponentiell gewichtete Durchschnitte betonen jüngere Daten")
            ema_report_lines.append(f"   • Kürzere EMA-Perioden: Mehr Trades, höhere Sensitivität")
            ema_report_lines.append(f"   • Längere EMA-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            ema_report_lines.append(f"\n🎯 EMA PARAMETER-OPTIMIERUNG")
            ema_report_lines.append("-" * 60)
            ema_report_lines.append(f"📌 OPTIMALE EMA-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if 'best_ema_combined' in best_ema_calibration:
                combined_score = best_ema_calibration.get('avg_combined_score', 0.0)
                ema_report_lines.append(f"   🥇 Beste Gesamtperformance: EMA-{best_ema_calibration['best_ema_combined']:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_ema_calibration and avg_key in best_ema_calibration:
                    avg_val = best_ema_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    ema_report_lines.append(f"   📈 Höchste {metric_name}: EMA-{best_ema_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            if 'best_ema_max_drawdown' in best_ema_calibration and 'avg_max_drawdown' in best_ema_calibration:
                avg_dd = best_ema_calibration['avg_max_drawdown']
                ema_report_lines.append(f"   🛡️ Niedrigster Drawdown: EMA-{best_ema_calibration['best_ema_max_drawdown']:.0f} ({avg_dd:.1%})")
            
            # EMA-spezifische Trend-Analyse
            ema_report_lines.append(f"\n📊 EMA TREND-ANALYSE")
            ema_report_lines.append("-" * 60)
            ema_report_lines.append("💡 EMA CROSSOVER INSIGHTS:")
            ema_report_lines.append(f"   • EMA als dynamischer Support/Resistance Level")
            ema_report_lines.append(f"   • Preis über EMA: Bullisher Trend (Long-Bias)")
            ema_report_lines.append(f"   • Preis unter EMA: Bearisher Trend (Short-Bias)")
            ema_report_lines.append(f"   • EMA-Steigung zeigt Trend-Stärke an")
            ema_report_lines.append(f"   • Crossover-Punkte als Entry/Exit-Signale")
            ema_report_lines.append(f"   • Whipsaws möglich in seitwärts gerichteten Märkten")
            
            # Trading-Empfehlungen
            ema_report_lines.append(f"\n⚡ EMA TRADING EMPFEHLUNGEN")
            ema_report_lines.append("-" * 60)
            ema_report_lines.append("🔄 LONG/SHORT TRADING PRINZIPIEN:")
            ema_report_lines.append(f"   • Long Entry: Preis crosst EMA von unten nach oben")
            ema_report_lines.append(f"   • Short Entry: Preis crosst EMA von oben nach unten")
            ema_report_lines.append(f"   • Trendbestätigung durch EMA-Richtung")
            ema_report_lines.append(f"   • Risk Management: Stop-Loss bei EMA-Breach")
            
            ema_report_lines.append(f"\n📈 EMA-LÄNGEN CHARAKTERISTIKA:")
            ema_report_lines.append(f"   • EMA-10 bis EMA-20: Kurzfristig, viele Signale, höhere Volatilität")
            ema_report_lines.append(f"   • EMA-21 bis EMA-50: Mittelfristig, balanced Performance")
            ema_report_lines.append(f"   • EMA-50 bis EMA-100: Langfristig, weniger aber qualitativ bessere Signale")
            ema_report_lines.append(f"   • EMA-100+: Sehr langfristig, starke Trend-Identifikation")
            
            ema_report_lines.append(f"\n" + "=" * 68)
            ema_report_lines.append(f"🏁 EMA-ANALYSE ABGESCHLOSSEN")
            ema_report_lines.append("=" * 68)
            
            # Speichere EMA-spezifischen Bericht
            ema_report_text = "\n".join(ema_report_lines)
            ema_report_path = os.path.join(self.results_folder, 'ema_specific_analysis.txt')
            
            with open(ema_report_path, 'w', encoding='utf-8') as f:
                f.write(ema_report_text)
            
            print(f"📄 EMA-spezifische Analyse gespeichert: {ema_report_path}")

def main():
    """
    Hauptfunktion für EMA Backtesting System
    """
    print("🚀 EMA BACKTESTING SYSTEM START (Long/Short)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long/Short (Price > EMA = Long, Price <= EMA = Short)")
    print("   • EMA-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • EMA: Exponentiell gewichteter gleitender Durchschnitt")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe EMA Backtesting System aus
        ema_system = EMABacktestingSystem(max_assets=20)
        
        # Teste verschiedene EMA-Perioden (Einser-Schritte wie im Original)
        ema_range = range(5, 151)  # 5, 6, 7, 8, ..., 150 (Einser-Schritte)
        
        # Führe Backtests durch
        results_df = ema_system.run_ema_backtests(ema_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            ema_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 EMA-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {ema_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim EMA-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()