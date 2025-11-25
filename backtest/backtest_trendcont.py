#!/usr/bin/env python3
"""
TREND CONTINUATION (Hull MA) BACKTESTING SYSTEM
================================================

Backtesting-System für Trend Continuation Indikator mit:
- Dual-Parameter Matrix-Analyse (Fast HMA vs Slow HMA)
- Uptrend (nicht-neutral): Long-Position | Downtrend/Neutral: Cash-Position
- Matrix-basierte Optimierung über verschiedene HMA-Kombinationen
- Umfassende Performance-Analyse mit Heatmaps
- Integration mit backtesting_base.py für Code-Wiederverwendung

Trend Continuation Indicator:
- Verwendet zwei Hull Moving Averages (HMA)
- Fast HMA steigend + über Slow HMA = Uptrend (Long)
- Fast HMA fallend + unter Slow HMA = Downtrend (Cash)
- Neutral-Zonen (Transition) = Cash
- Signal: 1 (Uptrend), -1 (Downtrend), 0 (Neutral)

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

class TrendContinuationBacktestingSystem(BaseBacktestingSystem):
    """
    Trend Continuation Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    Besonderheit: Zwei Parameter (Fast HMA und Slow HMA) mit Matrix-Analyse
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "TRENDCONT", **kwargs)
        self.indicator_name = "TRENDCONT"
        self.strategy_description = "Uptrend (Signal=1): Long-Position | Downtrend/Neutral: Cash-Position"
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
    
    def calculate_trendcont_signals(self, data: pd.DataFrame, fast_hma: int = 9, 
                                   slow_hma: int = 21) -> pd.DataFrame:
        """
        Berechnet Trend Continuation-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            fast_hma: Schnelle HMA-Periode
            slow_hma: Langsame HMA-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        min_length = max(fast_hma, slow_hma) + 10
        if len(data) < min_length:
            return pd.DataFrame()
        
        # Validierung: Fast muss kleiner als Slow sein
        if fast_hma >= slow_hma:
            return pd.DataFrame()
        
        try:
            source = data['close'].values
            
            # Berechne HMAs
            hma_fast = self.calculate_hma(source, fast_hma)
            hma_slow = self.calculate_hma(source, slow_hma)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['hma_fast'] = hma_fast
            signals_df['hma_slow'] = hma_slow
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # Bestimme Trend-Richtung
            signals_df['uptrend'] = signals_df['hma_fast'] > signals_df['hma_fast'].shift(1)
            signals_df['downtrend'] = signals_df['hma_fast'] < signals_df['hma_fast'].shift(1)
            
            # Bestimme Neutral-Zonen
            signals_df['neutral'] = (
                (signals_df['uptrend'] & (signals_df['hma_slow'] < signals_df['hma_fast'])) |
                (signals_df['downtrend'] & (signals_df['hma_slow'] > signals_df['hma_fast']))
            )
            
            # Signal: 1=Uptrend, -1=Downtrend, 0=Neutral
            # Long-Only: Nur bei Signal=1 (klarer Uptrend ohne Neutral)
            signals_df['signal'] = 0
            signals_df.loc[signals_df['neutral'], 'signal'] = 0
            signals_df.loc[~signals_df['neutral'] & signals_df['uptrend'], 'signal'] = 1
            signals_df.loc[~signals_df['neutral'] & signals_df['downtrend'], 'signal'] = -1
            
            # Position: Long nur bei Signal=1
            signals_df['position'] = np.where(signals_df['signal'] == 1, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei Trend Continuation-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_trendcont_backtests(self, fast_range: range = None, 
                               slow_range: range = None) -> pd.DataFrame:
        """
        Führt Trend Continuation-Backtests über vollständige Matrix durch
        
        Args:
            fast_range: Range der Fast-HMA-Perioden zum Testen (Standard: 5-150)
            slow_range: Range der Slow-HMA-Perioden zum Testen (Standard: 5-150)
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        if fast_range is None:
            fast_range = range(5, 151)
        
        if slow_range is None:
            slow_range = range(5, 151)
        
        print(f"🚀 TREND CONTINUATION BACKTESTING SYSTEM START")
        print("=" * 60)
        print(f"⚙️ KONFIGURATION:")
        print(f"   Fast-HMA-Range: {fast_range.start} bis {fast_range.stop-1}")
        print(f"   Slow-HMA-Range: {slow_range.start} bis {slow_range.stop-1}")
        print(f"   Validation: Fast < Slow (nur gültige Kombinationen)")
        
        # Berechne gültige Kombinationen
        valid_combinations = sum(1 for fast in fast_range for slow in slow_range if fast < slow)
        
        print(f"   Gültige Matrix: {valid_combinations} Kombinationen")
        print(f"   Max Assets: {len(self.assets)} (aus CSV)")
        print(f"   Strategie: Uptrend (Signal=1) = Long | Downtrend/Neutral = Cash")
        print()
        
        all_results = []
        total_combinations = valid_combinations * len(self.assets_data)
        current_combination = 0
        
        for fast_hma in fast_range:
            for slow_hma in slow_range:
                # Nur gültige Kombinationen (Fast < Slow)
                if fast_hma >= slow_hma:
                    continue
                
                for asset_name, asset_data in self.assets_data.items():
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    # Statische Progress-Anzeige
                    print(f"\r📊 Teste TRENDCONT Fast HMA({fast_hma:3d}) / Slow HMA({slow_hma:3d}) | Progress: {progress:5.1f}% ({current_combination}/{total_combinations})     ", end='', flush=True)
                    
                    # Berechne Signale
                    signals_df = self.calculate_trendcont_signals(asset_data, fast_hma, slow_hma)
                    
                    if signals_df.empty:
                        continue
                    
                    # Berechne Performance-Metriken
                    metrics = self.calculate_performance_metrics(signals_df['strategy_returns'])
                    
                    # Speichere Ergebnisse
                    result = {
                        'asset': asset_name,
                        'fast_hma': fast_hma,
                        'slow_hma': slow_hma,
                        'trendcont_combination': f"Fast={fast_hma}_Slow={slow_hma}",
                        'hma_ratio': slow_hma / fast_hma,
                        **metrics
                    }
                    all_results.append(result)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        results_df = pd.DataFrame(all_results)
        print(f"✅ Trend Continuation Matrix Backtests abgeschlossen: {len(all_results)} Kombinationen getestet")
        
        return results_df
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden Trend Continuation-Bericht
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        print(f"\n📋 Generiere umfassenden Trend Continuation Bericht...")
        
        # Erstelle Heatmaps für verschiedene Metriken
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown']
        
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Pivot für Heatmap
                pivot_data = results_df.pivot_table(
                    values=metric,
                    index='slow_hma',
                    columns='fast_hma',
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(15, 10))
                sns.heatmap(pivot_data, cmap='RdYlGn', center=0, annot=False, fmt='.2f')
                plt.title(f'Trend Continuation: Average {metric.replace("_", " ").title()} Heatmap')
                plt.xlabel('Fast HMA Period')
                plt.ylabel('Slow HMA Period')
                
                filename = f'trendcont_{metric}_heatmap.png'
                filepath = os.path.join(self.results_folder, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Heatmap gespeichert: {filename}")
        
        # Speichere CSV
        csv_path = os.path.join(self.results_folder, 'all_trendcont_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"📊 CSV-Daten gespeichert: {csv_path}")
        
        # Zusätzliche TRENDCONT-spezifische Analyse
        print(f"\n📋 Erstelle TRENDCONT-spezifische Analyse...")
        self.generate_trendcont_specific_analysis(results_df)
        
        print(f"\n✅ TREND CONTINUATION BACKTESTING ABGESCHLOSSEN")
        print(f"📊 {len(results_df)} Konfigurationen getestet")
        print(f"📁 Alle Ergebnisse in: {self.results_folder}/")
    
    def generate_trendcont_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert TRENDCONT-spezifische Analyse
        Fokussiert auf die besten Fast/Slow HMA-Kombinationen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für TRENDCONT-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle TRENDCONT-spezifische Analyse...")
        
        # Erstelle TRENDCONT-spezifischen Bericht
        trendcont_report_lines = []
        trendcont_report_lines.append("=" * 80)
        trendcont_report_lines.append("🎯 TRENDCONT-SPEZIFISCHE ANALYSE")
        trendcont_report_lines.append("=" * 80)
        trendcont_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trendcont_report_lines.append("")
        
        # Beste Kombinationen nach verschiedenen Metriken
        if 'sharpe_ratio' in results_df.columns:
            best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            trendcont_report_lines.append(f"🥇 Beste Kombination (Sharpe Ratio):")
            trendcont_report_lines.append(f"   Fast HMA: {int(best_sharpe['fast_hma'])}, Slow HMA: {int(best_sharpe['slow_hma'])}")
            trendcont_report_lines.append(f"   Sharpe: {best_sharpe['sharpe_ratio']:.3f} | Return: {best_sharpe['total_return']:.1%} | DD: {best_sharpe['max_drawdown']:.1%}")
            trendcont_report_lines.append(f"   Asset: {best_sharpe['asset']}")
            trendcont_report_lines.append("")
        
        if 'total_return' in results_df.columns:
            best_return = results_df.loc[results_df['total_return'].idxmax()]
            trendcont_report_lines.append(f"💰 Beste Kombination (Total Return):")
            trendcont_report_lines.append(f"   Fast HMA: {int(best_return['fast_hma'])}, Slow HMA: {int(best_return['slow_hma'])}")
            trendcont_report_lines.append(f"   Return: {best_return['total_return']:.1%} | Sharpe: {best_return['sharpe_ratio']:.3f} | DD: {best_return['max_drawdown']:.1%}")
            trendcont_report_lines.append(f"   Asset: {best_return['asset']}")
            trendcont_report_lines.append("")
        
        # Durchschnitts-Performance nach Fast HMA
        trendcont_report_lines.append("📈 Durchschnitts-Performance nach Fast HMA:")
        fast_hma_avg = results_df.groupby('fast_hma').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean'
        }).round(3)
        
        top_5_fast = fast_hma_avg.nlargest(5, 'sharpe_ratio')
        for idx, row in top_5_fast.iterrows():
            trendcont_report_lines.append(f"   Fast HMA {int(idx)}: Sharpe {row['sharpe_ratio']:.3f} | Return {row['total_return']:.1%} | DD {row['max_drawdown']:.1%}")
        trendcont_report_lines.append("")
        
        # Durchschnitts-Performance nach Slow HMA
        trendcont_report_lines.append("📈 Durchschnitts-Performance nach Slow HMA:")
        slow_hma_avg = results_df.groupby('slow_hma').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean'
        }).round(3)
        
        top_5_slow = slow_hma_avg.nlargest(5, 'sharpe_ratio')
        for idx, row in top_5_slow.iterrows():
            trendcont_report_lines.append(f"   Slow HMA {int(idx)}: Sharpe {row['sharpe_ratio']:.3f} | Return {row['total_return']:.1%} | DD {row['max_drawdown']:.1%}")
        trendcont_report_lines.append("")
        
        # TRENDCONT-spezifische Empfehlungen
        trendcont_report_lines.append(f"💡 TRENDCONT STRATEGIE EMPFEHLUNGEN")
        trendcont_report_lines.append("-" * 60)
        trendcont_report_lines.append("📋 TRENDCONT LONG-ONLY STRATEGIE INSIGHTS:")
        trendcont_report_lines.append(f"   • Long-Position bei Signal=1 (klarer Uptrend)")
        trendcont_report_lines.append(f"   • Cash-Position bei Signal=-1 oder 0 (Downtrend/Neutral)")
        trendcont_report_lines.append(f"   • Dual HMA System: Fast HMA muss steigend sein")
        trendcont_report_lines.append(f"   • Neutral-Zonen filtern unsichere Übergangsphasen")
        trendcont_report_lines.append(f"   • HMA reduziert Lag und erhöht Responsiveness")
        trendcont_report_lines.append(f"   • Kürzere Fast HMA: Schnellere Trend-Erkennung")
        trendcont_report_lines.append(f"   • Längere Slow HMA: Bessere Trend-Bestätigung")
        trendcont_report_lines.append(f"   • Optimal: Fast HMA << Slow HMA für klare Signaltrennung")
        
        trendcont_report_lines.append("")
        trendcont_report_lines.append("=" * 80)
        
        # Speichere TRENDCONT-spezifischen Bericht
        trendcont_report_text = "\n".join(trendcont_report_lines)
        trendcont_report_path = os.path.join(self.results_folder, 'trendcont_specific_analysis.txt')
        
        with open(trendcont_report_path, 'w', encoding='utf-8') as f:
            f.write(trendcont_report_text)
        
        print(f"📄 TRENDCONT-spezifische Analyse gespeichert: {trendcont_report_path}")

def main():
    """
    Hauptfunktion für Trend Continuation Backtesting System
    """
    print("🚀 TREND CONTINUATION BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (Uptrend ohne Neutral)")
    print("   • Fast-HMA-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Slow-HMA-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Validation: Fast < Slow")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Trend Continuation: Dual Hull MA System")
    
    try:
        # Erstelle und führe System aus
        tc_system = TrendContinuationBacktestingSystem(max_assets=20)
        
        # Teste verschiedene HMA-Kombinationen
        fast_range = range(5, 151)
        slow_range = range(5, 151)
        
        # Führe Backtests durch
        results_df = tc_system.run_trendcont_backtests(fast_range, slow_range)
        
        if not results_df.empty:
            # Generiere Bericht
            tc_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 Trend Continuation-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {tc_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim Trend Continuation-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
