#!/usr/bin/env python3
"""
APO (Absolute Price Oscillator) BACKTESTING SYSTEM - OPTIMIZED VERSION
=====================================================================

Backtesting-System für Absolute Price Oscillator (APO) Indikator mit:
- Matrix-Analyse für Fast/Slow Period Kombinationen
- APO > 0: Long-Position | APO <= 0: Cash-Position
- Umfassende Performance-Analyse
- Automatische Heatmap- und Matrix-Generierung
- Integration mit backtesting_base.py für Code-Wiederverwendung

APO (ABSOLUTE PRICE OSCILLATOR):
- Momentum-Indikator basierend auf der Differenz zwischen zwei Moving Averages
- APO = MA(Fast) - MA(Slow) (absolute Differenz, nicht Prozent)
- Positive Werte: Aufwärtstrend (schneller MA über langsamem MA)
- Negative Werte: Abwärtstrend (schneller MA unter langsamem MA)
- Long-Signal: APO > 0 (bullischer Momentum)
- Cash-Signal: APO <= 0 (bearisher Momentum)

Autor: Optimized Backtesting Framework
Datum: 2024
"""

import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Base-Klasse
from _backtesting_base_ import BaseBacktestingSystem

class APOBacktestingSystem(BaseBacktestingSystem):
    """
    APO (Absolute Price Oscillator) Backtesting System
    
    Der APO ist ein momentum-basierter Oszillator der die absolute Differenz zwischen
    einem schnellen und einem langsamen Moving Average berechnet.
    Strategie: APO > 0 = Long-Position, APO <= 0 = Cash-Position
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "APO", **kwargs)
        self.indicator_name = "APO"
        self.strategy_description = "APO > 0: Long-Position | APO <= 0: Cash-Position"
        self.threshold = 0.0

    def calculate_apo_signals(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
        """
        Berechnet APO-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            fast_period: Schnelle Moving Average Periode
            slow_period: Langsame Moving Average Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        if len(data) < max(fast_period, slow_period) + 1:
            return pd.DataFrame()
        
        # Validierung: Fast Period muss kleiner als Slow Period sein
        if fast_period >= slow_period:
            return pd.DataFrame()
        
        try:
            # Berechne APO
            apo = ta.APO(data['close'].astype(float).values, 
                        fastperiod=fast_period, 
                        slowperiod=slow_period, 
                        matype=0)  # 0 = Simple Moving Average
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['apo'] = apo
            signals_df['fast_period'] = fast_period
            signals_df['slow_period'] = slow_period
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # APO Long-Only Signale (APO > 0 = Long, APO <= 0 = Cash)
            signals_df['position'] = np.where(signals_df['apo'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei APO-Berechnung (Fast:{fast_period}, Slow:{slow_period}): {e}")
            return pd.DataFrame()

    def run_apo_backtests(self, fast_period_range: range = None, slow_period_range: range = None) -> pd.DataFrame:
        """
        Führt APO-Backtests über verschiedene Fast/Slow Period Kombinationen durch
        
        Args:
            fast_period_range: Range der Fast-Perioden zum Testen
            slow_period_range: Range der Slow-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        if fast_period_range is None:
            fast_period_range = range(2, 151)  # Fast: 2 bis 150
        
        if slow_period_range is None:
            slow_period_range = range(5, 151)  # Slow: 5 bis 150
        
        print(f"🚀 APO BACKTESTING SYSTEM START")
        print("=" * 60)
        print(f"⚙️ KONFIGURATION:")
        print(f"   Fast-Period-Range: {fast_period_range.start} bis {fast_period_range.stop-1}")
        print(f"   Slow-Period-Range: {slow_period_range.start} bis {slow_period_range.stop-1}")
        print(f"   Max Assets: {len(self.major_assets)} (aus backtesting_majors.csv)")
        print(f"   Strategie: {self.strategy_description}")
        print(f"   Schwelle: {self.threshold}")
        print()
        
        all_results = []
        total_combinations = 0
        valid_combinations = 0
        
        # Zähle gültige Kombinationen (Fast < Slow)
        for fast_period in fast_period_range:
            for slow_period in slow_period_range:
                if fast_period < slow_period:
                    valid_combinations += 1
        
        total_combinations = valid_combinations * len(self.major_assets)
        current_combination = 0
        
        print(f"📊 Gültige APO-Kombinationen: {valid_combinations} (Fast < Slow)")
        print(f"📊 Gesamte Tests: {total_combinations}")
        print()
        
        for fast_period in fast_period_range:
            for slow_period in slow_period_range:
                # Nur gültige Kombinationen (Fast < Slow)
                if fast_period >= slow_period:
                    continue
                
                for asset in self.assets:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    # Statische Progress-Anzeige
                    print(f"\r📊 Teste APO Fast({fast_period:3d}) / Slow({slow_period:3d}) | Progress: {progress:5.1f}% ({current_combination}/{total_combinations})     ", end='', flush=True)
                    
                    if asset not in self.assets_data:
                        continue
                    
                    # Berechne Signale für diese APO-Kombination
                    signals_df = self.calculate_apo_signals(
                        self.assets_data[asset], 
                        fast_period=fast_period, 
                        slow_period=slow_period
                    )
                    
                    if signals_df.empty:
                        continue
                    
                    # Stelle sicher, dass die erwarteten Spalten existieren
                    if 'strategy_returns' not in signals_df.columns:
                        continue
                    
                    # Berechne Performance-Metriken
                    metrics = self.calculate_performance_metrics(signals_df['strategy_returns'])
                    
                    if metrics:
                        metrics.update({
                            'asset': asset,
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'apo_combination': f"{fast_period}-{slow_period}",
                            'period_ratio': slow_period / fast_period
                        })
                        all_results.append(metrics)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        results_df = pd.DataFrame(all_results)
        print(f"✅ APO Matrix Backtests abgeschlossen: {len(all_results)} Kombinationen getestet")
        print(f"📊 Matrix-Dimensionen: {valid_combinations} (gültige Fast/Slow) × {len(self.assets_data)} (Assets)")
        
        return results_df

    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden APO-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        print(f"\n📋 Generiere umfassenden APO Bericht...")
        
        # Erstelle Heatmaps für verschiedene Metriken
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio', 'calmar_ratio', 'total_return']
        
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                try:
                    self._create_apo_heatmap(results_df, metric)
                except Exception as e:
                    print(f"⚠️ Fehler beim Erstellen der Heatmap für {metric}: {e}")
        
        # Generiere Text-Bericht
        self._generate_apo_combinations_report(results_df)
        
        # Generiere APO-spezifische Analyse
        self.generate_apo_specific_analysis(results_df)
        
        # Speichere CSV-Daten
        csv_path = os.path.join(self.results_folder, 'all_apo_combinations_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"📊 CSV-Daten gespeichert: {csv_path}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ APO BACKTESTING ABGESCHLOSSEN")
        print("=" * 60)
        print(f"📊 {len(results_df)} Konfigurationen getestet")
        if not results_df.empty:
            print(f"🥇 Beste Sharpe Ratio: {results_df['sharpe_ratio'].max():.3f}")
            print(f"💰 Höchster Return: {results_df['total_return'].max():.1%}")
        print(f"📁 Alle Ergebnisse in: {self.results_folder}/")

    def _create_apo_heatmap(self, results_df: pd.DataFrame, metric: str):
        """
        Erstellt APO-spezifische Heatmap (Fast vs Slow Period)
        """
        # Erstelle Pivot-Tabelle für Heatmap (Fast vs Slow)
        pivot_df = results_df.pivot_table(
            values=metric, 
            index='fast_period', 
            columns='slow_period', 
            aggfunc='mean'
        )
        
        if pivot_df.empty:
            return
        
        plt.figure(figsize=(16, 12))
        
        # Farbschema basierend auf Metrik
        cmap = 'RdYlGn' if 'ratio' in metric.lower() or 'return' in metric.lower() else 'viridis'
        
        # Bestimme robuste Farbgrenzen
        vmin_robust = pivot_df.quantile(0.05).min()
        vmax_robust = pivot_df.quantile(0.95).max()
        center_value = pivot_df.median().median()
        
        sns.heatmap(
            pivot_df,
            annot=False,
            cmap=cmap,
            center=center_value if 'ratio' in metric.lower() else None,
            vmin=vmin_robust,
            vmax=vmax_robust,
            cbar_kws={'label': f'Average {metric.replace("_", " ").title()}'},
            xticklabels=10,  # Zeige jeden 10. Tick
            yticklabels=10
        )
        
        plt.title(f'APO Strategy Performance Heatmap\n{metric.replace("_", " ").title()} by Fast Period (Y) vs Slow Period (X)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Slow Period', fontsize=12, fontweight='bold')
        plt.ylabel('Fast Period', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Speichere Heatmap
        heatmap_path = os.path.join(self.results_folder, f'apo_heatmap_{metric}.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 APO Heatmap gespeichert: {heatmap_path}")

    def _generate_apo_combinations_report(self, results_df: pd.DataFrame):
        """
        Generiert detaillierten Bericht für APO-Kombinationen
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 APO (ABSOLUTE PRICE OSCILLATOR) KOMBINATIONEN ANALYSE")
        report_lines.append("=" * 80)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📋 Getestete Kombinationen: {len(results_df)}")
        report_lines.append(f"💎 Assets: {results_df['asset'].nunique()}")
        report_lines.append(f"🔢 APO-Kombinationen: {results_df['apo_combination'].nunique()}")
        report_lines.append("")
        
        # Top 10 beste Kombinationen nach Sharpe Ratio
        if 'sharpe_ratio' in results_df.columns:
            top_sharpe = results_df.nlargest(10, 'sharpe_ratio')
            report_lines.append("🥇 TOP 10 BESTE APO-KOMBINATIONEN (Sharpe Ratio):")
            report_lines.append("-" * 60)
            for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
                report_lines.append(
                    f"{i:2d}. Fast({int(row['fast_period']):2d}) / Slow({int(row['slow_period']):2d}) | "
                    f"{row['asset']:4s} | Sharpe: {row['sharpe_ratio']:.3f} | "
                    f"Return: {row['total_return']:.1%} | DD: {row['max_drawdown']:.1%}"
                )
        
        # Beste durchschnittliche Kombinationen
        avg_performance = results_df.groupby(['fast_period', 'slow_period']).agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'sortino_ratio': 'mean',
            'win_rate': 'mean',
            'period_ratio': 'first'
        }).reset_index()
        
        avg_performance['combined_score'] = (
            avg_performance['sharpe_ratio'] * 0.3 +
            avg_performance['sortino_ratio'] * 0.3 +
            avg_performance['total_return'] * 0.2 +
            (1 - avg_performance['max_drawdown']) * 0.2
        )
        
        top_avg_combinations = avg_performance.nlargest(10, 'combined_score')
        
        report_lines.append("")
        report_lines.append("🎯 TOP 10 BESTE DURCHSCHNITTS-KOMBINATIONEN:")
        report_lines.append("-" * 60)
        for i, (_, row) in enumerate(top_avg_combinations.iterrows(), 1):
            report_lines.append(
                f"{i:2d}. Fast({int(row['fast_period']):2d}) / Slow({int(row['slow_period']):2d}) | "
                f"Ratio: {row['period_ratio']:.1f} | Avg Sharpe: {row['sharpe_ratio']:.3f} | "
                f"Avg Return: {row['total_return']:.1%} | Score: {row['combined_score']:.3f}"
            )
        
        # Analyse verschiedener Period-Ratios
        report_lines.append("")
        report_lines.append("⚖️ PERIODE-RATIO ANALYSE:")
        report_lines.append("-" * 60)
        
        # Klassifiziere Ratios
        small_ratio = avg_performance[avg_performance['period_ratio'] <= 2.0]  # z.B. 12/26 = 2.17
        medium_ratio = avg_performance[(avg_performance['period_ratio'] > 2.0) & 
                                     (avg_performance['period_ratio'] <= 4.0)]
        large_ratio = avg_performance[avg_performance['period_ratio'] > 4.0]
        
        for ratio_type, data, description in [
            ("KLEINE RATIOS (≤2.0)", small_ratio, "Ähnliche Perioden"),
            ("MITTLERE RATIOS (2.0-4.0)", medium_ratio, "Standard-Verhältnisse"),
            ("GROSSE RATIOS (>4.0)", large_ratio, "Große Unterschiede")
        ]:
            if not data.empty:
                best_in_category = data.loc[data['combined_score'].idxmax()]
                report_lines.append(f"📊 {ratio_type} ({description}):")
                report_lines.append(f"   • Anzahl: {len(data)} Kombinationen")
                report_lines.append(f"   • Avg Sharpe: {data['sharpe_ratio'].mean():.3f}")
                report_lines.append(f"   • Avg Return: {data['total_return'].mean():.1%}")
                report_lines.append(
                    f"   • Beste: Fast({int(best_in_category['fast_period'])}) / "
                    f"Slow({int(best_in_category['slow_period'])}) "
                    f"(Score: {best_in_category['combined_score']:.3f})"
                )
                report_lines.append("")
        
        # Spezielle APO-Kalibrierungen analysieren
        special_combos = [(12, 26), (9, 21), (5, 20), (8, 34), (21, 55), (10, 30)]
        
        report_lines.append("🎪 SPEZIELLE APO-KALIBRIERUNGEN ANALYSE:")
        report_lines.append("-" * 60)
        
        for fast, slow in special_combos:
            combo_data = results_df[
                (results_df['fast_period'] == fast) & 
                (results_df['slow_period'] == slow)
            ]
            if not combo_data.empty:
                avg_sharpe = combo_data['sharpe_ratio'].mean()
                avg_return = combo_data['total_return'].mean()
                avg_dd = combo_data['max_drawdown'].mean()
                ratio = slow / fast
                report_lines.append(
                    f"• Fast({int(fast):2d}) / Slow({int(slow):2d}) (Ratio {ratio:.1f}): "
                    f"Sharpe {avg_sharpe:.3f} | Return {avg_return:.1%} | DD {avg_dd:.1%}"
                )
        
        report_lines.append("")
        report_lines.append("=" * 68)
        report_lines.append("🏁 APO-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        report_lines.append("=" * 68)
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_folder, 'comprehensive_apo_combinations_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 APO Bericht gespeichert: {report_path}")

    def generate_apo_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert APO-spezifische Analyse ähnlich der CCI-Analyse
        Fokussiert auf die besten APO-Kombinationen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für APO-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle APO-spezifische Analyse...")
        
        # Finde beste APO-Kombinationen
        avg_performance = results_df.groupby(['fast_period', 'slow_period']).agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'sortino_ratio': 'mean',
            'win_rate': 'mean',
            'omega_ratio': 'mean',
            'period_ratio': 'first'
        }).reset_index()
        
        # Berechne kombinierten Score
        avg_performance['combined_score'] = (
            avg_performance['sharpe_ratio'] * 0.3 +
            avg_performance['sortino_ratio'] * 0.3 +
            avg_performance['total_return'] * 0.2 +
            (1 - avg_performance['max_drawdown']) * 0.2
        )
        
        # Erstelle APO-Bericht
        apo_report_lines = []
        apo_report_lines.append("=" * 80)
        apo_report_lines.append("🎯 APO-SPEZIFISCHE ANALYSE")
        apo_report_lines.append("=" * 80)
        apo_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        apo_report_lines.append("")
        
        # Beste Kombination gesamt
        best_combination = avg_performance.loc[avg_performance['combined_score'].idxmax()]
        apo_report_lines.append(f"🥇 BESTE APO-KOMBINATION (Kombinierter Score):")
        apo_report_lines.append(f"   Fast: {int(best_combination['fast_period'])} / Slow: {int(best_combination['slow_period'])}")
        apo_report_lines.append(f"   Periode-Ratio: {best_combination['period_ratio']:.2f}")
        apo_report_lines.append(f"   📊 Avg Sharpe: {best_combination['sharpe_ratio']:.3f}")
        apo_report_lines.append(f"   💰 Avg Return: {best_combination['total_return']:.1%}")
        apo_report_lines.append(f"   🛡️ Avg Drawdown: {best_combination['max_drawdown']:.1%}")
        apo_report_lines.append(f"   📈 Avg Sortino: {best_combination['sortino_ratio']:.3f}")
        apo_report_lines.append(f"   🎯 Combined Score: {best_combination['combined_score']:.3f}")
        
        # Beste für einzelne Metriken
        apo_report_lines.append("")
        apo_report_lines.append("📈 BESTE KOMBINATIONEN NACH METRIKEN:")
        
        metrics = [
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('total_return', 'Total Return'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('omega_ratio', 'Omega Ratio')
        ]
        
        for metric, name in metrics:
            if metric in avg_performance.columns:
                best = avg_performance.loc[avg_performance[metric].idxmax()]
                value = best[metric]
                value_str = f"{value:.3f}" if 'ratio' in metric.lower() else f"{value:.1%}"
                apo_report_lines.append(
                    f"   • {name}: Fast({int(best['fast_period'])}) / Slow({int(best['slow_period'])}) "
                    f"(Ø {value_str})"
                )
        
        # Niedrigster Drawdown
        best_dd = avg_performance.loc[avg_performance['max_drawdown'].idxmin()]
        apo_report_lines.append(
            f"   🛡️ Niedrigster Drawdown: Fast({int(best_dd['fast_period'])}) / Slow({int(best_dd['slow_period'])}) "
            f"(Ø {best_dd['max_drawdown']:.1%})"
        )
        
        # Top 5 Kombinationen
        top_5 = avg_performance.nlargest(5, 'combined_score')
        apo_report_lines.append(f"\n🏆 TOP 5 APO-KOMBINATIONEN:")
        apo_report_lines.append("-" * 60)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            apo_report_lines.append(
                f"{i}. Fast({int(row['fast_period'])}) / Slow({int(row['slow_period'])}) | "
                f"Ratio: {row['period_ratio']:.1f} | Score: {row['combined_score']:.3f} | "
                f"Sharpe: {row['sharpe_ratio']:.3f} | Return: {row['total_return']:.1%}"
            )
        
        # Strategische Empfehlungen
        apo_report_lines.append(f"\n💡 APO STRATEGIE EMPFEHLUNGEN")
        apo_report_lines.append("-" * 60)
        apo_report_lines.append("📋 APO LONG-ONLY STRATEGIE INSIGHTS:")
        apo_report_lines.append(f"   • Long-Position wenn APO > 0 (schneller MA über langsamem MA)")
        apo_report_lines.append(f"   • Cash-Position wenn APO <= 0 (schneller MA unter langsamem MA)")
        apo_report_lines.append(f"   • APO misst absolute Differenz zwischen Moving Averages")
        apo_report_lines.append(f"   • Kleinere Ratios (1.5-2.5): Schnelle Signale, mehr Trades")
        apo_report_lines.append(f"   • Mittlere Ratios (2.5-4.0): Ausgewogene Signale")
        apo_report_lines.append(f"   • Größere Ratios (>4.0): Langfristige Trends, weniger Trades")
        apo_report_lines.append(f"   • Klassische Kombinationen: 12/26, 9/21, 5/20")
        apo_report_lines.append(f"   • APO reagiert schneller als MACD (absolute vs. prozentuale Differenz)")
        
        apo_report_lines.append(f"\n" + "=" * 68)
        apo_report_lines.append(f"🏁 APO-ANALYSE ABGESCHLOSSEN")
        apo_report_lines.append("=" * 68)
        
        # Speichere APO-spezifischen Bericht
        apo_report_text = "\n".join(apo_report_lines)
        apo_report_path = os.path.join(self.results_folder, 'apo_specific_analysis.txt')
        
        with open(apo_report_path, 'w', encoding='utf-8') as f:
            f.write(apo_report_text)
        
        print(f"📄 APO-spezifische Analyse gespeichert: {apo_report_path}")

def main():
    """
    Hauptfunktion für APO Backtesting System
    """
    print("🚀 APO BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (APO > 0)")
    print("   • Fast-Range: 2 bis 150 (Einser-Schritte)")
    print("   • Slow-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Bedingung: Fast < Slow (nur gültige Kombinationen)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe APO Backtesting System aus
        apo_system = APOBacktestingSystem(max_assets=20)
        
        # Teste verschiedene Fast/Slow Kombinationen
        fast_range = range(2, 151)  # Fast: 2 bis 150
        slow_range = range(5, 151)  # Slow: 5 bis 150
        
        # Berechne gültige Kombinationen
        valid_combos = sum(1 for fast in fast_range for slow in slow_range if fast < slow)
        total_tests = valid_combos * 20  # 20 Assets
        
        print(f"\n📊 APO MATRIX ANALYSE:")
        print(f"   Fast-Range: {fast_range.start}-{fast_range.stop-1}")
        print(f"   Slow-Range: {slow_range.start}-{slow_range.stop-1}")
        print(f"   Gültige Kombinationen: {valid_combos} (Fast < Slow)")
        print(f"   Gesamt-Tests: {total_tests:,}")
        print(f"   ⚠️  Geschätzte Laufzeit: 5-8 Stunden")
        print()
        
        # Führe Backtests durch
        results_df = apo_system.run_apo_backtests(
            fast_period_range=fast_range,
            slow_period_range=slow_range
        )
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            apo_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 APO-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {apo_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim APO-Backtesting: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()