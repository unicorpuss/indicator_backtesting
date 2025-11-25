#!/usr/bin/env python3
"""
PPO (Percentage Price Oscillator) BACKTESTING SYSTEM - OPTIMIZED VERSION
=========================================================================

Backtesting-System für Percentage Price Oscillator (PPO) Indikator mit:
- Dual-Parameter Matrix-Analyse (Fast Period vs Slow Period)
- PPO > 0: Long-Position | PPO <= 0: Cash-Position
- Matrix-basierte Optimierung über verschiedene Fast/Slow-Kombinationen
- Umfassende Performance-Analyse mit Heatmaps
- Integration mit backtesting_base.py für Code-Wiederverwendung

PPO (Percentage Price Oscillator):
- Ähnlich dem MACD, aber in Prozent ausgedrückt
- PPO = ((EMA_fast - EMA_slow) / EMA_slow) * 100
- Normalisiert die Unterschiede zwischen EMAs
- > 0 = Fast EMA über Slow EMA (Bullish)
- < 0 = Fast EMA unter Slow EMA (Bearish)
- Unabhängig vom Preisniveau durch Prozentberechnung

Autor: Optimized Backtesting Framework
Datum: 2024
"""

import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import der gemeinsamen Backtesting-Funktionen
from _backtesting_base_ import BaseBacktestingSystem

class PPOBacktestingSystem(BaseBacktestingSystem):
    """
    Percentage Price Oscillator (PPO) Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    Besonderheit: Zwei Parameter (Fast Period und Slow Period) mit Matrix-Analyse
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "PPO", **kwargs)
        self.indicator_name = "PPO"
        self.strategy_description = "PPO > 0: Long-Position | PPO <= 0: Cash-Position"
        self.threshold = 0.0  # Mittellinie bei 0
    
    def calculate_ppo_signals(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
        """
        Berechnet PPO-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            fast_period: Schnelle EMA-Periode
            slow_period: Langsame EMA-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        min_length = max(fast_period, slow_period)
        if len(data) < min_length + 1:
            return pd.DataFrame()
        
        # Validierung: Fast Period muss kleiner als Slow Period sein
        if fast_period >= slow_period:
            return pd.DataFrame()
        
        try:
            # Berechne PPO mit talib
            ppo = ta.PPO(data['close'].astype(float).values, 
                        fastperiod=fast_period, 
                        slowperiod=slow_period, 
                        matype=0)  # Simple Moving Average
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['ppo'] = ppo
            signals_df['fast_period'] = fast_period
            signals_df['slow_period'] = slow_period
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # PPO Long-Only Signale (PPO > 0 = Long, PPO <= 0 = Cash)
            signals_df['position'] = np.where(signals_df['ppo'] > 0, 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei PPO-Berechnung (Fast:{fast_period}, Slow:{slow_period}): {e}")
            return pd.DataFrame()
    
    def run_ppo_backtests(self, fast_range: range = None, slow_range: range = None) -> pd.DataFrame:
        """
        Führt PPO-Backtests über vollständige Matrix verschiedener Fast/Slow-Kombinationen durch
        
        Args:
            fast_range: Range der Fast-Perioden zum Testen (Standard: 5-150)
            slow_range: Range der Slow-Perioden zum Testen (Standard: 5-150)
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        if fast_range is None:
            fast_range = range(5, 151)  # Fast: 5 bis 150 (Einser-Schritte)
        
        if slow_range is None:
            slow_range = range(5, 151)  # Slow: 5 bis 150 (Einser-Schritte)
        
        print(f"🚀 PPO BACKTESTING SYSTEM START")
        print("=" * 60)
        print(f"⚙️ KONFIGURATION:")
        print(f"   Fast-Range: {fast_range.start} bis {fast_range.stop-1} (Einser-Schritte)")
        print(f"   Slow-Range: {slow_range.start} bis {slow_range.stop-1} (Einser-Schritte)")
        print(f"   Validation: Fast < Slow (nur gültige Kombinationen)")
        
        # Berechne gültige Kombinationen (Fast < Slow)
        valid_combinations = 0
        for fast in fast_range:
            for slow in slow_range:
                if fast < slow:
                    valid_combinations += 1
        
        print(f"   Gültige Matrix: {valid_combinations} PPO-Kombinationen")
        print(f"   Max Assets: {len(self.major_assets)} (aus backtesting_majors.csv)")
        print(f"   Strategie: PPO > 0: Long-Position | PPO <= 0: Cash-Position")
        print(f"   Schwelle: {self.threshold}")
        print()
        
        all_results = []
        total_combinations = valid_combinations * len(self.major_assets)
        current_combination = 0
        
        for fast_period in fast_range:
            for slow_period in slow_range:
                # Überspringe ungültige Kombinationen
                if fast_period >= slow_period:
                    continue
                
                for asset in self.assets:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    # Statische Progress-Anzeige
                    print(f"\r📊 Teste PPO Fast({fast_period:3d}) / Slow({slow_period:3d}) | Progress: {progress:5.1f}% ({current_combination}/{total_combinations})     ", end='', flush=True)
                    
                    if asset not in self.assets_data:
                        continue
                    
                    # Berechne Signale für diese PPO-Kombination
                    signals_df = self.calculate_ppo_signals(
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
                    metrics = self._calculate_ppo_performance_metrics(signals_df)
                    
                    if metrics:
                        metrics.update({
                            'asset': asset,
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'ppo_combination': f"{fast_period}-{slow_period}",
                            'period_ratio': round(slow_period / fast_period, 2)
                        })
                        all_results.append(metrics)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        results_df = pd.DataFrame(all_results)
        print(f"✅ PPO Matrix Backtests abgeschlossen: {len(all_results)} Kombinationen getestet")
        print(f"📊 Matrix-Dimensionen: {len(fast_range)} (Fast) × {len(slow_range)} (Slow) = {valid_combinations} gültige Kombinationen × {len(self.assets_data)} (Assets)")
        
        return results_df
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden PPO-Bericht mit allen Analysen für verschiedene Kalibrierungen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        print(f"\n📋 Generiere umfassenden PPO Bericht...")
        
        # Erstelle Heatmaps für verschiedene Metriken
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio', 'calmar_ratio', 'total_return']
        
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                try:
                    # Erstelle Pivot-Tabelle für Heatmap (Fast vs Slow)
                    pivot_df = results_df.pivot_table(
                        values=metric, 
                        index='fast_period', 
                        columns='slow_period', 
                        aggfunc='mean'
                    )
                    
                    if not pivot_df.empty:
                        plt.figure(figsize=(16, 12))
                        
                        # Bestimme Farbbereich basierend auf Daten
                        vmin_robust = pivot_df.quantile(0.05).min()
                        vmax_robust = pivot_df.quantile(0.95).max()
                        center_value = pivot_df.median().median()
                        
                        # Erstelle Heatmap
                        sns.heatmap(pivot_df, 
                                  annot=False, 
                                  cmap='RdYlGn', 
                                  center=center_value,
                                  vmin=vmin_robust, 
                                  vmax=vmax_robust,
                                  cbar_kws={'label': metric.replace('_', ' ').title()})
                        
                        plt.title(f'PPO {metric.replace("_", " ").title()} Heatmap\n'
                                f'Fast Period (Y-Axis) vs Slow Period (X-Axis)', 
                                fontsize=14, fontweight='bold')
                        plt.xlabel('Slow Period', fontsize=12)
                        plt.ylabel('Fast Period', fontsize=12)
                        
                        # Speichere Heatmap
                        heatmap_path = os.path.join(self.results_folder, f'ppo_heatmap_{metric}.png')
                        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"📊 PPO Heatmap gespeichert: {heatmap_path}")
                        
                except Exception as e:
                    print(f"⚠️ Fehler beim Erstellen der Heatmap für {metric}: {e}")
        
        # Generiere Text-Bericht
        self._generate_ppo_combinations_report(results_df)
        
        # Speichere CSV-Daten
        csv_path = os.path.join(self.results_folder, 'all_ppo_combinations_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"📊 CSV-Daten gespeichert: {csv_path}")
        
        print(f"📄 PPO Bericht gespeichert: {os.path.join(self.results_folder, 'comprehensive_ppo_combinations_report.txt')}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ PPO BACKTESTING ABGESCHLOSSEN")
        print("=" * 60)
        print(f"📊 {len(results_df)} Konfigurationen getestet")
        if not results_df.empty:
            print(f"🥇 Beste Sharpe Ratio: {results_df['sharpe_ratio'].max():.3f}")
            print(f"💰 Höchster Return: {results_df['total_return'].max():.1%}")
        print(f"📁 Alle Ergebnisse in: {self.results_folder}/")
        
        # Zusätzliche PPO-spezifische Analyse
        print(f"\n📋 Erstelle PPO-spezifische Analyse...")
        self.generate_ppo_specific_analysis(results_df)
    
    def _generate_ppo_combinations_report(self, results_df: pd.DataFrame):
        """
        Generiert detaillierten Bericht für PPO-Kombinationen
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 PPO KOMBINATIONEN ANALYSE")
        report_lines.append("=" * 80)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📋 Getestete Kombinationen: {len(results_df)}")
        report_lines.append(f"💎 Assets: {results_df['asset'].nunique()}")
        report_lines.append(f"🔢 PPO-Kombinationen: {results_df['ppo_combination'].nunique()}")
        report_lines.append("")
        
        # Top 10 beste Kombinationen nach Sharpe Ratio
        if 'sharpe_ratio' in results_df.columns:
            top_sharpe = results_df.nlargest(10, 'sharpe_ratio')
            report_lines.append("🥇 TOP 10 BESTE PPO-KOMBINATIONEN (Sharpe Ratio):")
            report_lines.append("-" * 60)
            for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
                report_lines.append(
                    f"{i:2d}. Fast({int(row['fast_period']):2d}) / Slow({int(row['slow_period']):2d}) | "
                    f"{row['asset']:4s} | Sharpe: {row['sharpe_ratio']:.3f} | "
                    f"Return: {row['total_return']:.1%} | DD: {row['max_drawdown']:.1%} | "
                    f"Ratio: {row['period_ratio']:.1f}"
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
                f"Avg Sharpe: {row['sharpe_ratio']:.3f} | "
                f"Avg Return: {row['total_return']:.1%} | Ratio: {row['period_ratio']:.1f} | "
                f"Score: {row['combined_score']:.3f}"
            )
        
        # Analyse verschiedener Ratio-Bereiche
        report_lines.append("")
        report_lines.append("⚖️ ANALYSE NACH PERIOD-RATIOS:")
        report_lines.append("-" * 60)
        
        ratio_ranges = [
            (1.0, 2.0, "Enge Ratios (1.0-2.0)"),
            (2.0, 3.0, "Moderate Ratios (2.0-3.0)"),
            (3.0, 5.0, "Weite Ratios (3.0-5.0)"),
            (5.0, 10.0, "Sehr weite Ratios (5.0-10.0)"),
            (10.0, float('inf'), "Extreme Ratios (>10.0)")
        ]
        
        for min_ratio, max_ratio, description in ratio_ranges:
            if max_ratio == float('inf'):
                ratio_data = avg_performance[avg_performance['period_ratio'] >= min_ratio]
            else:
                ratio_data = avg_performance[
                    (avg_performance['period_ratio'] >= min_ratio) & 
                    (avg_performance['period_ratio'] < max_ratio)
                ]
            
            if not ratio_data.empty:
                avg_sharpe = ratio_data['sharpe_ratio'].mean()
                avg_return = ratio_data['total_return'].mean()
                avg_dd = ratio_data['max_drawdown'].mean()
                count = len(ratio_data)
                report_lines.append(
                    f"📊 {description}: {count} Kombinationen"
                )
                report_lines.append(
                    f"   • Avg Sharpe: {avg_sharpe:.3f} | Avg Return: {avg_return:.1%} | Avg DD: {avg_dd:.1%}"
                )
        
        # Klassische PPO-Kalibrierungen
        report_lines.append("")
        report_lines.append("🎪 KLASSISCHE PPO-KALIBRIERUNGEN:")
        report_lines.append("-" * 60)
        
        classic_combos = [(12, 26), (8, 21), (5, 35), (9, 18), (14, 30)]
        
        for fast, slow in classic_combos:
            combo_data = avg_performance[
                (avg_performance['fast_period'] == fast) & 
                (avg_performance['slow_period'] == slow)
            ]
            if not combo_data.empty:
                row = combo_data.iloc[0]
                report_lines.append(
                    f"• Fast({int(fast):2d}) / Slow({int(slow):2d}): "
                    f"Sharpe {row['sharpe_ratio']:.3f} | Return {row['total_return']:.1%} | "
                    f"DD {row['max_drawdown']:.1%} | Ratio {row['period_ratio']:.1f}"
                )
        
        report_lines.append("")
        report_lines.append("=" * 68)
        report_lines.append("🏁 PPO-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        report_lines.append("=" * 68)
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_folder, 'comprehensive_ppo_combinations_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def _calculate_ppo_performance_metrics(self, signals_df: pd.DataFrame) -> Dict:
        """
        Berechnet Performance-Metriken für PPO-Signale
        Wiederverwendung der bewährten Berechnungsmethode
        
        Args:
            signals_df: DataFrame mit Signalen und Returns
            
        Returns:
            Dictionary mit Performance-Metriken
        """
        if signals_df.empty or 'strategy_returns' not in signals_df.columns:
            return {}
        
        try:
            returns = signals_df['strategy_returns'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # Basis-Metriken - korrigierte Berechnung
            returns_cleaned = returns[(returns > -0.5) & (returns < 1.0)]
            
            if len(returns_cleaned) == 0:
                return {}
            
            # Total Return korrekt berechnen
            cumulative_returns = (1 + returns_cleaned).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1
            
            # Begrenze unrealistische Returns
            total_return = min(total_return, 10.0)  # Max 1000% Return
            total_return = max(total_return, -0.99)  # Max -99% Verlust
            
            # Annualisierte Metriken
            num_periods = len(returns_cleaned)
            if num_periods > 252:  # Mehr als ein Jahr Daten
                periods_per_year = 252
                annualized_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
            else:
                annualized_return = total_return
            
            volatility = returns_cleaned.std() * np.sqrt(252)
            
            # Sharpe Ratio
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Downside-Metriken
            downside_returns = returns_cleaned[returns_cleaned < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Win Rate
            win_rate = len(returns_cleaned[returns_cleaned > 0]) / len(returns_cleaned) if len(returns_cleaned) > 0 else 0
            
            # Omega Ratio (vereinfacht)
            positive_returns = returns_cleaned[returns_cleaned > 0].sum()
            negative_returns = abs(returns_cleaned[returns_cleaned < 0].sum())
            omega_ratio = positive_returns / negative_returns if negative_returns > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Zusätzliche Validierung
            if abs(sharpe_ratio) > 10:  # Unrealistisch hohe Sharpe Ratio
                sharpe_ratio = 0
            if abs(sortino_ratio) > 15:  # Unrealistisch hohe Sortino Ratio
                sortino_ratio = 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'omega_ratio': omega_ratio,
                'calmar_ratio': calmar_ratio,
                'num_trades': len(returns_cleaned),
                'avg_return': returns_cleaned.mean(),
                'std_return': returns_cleaned.std()
            }
            
        except Exception as e:
            print(f"Fehler bei Performance-Berechnung: {e}")
            return {}
    
    def generate_ppo_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert PPO-spezifische Analyse für verschiedene Kalibrierungen
        Fokussiert auf die besten PPO-Kombinationen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für PPO-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle PPO-spezifische Analyse...")
        
        # Finde beste PPO-Kombinationen
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
        
        # Erstelle PPO-Bericht
        ppo_report_lines = []
        ppo_report_lines.append("=" * 80)
        ppo_report_lines.append("🎯 PPO KOMBINATIONEN-SPEZIFISCHE ANALYSE")
        ppo_report_lines.append("=" * 80)
        ppo_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ppo_report_lines.append("")
        
        # Beste Kombination gesamt
        best_combination = avg_performance.loc[avg_performance['combined_score'].idxmax()]
        ppo_report_lines.append(f"🥇 BESTE PPO-KOMBINATION (Kombinierter Score):")
        ppo_report_lines.append(f"   Fast: {int(best_combination['fast_period'])} / Slow: {int(best_combination['slow_period'])} (Ratio: {best_combination['period_ratio']:.1f})")
        ppo_report_lines.append(f"   📊 Avg Sharpe: {best_combination['sharpe_ratio']:.3f}")
        ppo_report_lines.append(f"   💰 Avg Return: {best_combination['total_return']:.1%}")
        ppo_report_lines.append(f"   🛡️ Avg Drawdown: {best_combination['max_drawdown']:.1%}")
        ppo_report_lines.append(f"   📈 Avg Sortino: {best_combination['sortino_ratio']:.3f}")
        ppo_report_lines.append(f"   🎯 Combined Score: {best_combination['combined_score']:.3f}")
        
        # Beste für einzelne Metriken
        ppo_report_lines.append("")
        ppo_report_lines.append("📈 BESTE KOMBINATIONEN NACH METRIKEN:")
        
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
                ppo_report_lines.append(
                    f"   • {name}: Fast({int(best['fast_period'])}) / Slow({int(best['slow_period'])}) "
                    f"(Ø {value_str}, Ratio: {best['period_ratio']:.1f})"
                )
        
        # Niedrigster Drawdown
        best_dd = avg_performance.loc[avg_performance['max_drawdown'].idxmin()]
        ppo_report_lines.append(
            f"   🛡️ Niedrigster Drawdown: Fast({int(best_dd['fast_period'])}) / Slow({int(best_dd['slow_period'])}) "
            f"(Ø {best_dd['max_drawdown']:.1%}, Ratio: {best_dd['period_ratio']:.1f})"
        )
        
        # Top 5 Kombinationen
        top_5 = avg_performance.nlargest(5, 'combined_score')
        ppo_report_lines.append(f"\n🏆 TOP 5 PPO-KOMBINATIONEN:")
        ppo_report_lines.append("-" * 60)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            ppo_report_lines.append(
                f"{i}. Fast({int(row['fast_period'])}) / Slow({int(row['slow_period'])}) | "
                f"Ratio: {row['period_ratio']:.1f} | Score: {row['combined_score']:.3f} | "
                f"Sharpe: {row['sharpe_ratio']:.3f} | Return: {row['total_return']:.1%}"
            )
        
        # Strategische Empfehlungen
        ppo_report_lines.append(f"\n🎯 STRATEGISCHE EMPFEHLUNGEN:")
        ppo_report_lines.append("-" * 60)
        ppo_report_lines.append("💡 PPO KOMBINATIONS-STRATEGIE INSIGHTS:")
        ppo_report_lines.append(f"   • PPO normalisiert MACD-ähnliche Signale durch Prozentberechnung")
        ppo_report_lines.append(f"   • Ratio 2-3: Ausgewogene Balance zwischen Sensitivität und Stabilität")
        ppo_report_lines.append(f"   • Ratio >5: Langfristige Trends, weniger Fehlsignale")
        ppo_report_lines.append(f"   • Fast < 10 + Slow > 20: Responsive Signale für mittelfristige Trends")
        ppo_report_lines.append(f"   • Klassische 12/26: Bewährte Kombination aus MACD-Tradition")
        ppo_report_lines.append(f"   • PPO > 0: EMA-Fast über EMA-Slow (Bullish Momentum)")
        ppo_report_lines.append(f"   • PPO < 0: EMA-Fast unter EMA-Slow (Bearish Momentum)")
        
        ppo_report_lines.append(f"\n" + "=" * 68)
        ppo_report_lines.append(f"🏁 PPO-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        ppo_report_lines.append("=" * 68)
        
        # Speichere PPO-spezifischen Bericht
        ppo_report_text = "\n".join(ppo_report_lines)
        ppo_report_path = os.path.join(self.results_folder, 'ppo_combinations_specific_analysis.txt')
        
        with open(ppo_report_path, 'w', encoding='utf-8') as f:
            f.write(ppo_report_text)
        
        print(f"📄 PPO-Kombinationen Analyse gespeichert: {ppo_report_path}")

def main():
    """
    Hauptfunktion für PPO Backtesting System
    """
    print("🚀 PPO BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (PPO > 0)")
    print("   • Fast-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Slow-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Validation: Fast < Slow")
    print("   • Assets: Major Cryptocurrencies")
    print("   • PPO: Percentage Price Oscillator (normalisierter MACD)")
    print("   • Matrix-Analyse mit Period-Ratios")
    
    try:
        # Erstelle und führe PPO Backtesting System aus
        ppo_system = PPOBacktestingSystem(max_assets=20)
        
        # Führe Backtests mit vollständiger Matrix durch (5-150 in Einser-Schritten)
        fast_range = range(5, 151)   # Fast: 5 bis 150 (Vollständige Range)
        slow_range = range(5, 151)   # Slow: 5 bis 150 (Vollständige Range)
        
        print(f"📊 PPO-MATRIX VOLLSTÄNDIGE ANALYSE")
        print(f"   Fast-Range: {fast_range.start}-{fast_range.stop-1} (Einser-Schritte)")
        print(f"   Slow-Range: {slow_range.start}-{slow_range.stop-1} (Einser-Schritte)")
        
        # Berechne gültige Kombinationen für vollständige Range
        valid_combinations = 0
        for fast in fast_range:
            for slow in slow_range:
                if fast < slow:
                    valid_combinations += 1
        
        print(f"   Gültige Kombinationen: {valid_combinations}")
        print(f"   Gesamt-Tests: {valid_combinations * 17} (mit 17 Assets)")
        print(f"   ⚠️ WARNUNG: Vollständige Matrix = {valid_combinations * 17:,} Tests (kann lange dauern!)")
        print()
        
        results_df = ppo_system.run_ppo_backtests(fast_range=fast_range, slow_range=slow_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            ppo_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 PPO-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {ppo_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim PPO-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()