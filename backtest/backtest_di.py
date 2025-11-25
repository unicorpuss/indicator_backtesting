"""
DIRECTIONAL INDICATORS (+DI/-DI) Backtesting System (Optimiert mit Base Class)

Dieses System testet Directional Indicators-basierte Trading-Strategien über verschiedene Assets und Perioden:
- DI-Perioden von 5 bis 150
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: +DI > -DI = Long Position, sonst Cash Position
- Verwendet gemeinsame Funktionen aus backtesting_base.py

DIRECTIONAL INDICATORS (+DI/-DI):
- Momentum-Indikatoren entwickelt von J. Welles Wilder
- +DI (Plus Directional Indicator): Misst bullische Preisbewegung
- -DI (Minus Directional Indicator): Misst bearische Preisbewegung
- Berechnung basiert auf Directional Movement (DM+/DM-) und True Range
- Werte zwischen 0 und 100
- Long-Signal: +DI > -DI (bullisches Momentum dominiert)
- Cash-Signal: +DI <= -DI (bearisches Momentum dominiert)
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

class DIBacktestingSystem(BaseBacktestingSystem):
    """
    Directional Indicators (+DI/-DI) Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    Besonderheit: Zwei Indikatoren (+DI und -DI) mit einem Parameter (DI-Länge)
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "DI", **kwargs)
        self.indicator_name = "DI"
        self.strategy_description = "+DI > -DI: Long-Position | +DI <= -DI: Cash-Position"
        self.threshold = 0.0  # Kein absoluter Schwellenwert, nur Vergleich +DI vs -DI
    
    def calculate_di_signals(self, data: pd.DataFrame, plus_di_length: int = 14, minus_di_length: int = 14) -> pd.DataFrame:
        """
        Berechnet Directional Indicators (+DI/-DI) Signale für die Long-Only Strategie
        Unterstützt unterschiedliche Perioden für +DI und -DI
        
        Args:
            data: DataFrame mit OHLCV-Daten
            plus_di_length: +DI-Periode
            minus_di_length: -DI-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        min_length = max(plus_di_length, minus_di_length)
        if len(data) < min_length + 1:
            return pd.DataFrame()
        
        try:
            # Berechne +DI und -DI mit unterschiedlichen Perioden
            plus_di = ta.PLUS_DI(data['high'].astype(float).values, 
                                data['low'].astype(float).values, 
                                data['close'].astype(float).values, 
                                timeperiod=plus_di_length)
            
            minus_di = ta.MINUS_DI(data['high'].astype(float).values, 
                                  data['low'].astype(float).values, 
                                  data['close'].astype(float).values, 
                                  timeperiod=minus_di_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['plus_di'] = plus_di
            signals_df['minus_di'] = minus_di
            signals_df['plus_di_length'] = plus_di_length
            signals_df['minus_di_length'] = minus_di_length
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # DI Long-Only Signale (+DI > -DI = Long, +DI <= -DI = Cash)
            signals_df['position'] = np.where(signals_df['plus_di'] > signals_df['minus_di'], 1, 0)
            
            # Berechne DI-Differenz für zusätzliche Analyse
            signals_df['di_diff'] = signals_df['plus_di'] - signals_df['minus_di']
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei DI-Berechnung (+DI:{plus_di_length}, -DI:{minus_di_length}): {e}")
            return pd.DataFrame()
    
    def run_di_backtests(self, plus_di_range: range = None, minus_di_range: range = None) -> pd.DataFrame:
        """
        Führt DI-Backtests über vollständige Matrix verschiedener Perioden-Kombinationen durch
        
        Args:
            plus_di_range: Range der +DI-Perioden zum Testen (Standard: 5-50)
            minus_di_range: Range der -DI-Perioden zum Testen (Standard: 5-50)
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        if plus_di_range is None:
            plus_di_range = range(1, 151)  # +DI: 1 bis 150
        
        if minus_di_range is None:
            minus_di_range = range(1, 151)  # -DI: 1 bis 150
        
        print(f"🚀 DI BACKTESTING SYSTEM START")
        print("=" * 60)
        print(f"⚙️ KONFIGURATION:")
        print(f"   +DI-Range: {plus_di_range.start} bis {plus_di_range.stop-1}")
        print(f"   -DI-Range: {minus_di_range.start} bis {minus_di_range.stop-1}")
        print(f"   Vollständige Matrix: {len(plus_di_range)} × {len(minus_di_range)} = {len(plus_di_range) * len(minus_di_range)} DI-Kombinationen")
        print(f"   Max Assets: {len(self.assets)} (aus CSV)")
        print(f"   Strategie: +DI > -DI: Long-Position | +DI <= -DI: Cash-Position")
        print(f"   Schwelle: {self.threshold}")
        print()
        
        all_results = []
        total_combinations = len(plus_di_range) * len(minus_di_range) * len(self.assets_data)
        current_combination = 0
        
        for plus_di_length in plus_di_range:
            for minus_di_length in minus_di_range:
                for asset in self.assets:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    # Statische Progress-Anzeige
                    print(f"\r📊 Teste DI +DI({plus_di_length:3d}) / -DI({minus_di_length:3d}) | Progress: {progress:5.1f}% ({current_combination}/{total_combinations})     ", end='', flush=True)
                    
                    if asset not in self.assets_data:
                        continue
                    
                    # Berechne Signale für diese DI-Kombination
                    signals_df = self.calculate_di_signals(
                        self.assets_data[asset], 
                        plus_di_length=plus_di_length, 
                        minus_di_length=minus_di_length
                    )
                    
                    if signals_df.empty:
                        continue
                    
                    # Stelle sicher, dass die erwarteten Spalten existieren
                    if 'strategy_returns' not in signals_df.columns:
                        continue
                    
                    # Berechne Performance-Metriken (spezifische Implementierung)
                    metrics = self._calculate_di_performance_metrics(signals_df)
                    
                    if metrics:
                        metrics.update({
                            'asset': asset,
                            'plus_di_length': plus_di_length,
                            'minus_di_length': minus_di_length,
                            'di_combination': f"{plus_di_length}-{minus_di_length}"
                        })
                        all_results.append(metrics)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        results_df = pd.DataFrame(all_results)
        print(f"✅ DI Matrix Backtests abgeschlossen: {len(all_results)} Kombinationen getestet")
        print(f"📊 Matrix-Dimensionen: {len(plus_di_range)} (+DI) × {len(minus_di_range)} (-DI) × {len(self.assets_data)} (Assets)")
        
        return results_df
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden DI-Bericht mit allen Analysen für verschiedene Kalibrierungen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        print(f"\n📋 Generiere umfassenden DI Bericht...")
        
        # Erstelle Heatmaps für verschiedene Metriken
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio', 'calmar_ratio', 'total_return']
        
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                try:
                    # Erstelle Pivot-Tabelle für Heatmap (+DI vs -DI)
                    pivot_df = results_df.pivot_table(
                        values=metric, 
                        index='plus_di_length', 
                        columns='minus_di_length', 
                        aggfunc='mean'
                    )
                    
                    if not pivot_df.empty:
                        plt.figure(figsize=(16, 12))
                        
                        # Bestimme Farbbereich basierend auf Daten für stärkeren Kontrast
                        vmin = pivot_df.min().min()
                        vmax = pivot_df.max().max()
                        
                        # Verwende robustere Perzentile für besseren Kontrast
                        vmin_robust = pivot_df.quantile(0.05).min()
                        vmax_robust = pivot_df.quantile(0.95).max()
                        
                        # Stärkerer Farbkontrast mit RdYlGn_r (reversed) und robustem Center
                        center_value = pivot_df.median().median()
                        
                        sns.heatmap(pivot_df, 
                                  annot=False, 
                                  cmap='RdYlGn', 
                                  center=center_value,
                                  vmin=vmin_robust,
                                  vmax=vmax_robust,
                                  cbar_kws={'label': metric, 'shrink': 0.8})
                        
                        plt.title(f'DI Combinations Heatmap: {metric.replace("_", " ").title()}', 
                                fontsize=16, fontweight='bold')
                        plt.xlabel('-DI Length', fontsize=12, fontweight='bold')
                        plt.ylabel('+DI Length', fontsize=12, fontweight='bold')
                        plt.tight_layout()
                        
                        heatmap_path = os.path.join(self.results_folder, f'heatmap_{metric}.png')
                        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"📊 Heatmap gespeichert: {heatmap_path}")
                        
                except Exception as e:
                    print(f"⚠️ Fehler beim Erstellen der Heatmap für {metric}: {e}")
        
        # Generiere Text-Bericht
        self._generate_di_combinations_report(results_df)
        
        # Speichere CSV-Daten
        csv_path = os.path.join(self.results_folder, 'all_di_combinations_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"📊 CSV-Daten gespeichert: {csv_path}")
        
        print(f"📄 DI Bericht gespeichert: {os.path.join(self.results_folder, 'comprehensive_di_combinations_report.txt')}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ DI BACKTESTING ABGESCHLOSSEN")
        print("=" * 60)
        print(f"📊 {len(results_df)} Konfigurationen getestet")
        if not results_df.empty:
            print(f"🥇 Beste Sharpe Ratio: {results_df['sharpe_ratio'].max():.3f}")
            print(f"� Höchster Return: {results_df['total_return'].max():.1%}")
        print(f"📁 Alle Ergebnisse in: {self.results_folder}/")
        
        # Zusätzliche DI-spezifische Analyse
        print(f"\n📋 Erstelle DI-spezifische Analyse...")
        self.generate_di_specific_analysis(results_df)
    
    def _generate_di_combinations_report(self, results_df: pd.DataFrame):
        """
        Generiert detaillierten Bericht für DI-Kombinationen
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 DIRECTIONAL INDICATORS KOMBINATIONEN ANALYSE")
        report_lines.append("=" * 80)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📋 Getestete Kombinationen: {len(results_df)}")
        report_lines.append(f"💎 Assets: {results_df['asset'].nunique()}")
        report_lines.append(f"🔢 DI-Kombinationen: {results_df['di_combination'].nunique()}")
        report_lines.append("")
        
        # Top 10 beste Kombinationen nach Sharpe Ratio
        if 'sharpe_ratio' in results_df.columns:
            top_sharpe = results_df.nlargest(10, 'sharpe_ratio')
            report_lines.append("🥇 TOP 10 BESTE DI-KOMBINATIONEN (Sharpe Ratio):")
            report_lines.append("-" * 60)
            for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
                report_lines.append(
                    f"{i:2d}. +DI({int(row['plus_di_length']):2d}) / -DI({int(row['minus_di_length']):2d}) | "
                    f"{row['asset']:4s} | Sharpe: {row['sharpe_ratio']:.3f} | "
                    f"Return: {row['total_return']:.1%} | DD: {row['max_drawdown']:.1%}"
                )
        
        # Beste durchschnittliche Kombinationen
        avg_performance = results_df.groupby(['plus_di_length', 'minus_di_length']).agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'sortino_ratio': 'mean',
            'win_rate': 'mean'
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
                f"{i:2d}. +DI({int(row['plus_di_length']):2d}) / -DI({int(row['minus_di_length']):2d}) | "
                f"Avg Sharpe: {row['sharpe_ratio']:.3f} | "
                f"Avg Return: {row['total_return']:.1%} | Score: {row['combined_score']:.3f}"
            )
        
        # Analyse gleicher vs verschiedener Perioden
        same_periods = results_df[results_df['plus_di_length'] == results_df['minus_di_length']]
        diff_periods = results_df[results_df['plus_di_length'] != results_df['minus_di_length']]
        
        report_lines.append("")
        report_lines.append("⚖️ VERGLEICH: GLEICHE vs VERSCHIEDENE PERIODEN:")
        report_lines.append("-" * 60)
        
        if not same_periods.empty and not diff_periods.empty:
            report_lines.append(f"📊 Gleiche Perioden (+DI = -DI):")
            report_lines.append(f"   • Anzahl: {len(same_periods)} Kombinationen")
            report_lines.append(f"   • Avg Sharpe: {same_periods['sharpe_ratio'].mean():.3f}")
            report_lines.append(f"   • Avg Return: {same_periods['total_return'].mean():.1%}")
            report_lines.append(f"   • Avg Drawdown: {same_periods['max_drawdown'].mean():.1%}")
            
            report_lines.append(f"📊 Verschiedene Perioden (+DI ≠ -DI):")
            report_lines.append(f"   • Anzahl: {len(diff_periods)} Kombinationen")
            report_lines.append(f"   • Avg Sharpe: {diff_periods['sharpe_ratio'].mean():.3f}")
            report_lines.append(f"   • Avg Return: {diff_periods['total_return'].mean():.1%}")
            report_lines.append(f"   • Avg Drawdown: {diff_periods['max_drawdown'].mean():.1%}")
        
        # Spezielle Kalibrierungen analysieren
        special_combos = [(12, 50), (50, 12), (30, 21), (21, 30), (14, 28), (28, 14)]
        
        report_lines.append("")
        report_lines.append("🎪 SPEZIELLE KALIBRIERUNGEN ANALYSE:")
        report_lines.append("-" * 60)
        
        for plus_di, minus_di in special_combos:
            combo_data = results_df[
                (results_df['plus_di_length'] == plus_di) & 
                (results_df['minus_di_length'] == minus_di)
            ]
            if not combo_data.empty:
                avg_sharpe = combo_data['sharpe_ratio'].mean()
                avg_return = combo_data['total_return'].mean()
                avg_dd = combo_data['max_drawdown'].mean()
                report_lines.append(
                    f"• +DI({int(plus_di):2d}) / -DI({int(minus_di):2d}): "
                    f"Sharpe {avg_sharpe:.3f} | Return {avg_return:.1%} | DD {avg_dd:.1%}"
                )
        
        report_lines.append("")
        report_lines.append("=" * 68)
        report_lines.append("🏁 DI-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        report_lines.append("=" * 68)
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_folder, 'comprehensive_di_combinations_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def _calculate_di_performance_metrics(self, signals_df: pd.DataFrame) -> Dict:
        """
        Berechnet Performance-Metriken für DI-Signale
        
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
            # Entferne extreme Outliers (> 100% oder < -50% daily returns)
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
    
    def generate_di_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert DI-spezifische Analyse für verschiedene Kalibrierungen
        Fokussiert auf die besten DI-Kombinationen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für DI-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle DI-spezifische Analyse...")
        
        # Finde beste DI-Kombinationen
        avg_performance = results_df.groupby(['plus_di_length', 'minus_di_length']).agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'sortino_ratio': 'mean',
            'win_rate': 'mean',
            'omega_ratio': 'mean'
        }).reset_index()
        
        # Berechne kombinierten Score
        avg_performance['combined_score'] = (
            avg_performance['sharpe_ratio'] * 0.3 +
            avg_performance['sortino_ratio'] * 0.3 +
            avg_performance['total_return'] * 0.2 +
            (1 - avg_performance['max_drawdown']) * 0.2
        )
        
        # Erstelle DI-Bericht
        di_report_lines = []
        di_report_lines.append("=" * 80)
        di_report_lines.append("🎯 DIRECTIONAL INDICATORS KOMBINATIONEN-SPEZIFISCHE ANALYSE")
        di_report_lines.append("=" * 80)
        di_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        di_report_lines.append("")
        
        # Beste Kombination gesamt
        best_combination = avg_performance.loc[avg_performance['combined_score'].idxmax()]
        di_report_lines.append(f"🥇 BESTE DI-KOMBINATION (Kombinierter Score):")
        di_report_lines.append(f"   +DI: {int(best_combination['plus_di_length'])} / -DI: {int(best_combination['minus_di_length'])}")
        di_report_lines.append(f"   📊 Avg Sharpe: {best_combination['sharpe_ratio']:.3f}")
        di_report_lines.append(f"   💰 Avg Return: {best_combination['total_return']:.1%}")
        di_report_lines.append(f"   🛡️ Avg Drawdown: {best_combination['max_drawdown']:.1%}")
        di_report_lines.append(f"   📈 Avg Sortino: {best_combination['sortino_ratio']:.3f}")
        di_report_lines.append(f"   🎯 Combined Score: {best_combination['combined_score']:.3f}")
        
        # Beste für einzelne Metriken
        di_report_lines.append("")
        di_report_lines.append("📈 BESTE KOMBINATIONEN NACH METRIKEN:")
        
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
                di_report_lines.append(
                    f"   • {name}: +DI({int(best['plus_di_length'])}) / -DI({int(best['minus_di_length'])}) "
                    f"(Ø {value_str})"
                )
        
        # Niedrigster Drawdown
        best_dd = avg_performance.loc[avg_performance['max_drawdown'].idxmin()]
        di_report_lines.append(
            f"   🛡️ Niedrigster Drawdown: +DI({int(best_dd['plus_di_length'])}) / -DI({int(best_dd['minus_di_length'])}) "
            f"(Ø {best_dd['max_drawdown']:.1%})"
        )
        
        # Analyse verschiedener Kalibrierungstypen
        di_report_lines.append(f"\n💡 KALIBRIERUNGSTYPEN ANALYSE:")
        di_report_lines.append("-" * 60)
        
        # Gleiche Perioden
        same_periods = avg_performance[avg_performance['plus_di_length'] == avg_performance['minus_di_length']]
        if not same_periods.empty:
            best_same = same_periods.loc[same_periods['combined_score'].idxmax()]
            di_report_lines.append(f"📊 GLEICHE PERIODEN (+DI = -DI):")
            di_report_lines.append(f"   • Beste: DI({int(best_same['plus_di_length'])}) Score: {best_same['combined_score']:.3f}")
            di_report_lines.append(f"   • Performance: Sharpe {best_same['sharpe_ratio']:.3f} | Return {best_same['total_return']:.1%}")
        
        # Schnelle +DI, langsame -DI
        fast_plus_slow_minus = avg_performance[avg_performance['plus_di_length'] < avg_performance['minus_di_length']]
        if not fast_plus_slow_minus.empty:
            best_fast_plus = fast_plus_slow_minus.loc[fast_plus_slow_minus['combined_score'].idxmax()]
            di_report_lines.append(f"📊 SCHNELLE +DI / LANGSAME -DI:")
            di_report_lines.append(
                f"   • Beste: +DI({int(best_fast_plus['plus_di_length'])}) / -DI({int(best_fast_plus['minus_di_length'])}) "
                f"Score: {best_fast_plus['combined_score']:.3f}"
            )
            di_report_lines.append(f"   • Performance: Sharpe {best_fast_plus['sharpe_ratio']:.3f} | Return {best_fast_plus['total_return']:.1%}")
        
        # Langsame +DI, schnelle -DI
        slow_plus_fast_minus = avg_performance[avg_performance['plus_di_length'] > avg_performance['minus_di_length']]
        if not slow_plus_fast_minus.empty:
            best_slow_plus = slow_plus_fast_minus.loc[slow_plus_fast_minus['combined_score'].idxmax()]
            di_report_lines.append(f"📊 LANGSAME +DI / SCHNELLE -DI:")
            di_report_lines.append(
                f"   • Beste: +DI({int(best_slow_plus['plus_di_length'])}) / -DI({int(best_slow_plus['minus_di_length'])}) "
                f"Score: {best_slow_plus['combined_score']:.3f}"
            )
            di_report_lines.append(f"   • Performance: Sharpe {best_slow_plus['sharpe_ratio']:.3f} | Return {best_slow_plus['total_return']:.1%}")
        
        # Top 5 Kombinationen
        top_5 = avg_performance.nlargest(5, 'combined_score')
        di_report_lines.append(f"\n🏆 TOP 5 DI-KOMBINATIONEN:")
        di_report_lines.append("-" * 60)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            di_report_lines.append(
                f"{i}. +DI({int(row['plus_di_length'])}) / -DI({int(row['minus_di_length'])}) | "
                f"Score: {row['combined_score']:.3f} | Sharpe: {row['sharpe_ratio']:.3f} | "
                f"Return: {row['total_return']:.1%}"
            )
        
        # Strategische Empfehlungen
        di_report_lines.append(f"\n🎯 STRATEGISCHE EMPFEHLUNGEN:")
        di_report_lines.append("-" * 60)
        di_report_lines.append("� DI KOMBINATIONEN STRATEGIE INSIGHTS:")
        di_report_lines.append(f"   • Verschiedene +DI/-DI Perioden können Performance verbessern")
        di_report_lines.append(f"   • Schnelle +DI + langsame -DI: Frühe Trend-Erkennung, späte Exit-Signale")
        di_report_lines.append(f"   • Langsame +DI + schnelle -DI: Bestätigte Trends, schnelle Exits")
        di_report_lines.append(f"   • Gleiche Perioden: Klassischer Ansatz, ausgewogene Signale")
        di_report_lines.append(f"   • Extreme Unterschiede (z.B. 12/50) können bei volatilen Märkten helfen")
        di_report_lines.append(f"   • Mittlere Unterschiede (z.B. 21/30) bieten oft gute Balance")
        
        di_report_lines.append(f"\n" + "=" * 68)
        di_report_lines.append(f"🏁 DI-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        di_report_lines.append("=" * 68)
        
        # Speichere DI-spezifischen Bericht
        di_report_text = "\n".join(di_report_lines)
        di_report_path = os.path.join(self.results_folder, 'di_combinations_specific_analysis.txt')
        
        with open(di_report_path, 'w', encoding='utf-8') as f:
            f.write(di_report_text)
        
        print(f"📄 DI-Kombinationen Analyse gespeichert: {di_report_path}")

def main():
    """
    Hauptfunktion für DI Backtesting System
    """
    print("🚀 DIRECTIONAL INDICATORS BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (+DI > -DI)")
    print("   • DI-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • Directional Indicators: +DI vs -DI Crossover")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe DI Backtesting System aus
        di_system = DIBacktestingSystem(max_assets=20)
        
        # Führe Backtests mit vollständiger Matrix durch
        # +DI Range: 1-150, -DI Range: 1-150 (Vollständige Matrix in 1er-Schritten)
        plus_di_range = range(1, 151)  # +DI: 1 bis 150 (1er-Schritte)
        minus_di_range = range(1, 151)  # -DI: 1 bis 150 (1er-Schritte)
        
        print(f"📊 VOLLSTÄNDIGE DI-MATRIX ANALYSE (1-150)")
        print(f"   Matrix-Größe: {len(plus_di_range)} × {len(minus_di_range)} = {len(plus_di_range) * len(minus_di_range)} Kombinationen")
        print(f"   Gesamt-Tests: {len(plus_di_range) * len(minus_di_range) * 17} (mit 17 Assets)")
        print(f"   ⚠️  ACHTUNG: Das ist eine MASSIVE Analyse mit {len(plus_di_range) * len(minus_di_range) * 17:,} Tests!")
        print(f"   ⏱️  Geschätzte Laufzeit: 10-15 Stunden")
        print()
        
        results_df = di_system.run_di_backtests(plus_di_range=plus_di_range, minus_di_range=minus_di_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            di_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 DI-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {di_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim DI-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()