"""
MACD (Moving Average Convergence Divergence) Backtesting System (Optimiert mit Base Class)

Dieses System testet MACD-basierte Trading-Strategien über verschiedene Assets und Perioden:
- Fast EMA-Perioden von 1 bis 150
- Slow EMA-Perioden von 1 bis 150
- Signal-Periode fest bei 9
- my_type immer 0 (Standard EMA)
- 8-20 verschiedene Major Crypto Assets
- Long-Only Strategie: MACD > Signal = Long Position, sonst Cash Position
- Verwendet gemeinsame Funktionen aus backtesting_base.py

MACD (Moving Average Convergence Divergence):
- Trend-folgender Momentum-Indikator entwickelt von Gerald Appel
- MACD Line: Differenz zwischen schneller und langsamer EMA
- Signal Line: EMA der MACD Line (hier fest bei 9 Perioden)
- my_type = 0: Standard EMA-basierte Berechnung
- Long-Signal: MACD > Signal Line (bullisches Momentum)
- Cash-Signal: MACD <= Signal Line (bearisches Momentum)
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

class MACDBacktestingSystem(BaseBacktestingSystem):
    """
    MACD Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    Besonderheit: Zwei EMA-Perioden (fast/slow) mit fester Signal-Periode (9) und my_type=0
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "MACD", **kwargs)
        self.indicator_name = "MACD"
        self.strategy_description = "MACD > Signal: Long-Position | MACD <= Signal: Cash-Position"
        self.threshold = 0.0  # Kein absoluter Schwellenwert, nur Vergleich MACD vs Signal
        self.signal_period = 9  # Feste Signal-Periode
        self.my_type = 0  # Standard EMA-basiert
    
    def calculate_macd_signals(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
        """
        Berechnet MACD Signale für die Long-Only Strategie
        Unterstützt unterschiedliche Fast/Slow EMA-Perioden mit fester Signal-Periode
        
        Args:
            data: DataFrame mit OHLCV-Daten
            fast_period: Schnelle EMA-Periode
            slow_period: Langsame EMA-Periode
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        min_length = max(fast_period, slow_period) + self.signal_period
        if len(data) < min_length + 1:
            return pd.DataFrame()
        
        try:
            # Validiere Parameter vor MACD-Berechnung
            if fast_period >= slow_period:
                print(f"⚠️ Ungültige MACD-Parameter: Fast({fast_period}) >= Slow({slow_period})")
                return pd.DataFrame()
            
            if fast_period < 2 or slow_period < 3:
                print(f"⚠️ MACD-Parameter zu klein: Fast({fast_period}), Slow({slow_period})")
                return pd.DataFrame()
            
            # Berechne MACD mit festen Parametern (my_type=0, signal_period=9)
            macd, macdsignal, macdhist = ta.MACD(
                data['close'].astype(float).values, 
                fastperiod=fast_period, 
                slowperiod=slow_period, 
                signalperiod=self.signal_period
            )
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['macd'] = macd
            signals_df['macd_signal'] = macdsignal
            signals_df['macd_histogram'] = macdhist
            signals_df['fast_period'] = fast_period
            signals_df['slow_period'] = slow_period
            signals_df['signal_period'] = self.signal_period
            signals_df['my_type'] = self.my_type
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # MACD Long-Only Signale (MACD > Signal = Long, MACD <= Signal = Cash)
            signals_df['position'] = np.where(signals_df['macd'] > signals_df['macd_signal'], 1, 0)
            
            # Berechne MACD-Signal Differenz für zusätzliche Analyse
            signals_df['macd_diff'] = signals_df['macd'] - signals_df['macd_signal']
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile (NaN wegen pct_change)
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei MACD-Berechnung (Fast:{fast_period}, Slow:{slow_period}): {e}")
            return pd.DataFrame()
    
    def run_macd_backtests(self, fast_period_range: range = None, slow_period_range: range = None) -> pd.DataFrame:
        """
        Führt MACD-Backtests über vollständige Matrix verschiedener EMA-Perioden-Kombinationen durch
        
        Args:
            fast_period_range: Range der Fast EMA-Perioden zum Testen (Standard: 1-150)
            slow_period_range: Range der Slow EMA-Perioden zum Testen (Standard: 1-150)
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        if fast_period_range is None:
            fast_period_range = range(1, 151)  # Fast EMA: 1 bis 150
        
        if slow_period_range is None:
            slow_period_range = range(1, 151)  # Slow EMA: 1 bis 150
        
        print(f"🚀 MACD BACKTESTING SYSTEM START")
        print("=" * 60)
        print(f"⚙️ KONFIGURATION:")
        print(f"   Fast EMA-Range: {fast_period_range.start} bis {fast_period_range.stop-1}")
        print(f"   Slow EMA-Range: {slow_period_range.start} bis {slow_period_range.stop-1}")
        print(f"   Signal-Periode: {self.signal_period} (fest)")
        print(f"   my_type: {self.my_type} (Standard EMA)")
        print(f"   Vollständige Matrix: {len(fast_period_range)} × {len(slow_period_range)} = {len(fast_period_range) * len(slow_period_range)} MACD-Kombinationen")
        print(f"   Max Assets: {len(self.assets)} (aus CSV)")
        print(f"   Strategie: MACD > Signal: Long-Position | MACD <= Signal: Cash-Position")
        print(f"   Schwelle: {self.threshold}")
        print()
        
        all_results = []
        total_combinations = len(fast_period_range) * len(slow_period_range) * len(self.assets_data)
        current_combination = 0
        
        for fast_period in fast_period_range:
            for slow_period in slow_period_range:
                # Überspringe ungültige Kombinationen (Fast >= Slow oder zu kleiner Abstand)
                if fast_period >= slow_period or (slow_period - fast_period) < 2:
                    continue
                
                for asset in self.assets:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    # Statische Progress-Anzeige
                    print(f"\r📊 Teste MACD Fast({fast_period:3d}) / Slow({slow_period:3d}) | Progress: {progress:5.1f}% ({current_combination}/{total_combinations})     ", end='', flush=True)
                    
                    if asset not in self.assets_data:
                        continue
                    
                    # Berechne Signale für diese MACD-Kombination
                    signals_df = self.calculate_macd_signals(
                        self.assets_data[asset], 
                        fast_period=fast_period, 
                        slow_period=slow_period
                    )
                    
                    if signals_df.empty:
                        continue
                    
                    # Stelle sicher, dass die erwarteten Spalten existieren
                    if 'strategy_returns' not in signals_df.columns:
                        continue
                    
                    # Berechne Performance-Metriken (spezifische Implementierung)
                    metrics = self._calculate_macd_performance_metrics(signals_df)
                    
                    if metrics:
                        metrics.update({
                            'asset': asset,
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'signal_period': self.signal_period,
                            'my_type': self.my_type,
                            'macd_combination': f"{fast_period}-{slow_period}-{self.signal_period}"
                        })
                        all_results.append(metrics)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        results_df = pd.DataFrame(all_results)
        print(f"✅ MACD Matrix Backtests abgeschlossen: {len(all_results)} Kombinationen getestet")
        print(f"📊 Matrix-Dimensionen: {len(fast_period_range)} (Fast) × {len(slow_period_range)} (Slow) × {len(self.assets_data)} (Assets)")
        
        return results_df
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden MACD-Bericht mit allen Analysen für verschiedene Kalibrierungen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        print(f"\n📋 Generiere umfassenden MACD Bericht...")
        
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
                        
                        plt.title(f'MACD Combinations Heatmap: {metric.replace("_", " ").title()}', 
                                fontsize=16, fontweight='bold')
                        plt.xlabel('Slow EMA Period', fontsize=12, fontweight='bold')
                        plt.ylabel('Fast EMA Period', fontsize=12, fontweight='bold')
                        plt.tight_layout()
                        
                        heatmap_path = os.path.join(self.results_folder, f'heatmap_{metric}.png')
                        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"📊 Heatmap gespeichert: {heatmap_path}")
                        
                except Exception as e:
                    print(f"⚠️ Fehler beim Erstellen der Heatmap für {metric}: {e}")
        
        # Generiere Text-Bericht
        self._generate_macd_combinations_report(results_df)
        
        # Speichere CSV-Daten
        csv_path = os.path.join(self.results_folder, 'all_macd_combinations_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"📊 CSV-Daten gespeichert: {csv_path}")
        
        print(f"📄 MACD Bericht gespeichert: {os.path.join(self.results_folder, 'comprehensive_macd_combinations_report.txt')}")
        
        print(f"\n" + "=" * 60)
        print(f"✅ MACD BACKTESTING ABGESCHLOSSEN")
        print("=" * 60)
        print(f"📊 {len(results_df)} Konfigurationen getestet")
        if not results_df.empty:
            print(f"🥇 Beste Sharpe Ratio: {results_df['sharpe_ratio'].max():.3f}")
            print(f"💰 Höchster Return: {results_df['total_return'].max():.1%}")
        print(f"📁 Alle Ergebnisse in: {self.results_folder}/")
        
        # Zusätzliche MACD-spezifische Analyse
        print(f"\n📋 Erstelle MACD-spezifische Analyse...")
        self.generate_macd_specific_analysis(results_df)
    
    def _generate_macd_combinations_report(self, results_df: pd.DataFrame):
        """
        Generiert detaillierten Bericht für MACD-Kombinationen
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 MACD KOMBINATIONEN ANALYSE")
        report_lines.append("=" * 80)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📋 Getestete Kombinationen: {len(results_df)}")
        report_lines.append(f"💎 Assets: {results_df['asset'].nunique()}")
        report_lines.append(f"🔢 MACD-Kombinationen: {results_df['macd_combination'].nunique()}")
        report_lines.append(f"📊 Signal-Periode: {self.signal_period} (fest)")
        report_lines.append(f"⚙️ my_type: {self.my_type} (Standard EMA)")
        report_lines.append("")
        
        # Top 10 beste Kombinationen nach Sharpe Ratio
        if 'sharpe_ratio' in results_df.columns:
            top_sharpe = results_df.nlargest(10, 'sharpe_ratio')
            report_lines.append("🥇 TOP 10 BESTE MACD-KOMBINATIONEN (Sharpe Ratio):")
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
                f"{i:2d}. Fast({int(row['fast_period']):2d}) / Slow({int(row['slow_period']):2d}) | "
                f"Avg Sharpe: {row['sharpe_ratio']:.3f} | "
                f"Avg Return: {row['total_return']:.1%} | Score: {row['combined_score']:.3f}"
            )
        
        # Analyse verschiedener EMA-Verhältnisse
        report_lines.append("")
        report_lines.append("⚖️ EMA-VERHÄLTNIS ANALYSE:")
        report_lines.append("-" * 60)
        
        # Berechne EMA-Verhältnis
        avg_performance['ema_ratio'] = avg_performance['slow_period'] / avg_performance['fast_period']
        
        # Verschiedene Verhältnis-Kategorien
        ratio_categories = [
            (1.5, 2.0, "Enge Verhältnisse (1.5-2.0)"),
            (2.0, 3.0, "Mittlere Verhältnisse (2.0-3.0)"),
            (3.0, 5.0, "Weite Verhältnisse (3.0-5.0)"),
            (5.0, float('inf'), "Sehr weite Verhältnisse (>5.0)")
        ]
        
        for min_ratio, max_ratio, category_name in ratio_categories:
            category_data = avg_performance[
                (avg_performance['ema_ratio'] >= min_ratio) & 
                (avg_performance['ema_ratio'] < max_ratio)
            ]
            
            if not category_data.empty:
                avg_sharpe = category_data['sharpe_ratio'].mean()
                avg_return = category_data['total_return'].mean()
                avg_dd = category_data['max_drawdown'].mean()
                count = len(category_data)
                report_lines.append(
                    f"📊 {category_name}: {count} Kombinationen"
                )
                report_lines.append(
                    f"   • Avg Sharpe: {avg_sharpe:.3f} | Avg Return: {avg_return:.1%} | Avg DD: {avg_dd:.1%}"
                )
        
        # Spezielle klassische Kalibrierungen analysieren
        classic_combos = [(12, 26), (8, 21), (5, 35), (21, 55), (13, 34)]
        
        report_lines.append("")
        report_lines.append("📈 KLASSISCHE MACD KALIBRIERUNGEN ANALYSE:")
        report_lines.append("-" * 60)
        
        for fast, slow in classic_combos:
            combo_data = results_df[
                (results_df['fast_period'] == fast) & 
                (results_df['slow_period'] == slow)
            ]
            if not combo_data.empty:
                avg_sharpe = combo_data['sharpe_ratio'].mean()
                avg_return = combo_data['total_return'].mean()
                avg_dd = combo_data['max_drawdown'].mean()
                report_lines.append(
                    f"• Fast({int(fast):2d}) / Slow({int(slow):2d}): "
                    f"Sharpe {avg_sharpe:.3f} | Return {avg_return:.1%} | DD {avg_dd:.1%}"
                )
        
        report_lines.append("")
        report_lines.append("=" * 68)
        report_lines.append("🏁 MACD-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        report_lines.append("=" * 68)
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_folder, 'comprehensive_macd_combinations_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def _calculate_macd_performance_metrics(self, signals_df: pd.DataFrame) -> Dict:
        """
        Berechnet Performance-Metriken für MACD-Signale
        
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
    
    def generate_macd_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert MACD-spezifische Analyse für verschiedene Kalibrierungen
        Fokussiert auf die besten MACD-Kombinationen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für MACD-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle MACD-spezifische Analyse...")
        
        # Finde beste MACD-Kombinationen
        avg_performance = results_df.groupby(['fast_period', 'slow_period']).agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'sortino_ratio': 'mean',
            'win_rate': 'mean',
            'omega_ratio': 'mean'
        }).reset_index()
        
        # Berechne EMA-Verhältnis und kombinierten Score
        avg_performance['ema_ratio'] = avg_performance['slow_period'] / avg_performance['fast_period']
        avg_performance['combined_score'] = (
            avg_performance['sharpe_ratio'] * 0.3 +
            avg_performance['sortino_ratio'] * 0.3 +
            avg_performance['total_return'] * 0.2 +
            (1 - avg_performance['max_drawdown']) * 0.2
        )
        
        # Erstelle MACD-Bericht
        macd_report_lines = []
        macd_report_lines.append("=" * 80)
        macd_report_lines.append("🎯 MACD KOMBINATIONEN-SPEZIFISCHE ANALYSE")
        macd_report_lines.append("=" * 80)
        macd_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        macd_report_lines.append(f"📊 Signal-Periode: {self.signal_period} (fest)")
        macd_report_lines.append(f"⚙️ my_type: {self.my_type} (Standard EMA)")
        macd_report_lines.append("")
        
        # Beste Kombination gesamt
        best_combination = avg_performance.loc[avg_performance['combined_score'].idxmax()]
        macd_report_lines.append(f"🥇 BESTE MACD-KOMBINATION (Kombinierter Score):")
        macd_report_lines.append(f"   Fast EMA: {int(best_combination['fast_period'])} / Slow EMA: {int(best_combination['slow_period'])}")
        macd_report_lines.append(f"   📊 EMA-Verhältnis: {best_combination['ema_ratio']:.2f}")
        macd_report_lines.append(f"   📊 Avg Sharpe: {best_combination['sharpe_ratio']:.3f}")
        macd_report_lines.append(f"   💰 Avg Return: {best_combination['total_return']:.1%}")
        macd_report_lines.append(f"   🛡️ Avg Drawdown: {best_combination['max_drawdown']:.1%}")
        macd_report_lines.append(f"   📈 Avg Sortino: {best_combination['sortino_ratio']:.3f}")
        macd_report_lines.append(f"   🎯 Combined Score: {best_combination['combined_score']:.3f}")
        
        # Beste für einzelne Metriken
        macd_report_lines.append("")
        macd_report_lines.append("📈 BESTE KOMBINATIONEN NACH METRIKEN:")
        
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
                macd_report_lines.append(
                    f"   • {name}: Fast({int(best['fast_period'])}) / Slow({int(best['slow_period'])}) "
                    f"(Ø {value_str}, Ratio: {best['ema_ratio']:.2f})"
                )
        
        # Niedrigster Drawdown
        best_dd = avg_performance.loc[avg_performance['max_drawdown'].idxmin()]
        macd_report_lines.append(
            f"   🛡️ Niedrigster Drawdown: Fast({int(best_dd['fast_period'])}) / Slow({int(best_dd['slow_period'])}) "
            f"(Ø {best_dd['max_drawdown']:.1%}, Ratio: {best_dd['ema_ratio']:.2f})"
        )
        
        # Analyse verschiedener EMA-Verhältnisse
        macd_report_lines.append(f"\n💡 EMA-VERHÄLTNIS ANALYSE:")
        macd_report_lines.append("-" * 60)
        
        ratio_ranges = [
            (1.0, 2.0, "Enge Verhältnisse (1.0-2.0)"),
            (2.0, 3.0, "Klassische Verhältnisse (2.0-3.0)"),
            (3.0, 5.0, "Weite Verhältnisse (3.0-5.0)"),
            (5.0, 10.0, "Sehr weite Verhältnisse (5.0-10.0)"),
            (10.0, float('inf'), "Extreme Verhältnisse (>10.0)")
        ]
        
        for min_ratio, max_ratio, range_name in ratio_ranges:
            range_data = avg_performance[
                (avg_performance['ema_ratio'] >= min_ratio) & 
                (avg_performance['ema_ratio'] < max_ratio)
            ]
            
            if not range_data.empty:
                best_in_range = range_data.loc[range_data['combined_score'].idxmax()]
                avg_score = range_data['combined_score'].mean()
                count = len(range_data)
                
                macd_report_lines.append(f"📊 {range_name}: {count} Kombinationen")
                macd_report_lines.append(
                    f"   • Beste: Fast({int(best_in_range['fast_period'])}) / Slow({int(best_in_range['slow_period'])}) "
                    f"Score: {best_in_range['combined_score']:.3f}"
                )
                macd_report_lines.append(f"   • Ø Score: {avg_score:.3f}")
        
        # Klassische MACD-Kalibrierungen
        classic_configs = [
            (12, 26, "Standard MACD (12,26,9)"),
            (8, 21, "Schnellerer MACD (8,21,9)"),
            (5, 35, "Aggressiver MACD (5,35,9)"),
            (21, 55, "Konservativer MACD (21,55,9)"),
            (13, 34, "Fibonacci MACD (13,34,9)")
        ]
        
        macd_report_lines.append(f"\n📈 KLASSISCHE MACD KALIBRIERUNGEN:")
        macd_report_lines.append("-" * 60)
        
        for fast, slow, config_name in classic_configs:
            config_data = avg_performance[
                (avg_performance['fast_period'] == fast) & 
                (avg_performance['slow_period'] == slow)
            ]
            
            if not config_data.empty:
                config = config_data.iloc[0]
                macd_report_lines.append(f"📊 {config_name}:")
                macd_report_lines.append(
                    f"   • Score: {config['combined_score']:.3f} | Sharpe: {config['sharpe_ratio']:.3f} | "
                    f"Return: {config['total_return']:.1%}"
                )
        
        # Top 5 Kombinationen nach verschiedenen Kriterien
        macd_report_lines.append(f"\n🏆 TOP 5 MACD-KOMBINATIONEN:")
        macd_report_lines.append("-" * 60)
        
        top_5 = avg_performance.nlargest(5, 'combined_score')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            macd_report_lines.append(
                f"{i}. Fast({int(row['fast_period'])}) / Slow({int(row['slow_period'])}) | "
                f"Ratio: {row['ema_ratio']:.2f} | Score: {row['combined_score']:.3f} | "
                f"Sharpe: {row['sharpe_ratio']:.3f} | Return: {row['total_return']:.1%}"
            )
        
        # Strategische Empfehlungen
        macd_report_lines.append(f"\n🎯 STRATEGISCHE EMPFEHLUNGEN:")
        macd_report_lines.append("-" * 60)
        macd_report_lines.append("📊 MACD KOMBINATIONEN STRATEGIE INSIGHTS:")
        macd_report_lines.append(f"   • EMA-Verhältnis 2.0-3.0: Klassisch bewährt, ausgewogene Signale")
        macd_report_lines.append(f"   • Enge Verhältnisse (<2.0): Frühe Signale, mehr Whipsaws")
        macd_report_lines.append(f"   • Weite Verhältnisse (>3.0): Späte aber bestätigte Signale")
        macd_report_lines.append(f"   • Signal-Periode 9: Standard und bewährt für die meisten Märkte")
        macd_report_lines.append(f"   • my_type 0: Standard EMA-Berechnung, keine Glättung")
        macd_report_lines.append(f"   • Fibonacci-Kombinationen (z.B. 13,34): Oft mathematisch harmonisch")
        
        macd_report_lines.append(f"\n" + "=" * 68)
        macd_report_lines.append(f"🏁 MACD-KOMBINATIONEN ANALYSE ABGESCHLOSSEN")
        macd_report_lines.append("=" * 68)
        
        # Speichere MACD-spezifischen Bericht
        macd_report_text = "\n".join(macd_report_lines)
        macd_report_path = os.path.join(self.results_folder, 'macd_combinations_specific_analysis.txt')
        
        with open(macd_report_path, 'w', encoding='utf-8') as f:
            f.write(macd_report_text)
        
        print(f"📄 MACD-Kombinationen Analyse gespeichert: {macd_report_path}")

def main():
    """
    Hauptfunktion für MACD Backtesting System
    """
    print("🚀 MACD BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (MACD > Signal)")
    print("   • Fast EMA-Range: 2 bis 150, Slow EMA-Range: 4 bis 150 (Einser-Schritte)")
    print("   • Signal-Periode: 9 (fest)")
    print("   • my_type: 0 (Standard EMA)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • MACD: Moving Average Convergence Divergence")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe MACD Backtesting System aus
        macd_system = MACDBacktestingSystem(max_assets=20)
        
        # Führe Backtests mit vollständiger Matrix durch
        # Fast EMA Range: 2-150, Slow EMA Range: 4-150 (aber nur Fast < Slow mit Mindestabstand)
        fast_period_range = range(2, 151)  # Fast EMA: 2 bis 150 (1er-Schritte)
        slow_period_range = range(4, 151)  # Slow EMA: 4 bis 150 (1er-Schritte)
        
        # Berechne tatsächliche Anzahl gültiger Kombinationen (Fast < Slow mit Mindestabstand 2)
        valid_combinations = sum(1 for fast in fast_period_range for slow in slow_period_range 
                               if fast < slow and (slow - fast) >= 2)
        
        print(f"📊 VOLLSTÄNDIGE MACD-MATRIX ANALYSE (2-150)")
        print(f"   Fast EMA: {fast_period_range.start}-{fast_period_range.stop-1}")
        print(f"   Slow EMA: {slow_period_range.start}-{slow_period_range.stop-1}")
        print(f"   Matrix-Größe: {len(fast_period_range)} × {len(slow_period_range)} = {len(fast_period_range) * len(slow_period_range)} theoretisch")
        print(f"   Gültige Kombinationen: {valid_combinations} (Fast < Slow, Mindestabstand 2)")
        print(f"   Gesamt-Tests: {valid_combinations * 17} (mit 17 Assets)")
        print(f"   ⚠️  ACHTUNG: Das ist eine MASSIVE Analyse mit {valid_combinations * 17:,} Tests!")
        print(f"   ⏱️  Geschätzte Laufzeit: 6-10 Stunden")
        print()
        
        results_df = macd_system.run_macd_backtests(fast_period_range=fast_period_range, slow_period_range=slow_period_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            macd_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 MACD-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {macd_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim MACD-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()