#!/usr/bin/env python3
"""
FSVZO (Fourier-Smoothed Volume Zone Oscillator) BACKTESTING SYSTEM
===================================================================

Backtesting-System für FSVZO Indikator mit:
- Single-Parameter Analyse (VZO Length)
- VZO > Signal: Long-Position | VZO <= Signal: Cash-Position
- Optimierung über verschiedene Length-Kalibrierungen (5-150)
- Umfassende Performance-Analyse
- Integration mit backtesting_base.py für Code-Wiederverwendung

FSVZO (Fourier-Smoothed Volume Zone Oscillator):
- Volume-basierter Oszillator mit Fourier-Glättung
- VZO Werte zwischen -100 und +100
- Signal Line für Crossover-Signale
- VZO > Signal = Long (bullisches Volume-Momentum)
- VZO <= Signal = Cash (bearisches Volume-Momentum)
- Benötigt Volume-Daten

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

class FSVZOBacktestingSystem(BaseBacktestingSystem):
    """
    FSVZO Backtesting-System (Long-Only Strategie)
    Erbt von BaseBacktestingSystem und nutzt gemeinsame Funktionen
    """
    
    def __init__(self, max_assets: int = 20, **kwargs):
        super().__init__(max_assets, "FSVZO", **kwargs)
        self.indicator_name = "FSVZO"
        self.strategy_description = "VZO > Signal: Long-Position | VZO <= Signal: Cash-Position"
        self.threshold = 0.0  # Signal Line als Referenz
    
    def fourier_smooth(self, series: np.ndarray, length: int) -> np.ndarray:
        """Fourier-basierte exponentielle Glättung"""
        result = np.full(len(series), np.nan)
        
        for i in range(length - 1, len(series)):
            sum_val = 0.0
            weight_sum = 0.0
            
            for j in range(length):
                if i - j >= 0 and not np.isnan(series[i - j]):
                    weight = np.exp(-j / (length * 0.3))
                    sum_val += series[i - j] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                result[i] = sum_val / weight_sum
        
        return result
    
    def adf_trend_filter(self, source: np.ndarray, window: int) -> np.ndarray:
        """Trend-Filter basierend auf ADF-Test-Approximation"""
        if window <= 0:
            return np.ones(len(source))
        
        result = np.ones(len(source))
        
        for i in range(window, len(source)):
            window_data = source[i-window+1:i+1]
            
            sma_short_period = max(1, window // 3)
            sma_short = np.mean(window_data[-sma_short_period:])
            sma_long = np.mean(window_data)
            
            volatility = np.std(window_data)
            
            if volatility > 0:
                trend_strength = (sma_short - sma_long) / volatility
            else:
                trend_strength = 0
            
            result[i] = 1.0 + max(-0.1, min(0.1, trend_strength * 0.2))
        
        return result
    
    def calculate_fsvzo_signals(self, data: pd.DataFrame, vzo_length: int = 9) -> pd.DataFrame:
        """
        Berechnet FSVZO-Signale für die Long-Only Strategie
        
        Args:
            data: DataFrame mit OHLCV-Daten
            vzo_length: VZO-Periode (feste Parameter für andere Werte)
            
        Returns:
            DataFrame mit Signalen und Returns
        """
        # Feste Parameter für stabiles Backtesting
        signal_length = 3
        smoothing_length = 3
        fourier_length = 31
        adf_window = 50
        
        min_length = max(vzo_length, adf_window) + 10
        
        if len(data) < min_length:
            return pd.DataFrame()
        
        # Prüfe ob Volume-Daten verfügbar sind
        if 'volume' not in data.columns:
            return pd.DataFrame()
        
        try:
            close = data['close'].values.astype(np.float64)  # Konvertiere zu float64
            volume = data['volume'].values.astype(np.float64)  # Konvertiere zu float64
            
            # Berechne Trend-Filter
            trend = self.adf_trend_filter(close, adf_window)
            
            # Relative Volume
            volume_sma = ta.SMA(volume, timeperiod=vzo_length)
            rel_volume = np.where(volume_sma != 0, volume / volume_sma, 1.0)
            
            # Smooth relative volume
            smoothed_vol = ta.EMA(rel_volume, timeperiod=smoothing_length)
            
            # Price change
            price_change = np.diff(close, prepend=close[0])
            smoothed_change = ta.EMA(price_change, timeperiod=smoothing_length)
            
            # Momentum
            base_momentum = ta.EMA(smoothed_change * smoothed_vol, timeperiod=smoothing_length)
            trend_momentum = ta.EMA(smoothed_change * smoothed_vol * trend, timeperiod=smoothing_length)
            momentum = base_momentum * 0.7 + trend_momentum * 0.3
            
            # Positive und negative Momentum
            pos_mom = ta.EMA(np.maximum(momentum, 0), timeperiod=vzo_length)
            neg_mom = ta.EMA(np.abs(np.minimum(momentum, 0)), timeperiod=vzo_length)
            
            # Berechne Ratio
            ratio = np.full(len(close), 1.0)
            for i in range(len(close)):
                if neg_mom[i] > 0.00001:
                    ratio[i] = pos_mom[i] / neg_mom[i]
                elif pos_mom[i] > 0.00001:
                    ratio[i] = 100.0
                else:
                    ratio[i] = 1.0
            
            # VZO raw
            vzo_raw = 100.0 * (ratio - 1.0) / (ratio + 1.0)
            
            # Fourier smoothing
            if fourier_length >= 5:
                fourier_component = self.fourier_smooth(vzo_raw, fourier_length)
                ema_component = ta.EMA(vzo_raw, timeperiod=smoothing_length)
                vzo = np.where(~np.isnan(fourier_component), 
                              ema_component * 0.6 + fourier_component * 0.4,
                              ema_component)
            else:
                vzo = ta.EMA(vzo_raw, timeperiod=smoothing_length)
            
            # Clip to [-100, 100]
            vzo = np.clip(vzo, -100, 100)
            
            # Signal line
            signal = ta.SMA(vzo, timeperiod=signal_length)
            
            # Erstelle DataFrame
            signals_df = data.copy()
            signals_df['vzo'] = vzo
            signals_df['signal'] = signal
            
            # Entferne NaN-Werte
            signals_df = signals_df.dropna()
            
            if len(signals_df) < 2:
                return pd.DataFrame()
            
            # FSVZO Signale (VZO > Signal = Long, VZO <= Signal = Cash)
            signals_df['position'] = np.where(signals_df['vzo'] > signals_df['signal'], 1, 0)
            
            # Berechne Returns
            signals_df['asset_returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['asset_returns']
            
            # Entferne erste Zeile
            signals_df = signals_df.dropna()
            
            return signals_df
            
        except Exception as e:
            print(f"Fehler bei FSVZO-Berechnung: {e}")
            return pd.DataFrame()
    
    def run_fsvzo_backtests(self, vzo_range: range = range(5, 151)) -> pd.DataFrame:
        """
        Führt FSVZO-Backtests über verschiedene Perioden durch
        
        Args:
            vzo_range: Range der VZO-Perioden zum Testen
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        return self.run_generic_backtests(
            indicator_range=vzo_range,
            length_param_name='vzo_length',
            calculate_signals_func=self.calculate_fsvzo_signals,
            indicator_name='FSVZO'
        )
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame):
        """
        Generiert umfassenden FSVZO-Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht verfügbar")
            return
        
        # Verwende die Basis-Methode
        strategy_description = "FSVZO Long-Only Strategy (VZO > Signal = Long)"
        super().generate_comprehensive_report(results_df, 'vzo_length', strategy_description)
        
        # Zusätzliche FSVZO-spezifische Analyse
        print(f"\n📋 Erstelle FSVZO-spezifische Analyse...")
        self.generate_fsvzo_specific_analysis(results_df)
    
    def generate_fsvzo_specific_analysis(self, results_df: pd.DataFrame) -> None:
        """
        Generiert FSVZO-spezifische Analyse
        Fokussiert auf die besten VZO-Längen und deren Performance-Charakteristika
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
        """
        if results_df.empty:
            print("❌ Keine Daten für FSVZO-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle FSVZO-spezifische Analyse...")
        
        # Finde beste FSVZO-Kalibrierungen
        best_fsvzo_calibration = self.find_best_average_calibration(results_df, 'vzo_length')
        
        if best_fsvzo_calibration:
            # Erstelle zusätzlichen FSVZO-Bericht
            fsvzo_report_lines = []
            fsvzo_report_lines.append("=" * 80)
            fsvzo_report_lines.append("🎯 FSVZO-SPEZIFISCHE ANALYSE")
            fsvzo_report_lines.append("=" * 80)
            fsvzo_report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            fsvzo_report_lines.append("")
            
            # Beste kombinierte VZO-Länge
            if 'best_vzo_combined' in best_fsvzo_calibration:
                best_len = best_fsvzo_calibration['best_vzo_combined']
                avg_sharpe = best_fsvzo_calibration.get('combined_vzo_sharpe_ratio', 0)
                avg_return = best_fsvzo_calibration.get('combined_vzo_total_return', 0)
                avg_dd = best_fsvzo_calibration.get('combined_vzo_max_drawdown', 0)
                avg_sortino = best_fsvzo_calibration.get('combined_vzo_sortino_ratio', 0)
                avg_win_rate = best_fsvzo_calibration.get('combined_vzo_win_rate', 0)
                combined_score = best_fsvzo_calibration.get('avg_combined_score', 0)
                
                fsvzo_report_lines.append(f"🥇 Beste Durchschnittliche VZO-Länge (Kombiniert): VZO-{best_len}")
                fsvzo_report_lines.append(f"   📊 Avg Sharpe: {avg_sharpe:.3f} | Avg Return: {avg_return:.1%} | Avg DD: {avg_dd:.1%}")
                fsvzo_report_lines.append(f"   📈 Avg Sortino: {avg_sortino:.3f} | Avg Win Rate: {avg_win_rate:.1%} | Score: {combined_score:.3f}")
                fsvzo_report_lines.append("")
            
            # Top VZOs für einzelne Metriken
            fsvzo_report_lines.append("📈 Beste Durchschnitts-VZOs nach Metriken:")
            
            metric_keys = [
                ('best_vzo_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                ('best_vzo_total_return', 'avg_total_return', 'Total Return'),
                ('best_vzo_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_fsvzo_calibration:
                    best_len = best_fsvzo_calibration[best_key]
                    avg_value = best_fsvzo_calibration.get(avg_key, 0)
                    if 'return' in avg_key or 'drawdown' in avg_key or 'win_rate' in avg_key:
                        fsvzo_report_lines.append(f"   • {metric_name}: VZO-{best_len} (Ø {avg_value:.1%})")
                    else:
                        fsvzo_report_lines.append(f"   • {metric_name}: VZO-{best_len} (Ø {avg_value:.3f})")
            
            # FSVZO-spezifische Empfehlungen
            fsvzo_report_lines.append(f"\n💡 FSVZO STRATEGIE EMPFEHLUNGEN")
            fsvzo_report_lines.append("-" * 60)
            fsvzo_report_lines.append("📋 FSVZO LONG-ONLY STRATEGIE INSIGHTS:")
            fsvzo_report_lines.append(f"   • Long-Position wenn VZO > Signal (bullisches Volume-Momentum)")
            fsvzo_report_lines.append(f"   • Cash-Position wenn VZO <= Signal (bearisches Volume-Momentum)")
            fsvzo_report_lines.append(f"   • FSVZO kombiniert Volume-Analyse mit Fourier-Glättung")
            fsvzo_report_lines.append(f"   • ADF-Trend-Filter verbessert Signal-Qualität in Trends")
            fsvzo_report_lines.append(f"   • Benötigt Volume-Daten für korrekte Berechnung")
            fsvzo_report_lines.append(f"   • Kürzere VZO-Perioden: Schnellere Reaktion auf Volume-Änderungen")
            fsvzo_report_lines.append(f"   • Längere VZO-Perioden: Stabilere Volume-Trend-Erkennung")
            fsvzo_report_lines.append(f"   • Besonders effektiv bei Volume-getriebenen Marktbewegungen")
            
            fsvzo_report_lines.append("")
            fsvzo_report_lines.append("=" * 80)
            
            # Speichere FSVZO-spezifischen Bericht
            fsvzo_report_text = "\n".join(fsvzo_report_lines)
            fsvzo_report_path = os.path.join(self.results_folder, 'fsvzo_specific_analysis.txt')
            
            with open(fsvzo_report_path, 'w', encoding='utf-8') as f:
                f.write(fsvzo_report_text)
            
            print(f"📄 FSVZO-spezifische Analyse gespeichert: {fsvzo_report_path}")

def main():
    """
    Hauptfunktion für FSVZO Backtesting System
    """
    print("🚀 FSVZO BACKTESTING SYSTEM START (Long-Only)")
    print("=" * 60)
    print("⚙️ SYSTEM-INFO:")
    print("   • Strategie: Long-Only (VZO > Signal)")
    print("   • VZO-Range: 5 bis 150 (Einser-Schritte)")
    print("   • Assets: Major Cryptocurrencies")
    print("   • FSVZO: Volume Zone Oscillator mit Fourier-Glättung")
    print("   • ⚠️  BENÖTIGT: Volume-Daten")
    print("   • Optimiert mit gemeinsamen Base-Funktionen")
    
    try:
        # Erstelle und führe FSVZO Backtesting System aus
        fsvzo_system = FSVZOBacktestingSystem(max_assets=20)
        
        # Teste verschiedene VZO-Perioden
        vzo_range = range(5, 151)
        
        # Führe Backtests durch
        results_df = fsvzo_system.run_fsvzo_backtests(vzo_range)
        
        if not results_df.empty:
            # Generiere umfassenden Bericht
            fsvzo_system.generate_comprehensive_report(results_df)
            
            print(f"\n🎉 FSVZO-Backtesting erfolgreich abgeschlossen!")
            print(f"📊 {len(results_df)} Kombinationen getestet")
            print(f"📁 Ergebnisse in: {fsvzo_system.results_folder}/")
        else:
            print("❌ Keine gültigen Backtests durchgeführt")
    
    except Exception as e:
        print(f"❌ Fehler beim FSVZO-Backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
