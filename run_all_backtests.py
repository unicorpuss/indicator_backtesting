"""
UNIVERSAL BACKTESTING SYSTEM - ALLE 16 INDIKATOREN
==================================================

Dieses System startet alle verfügbaren Backtesting-Systeme und erstellt eine umfassende Übersicht:
- Alle 16 technischen Indikatoren (EMA, RSI, CCI, ADX, AROON, CMO, MFI, WILLR, DI, MACD, APO, MOM, PPO, TRIX, ADOSC)
- Beste Kalibrierungen für jeden Indikator
- Vergleichende Performance-Analyse
- Ranking aller Indikatoren
- Zusammenfassende Berichte und Visualisierungen

EINZELPARAMETER-INDIKATOREN (12):
- EMA (Exponential Moving Average): Price > EMA = Long
- RSI (Relative Strength Index): RSI > 50 = Long  
- CCI (Commodity Channel Index): CCI > 0 = Long
- ADX (Average Directional Index): ADX > 25 = Long
- AROON (Aroon Oscillator): Aroon > 0 = Long
- CMO (Chande Momentum Oscillator): CMO > 0 = Long
- MFI (Money Flow Index): MFI > 50 = Long
- WILLR (Williams %R): WILLR > -50 = Long
- MOM (Momentum): MOM > 0 = Long
- TRIX (Triple EMA): TRIX > 0 = Long

MATRIX-INDIKATOREN (6):
- DI (Directional Indicators): +DI > -DI = Long (Plus/Minus DI Matrix)
- MACD (Moving Average Convergence Divergence): MACD > Signal = Long (Fast/Slow/Signal Matrix)
- APO (Absolute Price Oscillator): APO > 0 = Long (Fast/Slow MA Matrix)
- PPO (Percentage Price Oscillator): PPO > 0 = Long (Fast/Slow EMA Matrix)
- ADOSC (Chaikin A/D Oscillator): ADOSC > 0 = Long (Fast/Slow EMA Matrix, Volume-based)

Autor: Enhanced Universal Backtesting Framework
Datum: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import traceback
warnings.filterwarnings('ignore')

# Füge das Hauptverzeichnis zum Python-Suchpfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import aller Backtesting-Systeme
try:
    from backtest.backtest_ema import EMABacktestingSystem
    from backtest.backtest_rsi import RSIBacktestingSystem
    from backtest.backtest_cci import CCIBacktestingSystem
    from backtest.backtest_adx import ADXBacktestingSystem
    from backtest.backtest_aroon import AROONBacktestingSystem
    from backtest.backtest_cmo import CMOBacktestingSystem
    from backtest.backtest_mfi import MFIBacktestingSystem
    from backtest.backtest_willr import WILLRBacktestingSystem
    from backtest.backtest_di import DIBacktestingSystem
    from backtest.backtest_macd import MACDBacktestingSystem
    from backtest.backtest_apo import APOBacktestingSystem
    from backtest.backtest_mom import MOMBacktestingSystem
    from backtest.backtest_ppo import PPOBacktestingSystem
    from backtest.backtest_trix import TRIXBacktestingSystem
    from backtest.backtest_adosc import ADOSCBacktestingSystem
    from backtest.backtest_vidya import VIDYABacktestingSystem
    from backtest.backtest_trendcont import TrendContinuationBacktestingSystem
    from backtest.backtest_hullsuite import HullSuiteBacktestingSystem
    from backtest.backtest_fsvzo import FSVZOBacktestingSystem
    from backtest.backtest_bbpct import BBPctBacktestingSystem
    from backtest.backtest_frama import FRAMABacktestingSystem
    from backtest.backtest_supertrend import SupertrendBacktestingSystem
    from backtest.backtest_mpt import MultiPivotTrendBacktestingSystem
    
    print("✅ Alle Backtesting-Module erfolgreich importiert (23 Indikatoren)")
except ImportError as e:
    print(f"⚠️ Fehler beim Import: {e}")
    print("Stelle sicher, dass alle Backtesting-Dateien vorhanden sind")

class UniversalBacktestingSystem:
    """
    Universal Backtesting System für alle technischen Indikatoren
    Startet alle verfügbaren Systeme und erstellt umfassende Vergleiche
    """
    
    def __init__(self, max_assets: int = 20, quick_mode: bool = False, category: str = "majors"):
        """
        Initialisiert das Universal Backtesting System
        
        Args:
            max_assets: Maximale Anzahl Assets für Tests
            quick_mode: Schneller Modus mit reduzierten Ranges für Tests
            category: Asset-Kategorie ("majors", "alts", "memes")
        """
        self.max_assets = max_assets
        self.quick_mode = quick_mode
        self.category = category
        
        # Kategorie-spezifische CSV-Datei
        self.assets_csv = f"backtesting_{category}.csv"
        
        # Ergebnis-Ordner mit Kategorie
        results_dir_name = f"universal_backtesting_results_{category}" if category != "majors" else "universal_backtesting_results"
        self.results_dir = os.path.join(os.path.dirname(__file__), results_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Indikator-Konfigurationen
        self.indicator_configs = {
            'EMA': {
                'system_class': EMABacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'Price > EMA = Long',
                'threshold': None,
                'type': 'single_param'
            },
            'RSI': {
                'system_class': RSIBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'RSI > 50 = Long',
                'threshold': 50.0,
                'type': 'single_param'
            },
            'CCI': {
                'system_class': CCIBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'CCI > 0 = Long',
                'threshold': 0.0,
                'type': 'single_param'
            },
            'ADX': {
                'system_class': ADXBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'ADX > 25 = Long',
                'threshold': 25.0,
                'type': 'single_param'
            },
            'AROON': {
                'system_class': AROONBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'Aroon > 0 = Long',
                'threshold': 0.0,
                'type': 'single_param'
            },
            'CMO': {
                'system_class': CMOBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'CMO > 0 = Long',
                'threshold': 0.0,
                'type': 'single_param'
            },
            'MFI': {
                'system_class': MFIBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'MFI > 50 = Long',
                'threshold': 50.0,
                'type': 'single_param'
            },
            'WILLR': {
                'system_class': WILLRBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'WILLR > -50 = Long',
                'threshold': -50.0,
                'type': 'single_param'
            },
            'DI': {
                'system_class': DIBacktestingSystem,
                'range': range(5, 26) if quick_mode else range(1, 151),
                'description': '+DI > -DI = Long',
                'threshold': 0.0,
                'type': 'matrix'
            },
            'MACD': {
                'system_class': MACDBacktestingSystem,
                'range': range(5, 26) if quick_mode else range(2, 151),
                'description': 'MACD > Signal = Long',
                'threshold': 0.0,
                'type': 'matrix'
            },
            'APO': {
                'system_class': APOBacktestingSystem,
                'range': range(5, 26) if quick_mode else range(2, 151),
                'description': 'APO > 0 = Long',
                'threshold': 0.0,
                'type': 'matrix'
            },
            'MOM': {
                'system_class': MOMBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'MOM > 0 = Long',
                'threshold': 0.0,
                'type': 'single_param'
            },
            'PPO': {
                'system_class': PPOBacktestingSystem,
                'range': range(5, 26) if quick_mode else range(5, 151),
                'description': 'PPO > 0 = Long',
                'threshold': 0.0,
                'type': 'matrix'
            },
            'TRIX': {
                'system_class': TRIXBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'TRIX > 0 = Long',
                'threshold': 0.0,
                'type': 'single_param'
            },
            'ADOSC': {
                'system_class': ADOSCBacktestingSystem,
                'range': range(5, 26) if quick_mode else range(2, 151),
                'description': 'ADOSC > 0 = Long (Volume-based)',
                'threshold': 0.0,
                'type': 'matrix'
            },
            'VIDYA': {
                'system_class': VIDYABacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'VIDYA steigend = Long',
                'threshold': None,
                'type': 'single_param'
            },
            'TRENDCONT': {
                'system_class': TrendContinuationBacktestingSystem,
                'range': range(5, 26) if quick_mode else range(5, 151),
                'description': 'Uptrend = Long (Dual HMA)',
                'threshold': None,
                'type': 'matrix'
            },
            'HULLSUITE': {
                'system_class': HullSuiteBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'MHULL > SHULL = Long',
                'threshold': None,
                'type': 'single_param'
            },
            'FSVZO': {
                'system_class': FSVZOBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'VZO > Signal = Long (Volume)',
                'threshold': 0.0,
                'type': 'single_param'
            },
            'BBPCT': {
                'system_class': BBPctBacktestingSystem,
                'range': range(5, 51) if quick_mode else range(5, 151),
                'description': 'Position > 50 = Long (BB %)',
                'threshold': 50.0,
                'type': 'single_param'
            },
            'FRAMA': {
                'system_class': FRAMABacktestingSystem,
                'range': range(6, 51, 2) if quick_mode else range(6, 101, 2),  # nur gerade Zahlen
                'description': 'Signal = 1: Long (FRAMA Channel)',
                'threshold': None,
                'type': 'single_param'
            },
            'SUPERTREND': {
                'system_class': SupertrendBacktestingSystem,
                'range': None,  # Matrix-basiert (ATR Period × Factor)
                'description': 'Signal = 1: Long (Supertrend)',
                'threshold': None,
                'type': 'matrix'
            },
            'MPT': {
                'system_class': MultiPivotTrendBacktestingSystem,
                'range': range(2, 11) if quick_mode else range(2, 21),
                'description': 'Signal > 0.3: Long (Multi Pivot Trend)',
                'threshold': 0.3,
                'type': 'single_param'
            }
        }
        
        self.all_results = {}
        self.summary_results = []
        
    def run_all_backtests(self):
        """
        Startet alle Backtesting-Systeme nacheinander
        """
        # Kategorie-Emoji
        category_emoji = {"majors": "👑", "alts": "🚀", "memes": "🐕"}.get(self.category, "💎")
        
        print(f"🚀 UNIVERSAL BACKTESTING SYSTEM START - {category_emoji} {self.category.upper()}")
        print("=" * 80)
        print(f"⚙️ KONFIGURATION:")
        print(f"   • Kategorie: {self.category.upper()}")
        print(f"   • Modus: {'Quick Mode' if self.quick_mode else 'Full Mode'}")
        print(f"   • Max Assets: {self.max_assets}")
        print(f"   • Indikatoren: {len(self.indicator_configs)}")
        print(f"   • Assets CSV: {self.assets_csv}")
        print(f"   • Ergebnis-Ordner: {self.results_dir}")
        print()
        
        total_indicators = len(self.indicator_configs)
        completed = 0
        
        for indicator_name, config in self.indicator_configs.items():
            try:
                completed += 1
                print(f"\n{'='*80}")
                print(f"[{completed}/{total_indicators}] 📊 Starte {indicator_name} Backtesting ({self.category.upper()})...")
                print(f"{'='*80}")
                
                # Bestimme Data-Folder basierend auf Kategorie
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_folder = os.path.join(current_dir, "price_data", self.category)
                
                # Bestimme Results-Folder mit Kategorie-Suffix
                results_folder_suffix = f"_{self.category}" if self.category != "majors" else ""
                strategy_name = indicator_name
                results_folder = os.path.join(
                    os.path.dirname(__file__),
                    "details",
                    f"{strategy_name.lower()}_backtesting_results{results_folder_suffix}"
                )
                
                # Initialisiere System mit korrekten Parametern
                # Prüfe welche Parameter das System akzeptiert
                import inspect
                init_signature = inspect.signature(config['system_class'].__init__)
                init_params = init_signature.parameters
                
                # Bestimme vollständigen absoluten Pfad zur CSV-Datei (WICHTIG!)
                csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.assets_csv))
                
                # Prüfe ob CSV existiert
                if not os.path.exists(csv_path):
                    print(f"❌ CSV-Datei nicht gefunden: {csv_path}")
                    continue
                
                # Debug: Zeige was geladen wird
                print(f"📄 Lade Assets aus: {csv_path}")
                
                # Baue kwargs basierend auf TATSÄCHLICH verfügbaren Parametern
                system_kwargs = {}
                
                # Prüfe ob System **kwargs akzeptiert (flexibel für unbekannte Parameter)
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values())
                
                # Füge Parameter hinzu, die das System akzeptiert ODER wenn es **kwargs hat
                if 'max_assets' in init_params or has_kwargs:
                    system_kwargs['max_assets'] = self.max_assets
                
                if 'assets_csv' in init_params or has_kwargs:
                    system_kwargs['assets_csv'] = csv_path
                elif 'majors_csv' in init_params:
                    system_kwargs['majors_csv'] = csv_path
                    
                if 'data_folder' in init_params or has_kwargs:
                    system_kwargs['data_folder'] = data_folder
                    
                if 'results_folder' in init_params or has_kwargs:
                    system_kwargs['results_folder'] = results_folder
                
                # Wenn das System KEINE assets_csv/data_folder Parameter hat UND keine **kwargs:
                # Dann müssen wir nach Initialisierung überschreiben
                needs_override = (
                    'assets_csv' not in init_params and 
                    'majors_csv' not in init_params and 
                    'data_folder' not in init_params and
                    not has_kwargs
                )
                
                if needs_override:
                    # Sehr alte Systeme: Initialisiere mit _skip_load Flag
                    system_kwargs['_skip_load'] = True  # Überspringe initialen Asset-Load
                    system = config['system_class'](**system_kwargs)
                    
                    # Setze die richtigen Ordner NACH Initialisierung
                    system.data_folder = data_folder
                    system.results_folder = results_folder
                    
                    # Jetzt lade Assets mit korrektem CSV-Pfad
                    system.major_assets = system.load_majors_from_csv(csv_path)
                    system.assets = system.major_assets
                    system.assets_data = {}
                    system.load_asset_data()
                else:
                    # Moderne Systeme oder solche mit **kwargs: Direkt initialisieren
                    system = config['system_class'](**system_kwargs)
                
                # Führe Backtest durch (single_param oder matrix)
                if config['type'] == 'single_param':
                    results_df = self._run_single_param_backtest(system, indicator_name, config)
                else:  # matrix
                    results_df = self._run_matrix_backtest(system, indicator_name, config)
                
                if results_df is not None and not results_df.empty:
                    # Speichere Ergebnisse
                    self.all_results[indicator_name] = results_df
                    
                    # Generiere detaillierte Reports (Details-Ordner)
                    try:
                        print(f"\n📋 Generiere detaillierte Analysen für {indicator_name}...")
                        system.generate_comprehensive_report(results_df)
                        print(f"✅ Detaillierte Analysen erstellt in {system.results_folder}")
                    except Exception as report_error:
                        print(f"⚠️ Fehler beim Erstellen der Detail-Reports: {report_error}")
                    
                    # Berechne Summary
                    summary = self._calculate_indicator_summary(results_df, indicator_name, config)
                    self.summary_results.append(summary)
                    
                    # Quick Preview
                    self._show_quick_preview(results_df, indicator_name, config)
                else:
                    print(f"⚠️ Keine Ergebnisse für {indicator_name}")
                    
            except Exception as e:
                print(f"❌ Fehler bei {indicator_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generiere umfassenden Vergleichsbericht
        if self.summary_results:
            self.generate_universal_report()
        
        print(f"\n🎉 UNIVERSAL BACKTESTING ABGESCHLOSSEN!")
        print(f"📁 Alle Ergebnisse in: {self.results_dir}/")
    
    def _run_single_param_backtest(self, system, indicator_name: str, config: dict) -> pd.DataFrame:
        """
        Führt Backtest für Single-Parameter Indikator durch
        """
        method_name = f'run_{indicator_name.lower()}_backtests'
        
        if hasattr(system, method_name):
            method = getattr(system, method_name)
            return method(config['range'])
        else:
            print(f"⚠️ Methode {method_name} nicht gefunden")
            return pd.DataFrame()
    
    def _run_matrix_backtest(self, system, indicator_name: str, config: dict) -> pd.DataFrame:
        """
        Führt Matrix-Backtest für Multi-Parameter Indikatoren durch (DI, MACD, APO, PPO, ADOSC)
        """
        if indicator_name == 'DI':
            return system.run_di_backtests(
                plus_di_range=config['range'],
                minus_di_range=config['range']
            )
        elif indicator_name == 'MACD':
            # Für MACD erweitere die Slow-Range, um genügend gültige Kombinationen zu haben
            fast_range = config['range']
            slow_range = range(max(config['range'].start + 2, 5), config['range'].stop + 10)
            
            print(f"📊 MACD Matrix: Fast {fast_range.start}-{fast_range.stop-1}, Slow {slow_range.start}-{slow_range.stop-1}")
            
            return system.run_macd_backtests(
                fast_period_range=fast_range,
                slow_period_range=slow_range
            )
        elif indicator_name == 'APO':
            # APO Matrix: Fast < Slow
            fast_range = config['range']
            slow_range = range(max(config['range'].start + 2, 5), config['range'].stop + 10)
            
            print(f"📊 APO Matrix: Fast {fast_range.start}-{fast_range.stop-1}, Slow {slow_range.start}-{slow_range.stop-1}")
            
            return system.run_apo_backtests(
                fast_period_range=fast_range,
                slow_period_range=slow_range
            )
        elif indicator_name == 'PPO':
            # PPO Matrix: Fast < Slow  
            fast_range = config['range']
            slow_range = range(max(config['range'].start + 1, 6), config['range'].stop + 5)
            
            print(f"📊 PPO Matrix: Fast {fast_range.start}-{fast_range.stop-1}, Slow {slow_range.start}-{slow_range.stop-1}")
            
            return system.run_ppo_backtests(
                fast_range=fast_range,
                slow_range=slow_range
            )
        elif indicator_name == 'ADOSC':
            # ADOSC Matrix: Fast < Slow (Volume-basiert)
            fast_range = config['range']
            slow_range = range(max(config['range'].start + 1, 4), config['range'].stop + 5)
            
            print(f"📊 ADOSC Matrix: Fast {fast_range.start}-{fast_range.stop-1}, Slow {slow_range.start}-{slow_range.stop-1}")
            print(f"⚠️ ADOSC benötigt Volume-Daten!")
            
            return system.run_adosc_backtests(
                fast_range=fast_range,
                slow_range=slow_range
            )
        elif indicator_name == 'TRENDCONT':
            # TRENDCONT Matrix: Fast < Slow (Dual HMA)
            fast_range = config['range']
            slow_range = range(max(config['range'].start + 1, 6), config['range'].stop + 5)
            
            print(f"📊 TRENDCONT Matrix: Fast {fast_range.start}-{fast_range.stop-1}, Slow {slow_range.start}-{slow_range.stop-1}")
            
            return system.run_trendcont_backtests(
                fast_range=fast_range,
                slow_range=slow_range
            )
        elif indicator_name == 'SUPERTREND':
            # SUPERTREND Matrix: ATR Period × Factor
            atr_range = range(5, 26) if self.quick_mode else range(5, 51)
            factor_range = np.arange(0.5, 5.1, 0.5) if self.quick_mode else np.arange(0.1, 10.1, 0.1)
            
            print(f"📊 SUPERTREND Matrix: ATR {atr_range.start}-{atr_range.stop-1}, Factor {factor_range[0]:.1f}-{factor_range[-1]:.1f}")
            
            return system.run_supertrend_backtests(
                atr_range=atr_range,
                factor_range=factor_range
            )
        else:
            return pd.DataFrame()
    
    def _calculate_indicator_summary(self, results_df: pd.DataFrame, indicator_name: str, config: dict) -> Dict:
        """
        Berechnet Summary-Statistiken für einen Indikator
        """
        try:
            # Grundstatistiken
            summary = {
                'indicator': indicator_name,
                'description': config['description'],
                'threshold': config['threshold'],
                'type': config['type'],
                'total_tests': len(results_df),
                'total_assets': results_df['asset'].nunique() if 'asset' in results_df.columns else 0,
            }
            
            # Performance-Metriken
            metrics = ['sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown', 'win_rate', 'omega_ratio', 'calmar_ratio']
            
            for metric in metrics:
                if metric in results_df.columns:
                    summary[f'best_{metric}'] = results_df[metric].max()
                    summary[f'avg_{metric}'] = results_df[metric].mean()
                    summary[f'worst_{metric}'] = results_df[metric].min()
                    summary[f'std_{metric}'] = results_df[metric].std()
            
            # Beste Konfiguration
            if 'sharpe_ratio' in results_df.columns:
                best_idx = results_df['sharpe_ratio'].idxmax()
                best_config = results_df.loc[best_idx]
                
                if config['type'] == 'single_param':
                    # Single Parameter (z.B. RSI-Länge)
                    # Spezielle Spaltennamen für manche Indikatoren
                    param_col_map = {
                        'fsvzo': 'vzo_length',
                        'hullsuite': 'hull_length'
                    }
                    param_col = param_col_map.get(indicator_name.lower(), f"{indicator_name.lower()}_length")
                    if param_col in best_config:
                        summary['best_parameter'] = int(best_config[param_col])
                elif config['type'] == 'matrix':
                    # Matrix Parameter (z.B. DI: +DI/-DI, MACD: Fast/Slow)
                    if indicator_name == 'DI':
                        summary['best_plus_di'] = int(best_config['plus_di_length'])
                        summary['best_minus_di'] = int(best_config['minus_di_length'])
                        summary['best_parameter'] = f"+DI:{int(best_config['plus_di_length'])}, -DI:{int(best_config['minus_di_length'])}"
                    elif indicator_name == 'MACD':
                        summary['best_fast_ema'] = int(best_config['fast_period'])
                        summary['best_slow_ema'] = int(best_config['slow_period'])
                        summary['best_parameter'] = f"Fast:{int(best_config['fast_period'])}, Slow:{int(best_config['slow_period'])}"
                    elif indicator_name == 'TRENDCONT':
                        # TRENDCONT verwendet fast_hma und slow_hma
                        if 'fast_hma' in best_config:
                            summary['best_fast_hma'] = int(best_config['fast_hma'])
                            summary['best_slow_hma'] = int(best_config['slow_hma'])
                            summary['best_parameter'] = f"Fast HMA:{int(best_config['fast_hma'])}, Slow HMA:{int(best_config['slow_hma'])}"
                    elif indicator_name in ['APO', 'PPO', 'ADOSC']:
                        # APO, PPO, ADOSC verwenden fast_period und slow_period
                        if 'fast_period' in best_config:
                            summary['best_fast_period'] = int(best_config['fast_period'])
                            summary['best_slow_period'] = int(best_config['slow_period'])
                            summary['best_parameter'] = f"Fast:{int(best_config['fast_period'])}, Slow:{int(best_config['slow_period'])}"
                
                summary['best_asset'] = best_config['asset']
                summary['best_sharpe'] = best_config['sharpe_ratio']
                summary['best_return'] = best_config['total_return']
            
            return summary
            
        except Exception as e:
            print(f"Fehler bei Summary-Berechnung für {indicator_name}: {e}")
            return {'indicator': indicator_name, 'error': str(e)}
    
    def _show_quick_preview(self, results_df: pd.DataFrame, indicator_name: str, config: dict):
        """
        Zeigt Quick Preview der besten Ergebnisse
        """
        if results_df.empty or 'sharpe_ratio' not in results_df.columns:
            return
        
        # Top 3 Ergebnisse
        top_3 = results_df.nlargest(3, 'sharpe_ratio')
        
        print(f"\n📊 {indicator_name} TOP 3 ERGEBNISSE:")
        print("-" * 60)
        
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            if config['type'] == 'single_param':
                # Spezielle Spaltennamen für manche Indikatoren
                param_col_map = {
                    'fsvzo': 'vzo_length',
                    'hullsuite': 'hull_length'
                }
                param_col = param_col_map.get(indicator_name.lower(), f"{indicator_name.lower()}_length")
                param_info = f"Länge: {int(row[param_col])}" if param_col in row else "Länge: N/A"
            elif config['type'] == 'matrix':
                if indicator_name == 'DI':
                    param_info = f"+DI: {int(row['plus_di_length'])}, -DI: {int(row['minus_di_length'])}"
                elif indicator_name == 'MACD':
                    param_info = f"Fast: {int(row['fast_period'])}, Slow: {int(row['slow_period'])}"
                elif indicator_name == 'TRENDCONT':
                    if 'fast_hma' in row:
                        param_info = f"Fast HMA: {int(row['fast_hma'])}, Slow HMA: {int(row['slow_hma'])}"
                    else:
                        param_info = "HMA Matrix"
                elif indicator_name in ['APO', 'PPO', 'ADOSC']:
                    if 'fast_period' in row:
                        param_info = f"Fast: {int(row['fast_period'])}, Slow: {int(row['slow_period'])}"
                    else:
                        param_info = "Fast/Slow Matrix"
                else:
                    param_info = "Matrix"
            else:
                param_info = "Unknown"
            
            print(f"  {i}. {param_info} | {row['asset']} | "
                  f"Sharpe: {row['sharpe_ratio']:.3f} | Return: {row['total_return']:.1%} | DD: {row['max_drawdown']:.1%}")
    
    def generate_universal_report(self):
        """
        Generiert umfassenden Universal-Bericht mit allen Indikatoren
        """
        print(f"\n📋 Generiere Universal-Bericht...")
        
        # Erstelle Summary DataFrame
        summary_df = pd.DataFrame(self.summary_results)
        
        if summary_df.empty:
            print("❌ Keine Summary-Daten für Bericht")
            return
        
        # Generiere Vergleichs-Visualisierungen
        self._create_comparison_charts(summary_df)
        
        # Generiere Text-Bericht
        self._generate_text_report(summary_df)
        
        # Generiere Top Kalibrierungen Report
        self._generate_top_calibrations_report()
        
        # Speichere CSV-Daten
        csv_path = os.path.join(self.results_dir, 'universal_backtesting_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"📊 Summary CSV gespeichert: {csv_path}")
        
        # Speichere individuelle Ergebnisse
        for indicator, results in self.all_results.items():
            indicator_csv = os.path.join(self.results_dir, f'{indicator.lower()}_detailed_results.csv')
            results.to_csv(indicator_csv, index=False)
            print(f"📊 {indicator} Details gespeichert: {indicator_csv}")
    
    def _create_comparison_charts(self, summary_df: pd.DataFrame):
        """
        Erstellt Vergleichs-Charts für alle Indikatoren
        """
        try:
            # Chart 1: Beste Sharpe Ratios
            plt.figure(figsize=(15, 8))
            
            # Sortiere nach bester Sharpe Ratio
            sorted_df = summary_df.sort_values('best_sharpe_ratio', ascending=True)
            
            bars = plt.barh(sorted_df['indicator'], sorted_df['best_sharpe_ratio'])
            
            # Farbkodierung basierend auf Performance
            colors = plt.cm.RdYlGn([x/sorted_df['best_sharpe_ratio'].max() for x in sorted_df['best_sharpe_ratio']])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.title('Beste Sharpe Ratios aller Indikatoren', fontsize=16, fontweight='bold')
            plt.xlabel('Sharpe Ratio', fontsize=12)
            plt.ylabel('Indikator', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            chart_path = os.path.join(self.results_dir, 'indicators_sharpe_comparison.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Sharpe Ratio Vergleich gespeichert: {chart_path}")
            
            # Chart 2: Returns vs Drawdown Scatter Plot
            plt.figure(figsize=(12, 8))
            
            scatter = plt.scatter(summary_df['best_max_drawdown'] * 100, 
                                summary_df['best_total_return'] * 100,
                                s=summary_df['best_sharpe_ratio'] * 50,  # Größe basiert auf Sharpe
                                c=summary_df['best_sharpe_ratio'],
                                cmap='RdYlGn',
                                alpha=0.7)
            
            # Beschriftungen
            for i, row in summary_df.iterrows():
                plt.annotate(row['indicator'], 
                           (row['best_max_drawdown'] * 100, row['best_total_return'] * 100),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            plt.colorbar(scatter, label='Sharpe Ratio')
            plt.xlabel('Maximaler Drawdown (%)', fontsize=12)
            plt.ylabel('Gesamtrendite (%)', fontsize=12)
            plt.title('Return vs. Drawdown aller Indikatoren\n(Blasengröße = Sharpe Ratio)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            scatter_path = os.path.join(self.results_dir, 'return_vs_drawdown_scatter.png')
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Return vs Drawdown Scatter gespeichert: {scatter_path}")
            
            # Chart 3: Durchschnitts-Performance Heatmap
            metrics_for_heatmap = ['avg_sharpe_ratio', 'avg_total_return', 'avg_max_drawdown', 'avg_win_rate']
            available_metrics = [m for m in metrics_for_heatmap if m in summary_df.columns]
            
            if available_metrics:
                heatmap_data = summary_df.set_index('indicator')[available_metrics]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap_data.T, 
                          annot=True, 
                          fmt='.3f', 
                          cmap='RdYlGn',
                          center=0,
                          cbar_kws={'label': 'Performance'})
                
                plt.title('Durchschnitts-Performance Matrix aller Indikatoren', fontsize=14, fontweight='bold')
                plt.xlabel('Indikator', fontsize=12)
                plt.ylabel('Metrik', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                heatmap_path = os.path.join(self.results_dir, 'performance_heatmap.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Performance Heatmap gespeichert: {heatmap_path}")
                
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen der Charts: {e}")
    
    def _generate_text_report(self, summary_df: pd.DataFrame):
        """
        Generiert detaillierten Text-Bericht
        """
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("🌟 UNIVERSAL BACKTESTING SYSTEM - COMPREHENSIVE REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📊 Modus: {'Quick Mode' if self.quick_mode else 'Full Mode'}")
        report_lines.append(f"🎯 Getestete Indikatoren: {len(summary_df)}")
        report_lines.append(f"💎 Max Assets pro Test: {self.max_assets}")
        report_lines.append("")
        
        # Gesamt-Übersicht
        total_tests = summary_df['total_tests'].sum()
        report_lines.append("📈 GESAMT-ÜBERSICHT:")
        report_lines.append("-" * 60)
        report_lines.append(f"   • Gesamte Backtests: {total_tests:,}")
        report_lines.append(f"   • Beste Overall Sharpe Ratio: {summary_df['best_sharpe_ratio'].max():.3f}")
        report_lines.append(f"   • Beste Overall Return: {summary_df['best_total_return'].max():.1%}")
        report_lines.append(f"   • Niedrigster Drawdown: {summary_df['best_max_drawdown'].min():.1%}")
        report_lines.append("")
        
        # Ranking nach Sharpe Ratio
        sorted_by_sharpe = summary_df.sort_values('best_sharpe_ratio', ascending=False)
        report_lines.append("🏆 INDIKATOR-RANKING (Beste Sharpe Ratio):")
        report_lines.append("-" * 60)
        
        for i, (_, row) in enumerate(sorted_by_sharpe.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            
            report_lines.append(
                f"{medal} {row['indicator']:6s} | "
                f"Sharpe: {row['best_sharpe_ratio']:6.3f} | "
                f"Return: {row['best_total_return']:7.1%} | "
                f"DD: {row['best_max_drawdown']:6.1%} | "
                f"Param: {row.get('best_parameter', 'N/A')}"
            )
        
        # Detaillierte Indikator-Analyse
        report_lines.append("")
        report_lines.append("🔍 DETAILLIERTE INDIKATOR-ANALYSE:")
        report_lines.append("=" * 80)
        
        for _, row in summary_df.iterrows():
            report_lines.append(f"\n📊 {row['indicator']} ({row['description']}):")
            report_lines.append("-" * 50)
            report_lines.append(f"   🎯 Threshold: {row['threshold']}")
            report_lines.append(f"   📈 Typ: {row['type']}")
            report_lines.append(f"   🧪 Tests: {row['total_tests']:,}")
            report_lines.append(f"   💎 Assets: {row['total_assets']}")
            
            if 'best_parameter' in row:
                report_lines.append(f"   ⚙️ Beste Konfiguration: {row['best_parameter']}")
            
            if 'best_asset' in row:
                report_lines.append(f"   🏅 Bestes Asset: {row['best_asset']}")
            
            # Performance-Metriken
            metrics_info = []
            if 'best_sharpe_ratio' in row:
                metrics_info.append(f"Sharpe: {row['best_sharpe_ratio']:.3f}")
            if 'best_total_return' in row:
                metrics_info.append(f"Return: {row['best_total_return']:.1%}")
            if 'best_max_drawdown' in row:
                metrics_info.append(f"DD: {row['best_max_drawdown']:.1%}")
            if 'best_win_rate' in row:
                metrics_info.append(f"Win Rate: {row['best_win_rate']:.1%}")
            
            if metrics_info:
                report_lines.append(f"   📊 Performance: {' | '.join(metrics_info)}")
        
        # Strategische Empfehlungen
        report_lines.append("")
        report_lines.append("💡 STRATEGISCHE EMPFEHLUNGEN:")
        report_lines.append("-" * 60)
        
        # Top 3 Indikatoren
        top_3 = sorted_by_sharpe.head(3)
        report_lines.append("🔥 TOP 3 EMPFOHLENE INDIKATOREN:")
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            report_lines.append(f"   {i}. {row['indicator']}: {row['description']} (Sharpe: {row['best_sharpe_ratio']:.3f})")
        
        # Konsistenz-Analyse (niedrige Standardabweichung)
        if 'std_sharpe_ratio' in summary_df.columns:
            consistent = summary_df.loc[summary_df['std_sharpe_ratio'].idxmin()]
            report_lines.append(f"\n🎯 KONSISTENTESTER INDIKATOR: {consistent['indicator']} (Std: {consistent['std_sharpe_ratio']:.3f})")
        
        # Risiko-Analyse (niedrigster Drawdown)
        safest = summary_df.loc[summary_df['best_max_drawdown'].idxmin()]
        report_lines.append(f"🛡️ SICHERSTER INDIKATOR: {safest['indicator']} (DD: {safest['best_max_drawdown']:.1%})")
        
        # Return-Champion
        highest_return = summary_df.loc[summary_df['best_total_return'].idxmax()]
        report_lines.append(f"💰 HÖCHSTE RETURNS: {highest_return['indicator']} (Return: {highest_return['best_total_return']:.1%})")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("🏁 UNIVERSAL BACKTESTING ANALYSIS ABGESCHLOSSEN")
        report_lines.append("=" * 80)
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_dir, 'universal_backtesting_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 Universal Report gespeichert: {report_path}")
        
        # Zeige auch auf Konsole (gekürzt)
        print("\n" + "=" * 80)
        print("🌟 UNIVERSAL BACKTESTING SUMMARY")
        print("=" * 80)
        print(f"🏆 CHAMPION: {sorted_by_sharpe.iloc[0]['indicator']} (Sharpe: {sorted_by_sharpe.iloc[0]['best_sharpe_ratio']:.3f})")
        print(f"💰 HIGHEST RETURN: {highest_return['indicator']} ({highest_return['best_total_return']:.1%})")
        print(f"🛡️ SAFEST: {safest['indicator']} (DD: {safest['best_max_drawdown']:.1%})")
        print(f"📊 Total Tests: {total_tests:,}")
        print("=" * 80)
    
    def _generate_top_calibrations_report(self):
        """
        Generiert detaillierten Report mit den Top 10 Kalibrierungen für jeden Indikator
        """
        print(f"\n📋 Erstelle Top Kalibrierungen Report...")
        
        all_top_calibrations = {}
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("🎯 TOP 10 KALIBRIERUNGEN PRO INDIKATOR - DETAILLIERTER REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📊 Modus: {'Quick Mode' if self.quick_mode else 'Full Mode'}")
        report_lines.append("")
        
        for indicator_name, results_df in self.all_results.items():
            if results_df.empty or 'sharpe_ratio' not in results_df.columns:
                continue
            
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"📊 {indicator_name.upper()} - TOP 10 KALIBRIERUNGEN")
            report_lines.append(f"{'='*80}")
            
            config = self.indicator_configs[indicator_name]
            
            if config['type'] == 'single_param':
                # Single Parameter Indikator - Gruppiere nach Parameter-Länge
                # Spezielle Spaltennamen für manche Indikatoren
                param_col_map = {
                    'fsvzo': 'vzo_length',
                    'hullsuite': 'hull_length'
                }
                param_col = param_col_map.get(indicator_name.lower(), f"{indicator_name.lower()}_length")
                
                if param_col in results_df.columns:
                    # Berechne Durchschnitts-Performance pro Parameter
                    avg_performance = results_df.groupby(param_col).agg({
                        'sharpe_ratio': ['mean', 'std', 'count'],
                        'total_return': ['mean', 'std'],
                        'max_drawdown': ['mean', 'std'],
                        'sortino_ratio': ['mean', 'std'],
                        'win_rate': ['mean', 'std'],
                        'omega_ratio': ['mean', 'std'],
                        'calmar_ratio': ['mean', 'std']
                    }).round(4)
                    
                    # Flache Spalten für einfachere Verwendung
                    avg_performance.columns = ['_'.join(col).strip() for col in avg_performance.columns]
                    avg_performance = avg_performance.reset_index()
                    
                    # Berechne kombinierten Score
                    avg_performance['combined_score'] = (
                        avg_performance['sharpe_ratio_mean'] * 0.3 +
                        avg_performance['sortino_ratio_mean'] * 0.3 +
                        avg_performance['total_return_mean'] * 0.2 +
                        (1 - avg_performance['max_drawdown_mean']) * 0.2
                    )
                    
                    # Top 10 nach kombiniertem Score
                    top_10 = avg_performance.nlargest(10, 'combined_score')
                    
                    report_lines.append(f"📈 Strategie: {config['description']}")
                    report_lines.append(f"🎯 Threshold: {config['threshold']}")
                    report_lines.append(f"📊 Parameter: {param_col.replace('_length', '').upper()}-Länge")
                    report_lines.append(f"🧪 Getestete Parameter: {results_df[param_col].nunique()}")
                    report_lines.append(f"💎 Tests pro Parameter: Ø {avg_performance['sharpe_ratio_count'].mean():.1f}")
                    report_lines.append("")
                    
                    report_lines.append("🏆 TOP 10 PARAMETER-KALIBRIERUNGEN:")
                    report_lines.append("-" * 80)
                    
                    for i, (_, row) in enumerate(top_10.iterrows(), 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                        
                        param_value = int(row[param_col])
                        sharpe = row['sharpe_ratio_mean']
                        sharpe_std = row['sharpe_ratio_std']
                        total_return = row['total_return_mean']
                        drawdown = row['max_drawdown_mean']
                        win_rate = row['win_rate_mean']
                        tests = int(row['sharpe_ratio_count'])
                        score = row['combined_score']
                        
                        report_lines.append(
                            f"{medal} Parameter: {param_value:3d} | "
                            f"Sharpe: {sharpe:6.3f}±{sharpe_std:.3f} | "
                            f"Return: {total_return:7.1%} | "
                            f"DD: {drawdown:6.1%} | "
                            f"Win: {win_rate:5.1%} | "
                            f"Tests: {tests:2d} | "
                            f"Score: {score:.3f}"
                        )
                    
                    # Speichere für CSV
                    top_10_clean = top_10.copy()
                    top_10_clean['indicator'] = indicator_name
                    top_10_clean['parameter_type'] = 'single'
                    all_top_calibrations[indicator_name] = top_10_clean
                    
            elif config['type'] == 'matrix':
                # Matrix Parameter Indikator - Verschiedene Behandlung für DI und MACD
                if indicator_name == 'DI':
                    # DI: +DI und -DI Parameter
                    avg_performance = results_df.groupby(['plus_di_length', 'minus_di_length']).agg({
                        'sharpe_ratio': ['mean', 'std', 'count'],
                        'total_return': ['mean', 'std'],
                        'max_drawdown': ['mean', 'std'],
                        'sortino_ratio': ['mean', 'std'],
                        'win_rate': ['mean', 'std'],
                        'omega_ratio': ['mean', 'std']
                    }).round(4)
                    
                    avg_performance.columns = ['_'.join(col).strip() for col in avg_performance.columns]
                    avg_performance = avg_performance.reset_index()
                    
                    # DI-spezifischer kombinierter Score
                    avg_performance['combined_score'] = (
                        avg_performance['sharpe_ratio_mean'] * 0.35 +
                        avg_performance['sortino_ratio_mean'] * 0.25 +
                        avg_performance['total_return_mean'] * 0.2 +
                        (1 - avg_performance['max_drawdown_mean']) * 0.2
                    )
                    
                    top_10 = avg_performance.nlargest(10, 'combined_score')
                    
                    report_lines.append(f"📈 Strategie: {config['description']}")
                    report_lines.append(f"🎯 Threshold: {config['threshold']}")
                    report_lines.append(f"📊 Parameter: +DI/-DI Matrix")
                    report_lines.append(f"🧪 Getestete Kombinationen: {len(avg_performance)}")
                    report_lines.append(f"💎 Tests pro Kombination: Ø {avg_performance['sharpe_ratio_count'].mean():.1f}")
                    report_lines.append("")
                    
                    report_lines.append("🏆 TOP 10 DI-KOMBINATIONEN:")
                    report_lines.append("-" * 90)
                    
                    for i, (_, row) in enumerate(top_10.iterrows(), 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                        
                        plus_di = int(row['plus_di_length'])
                        minus_di = int(row['minus_di_length'])
                        sharpe = row['sharpe_ratio_mean']
                        sharpe_std = row['sharpe_ratio_std']
                        total_return = row['total_return_mean']
                        drawdown = row['max_drawdown_mean']
                        tests = int(row['sharpe_ratio_count'])
                        score = row['combined_score']
                        
                        report_lines.append(
                            f"{medal} +DI({plus_di:2d}) / -DI({minus_di:2d}) | "
                            f"Sharpe: {sharpe:6.3f}±{sharpe_std:.3f} | "
                            f"Return: {total_return:7.1%} | "
                            f"DD: {drawdown:6.1%} | "
                            f"Tests: {tests:2d} | "
                            f"Score: {score:.3f}"
                        )
                
                elif indicator_name == 'MACD':
                    # MACD: Fast und Slow EMA Parameter
                    avg_performance = results_df.groupby(['fast_period', 'slow_period']).agg({
                        'sharpe_ratio': ['mean', 'std', 'count'],
                        'total_return': ['mean', 'std'],
                        'max_drawdown': ['mean', 'std'],
                        'sortino_ratio': ['mean', 'std'],
                        'win_rate': ['mean', 'std'],
                        'omega_ratio': ['mean', 'std']
                    }).round(4)
                    
                    avg_performance.columns = ['_'.join(col).strip() for col in avg_performance.columns]
                    avg_performance = avg_performance.reset_index()
                    
                    # Berechne EMA-Verhältnis
                    avg_performance['ema_ratio'] = avg_performance['slow_period'] / avg_performance['fast_period']
                    
                    # MACD-spezifischer kombinierter Score
                    avg_performance['combined_score'] = (
                        avg_performance['sharpe_ratio_mean'] * 0.3 +
                        avg_performance['sortino_ratio_mean'] * 0.3 +
                        avg_performance['total_return_mean'] * 0.2 +
                        (1 - avg_performance['max_drawdown_mean']) * 0.2
                    )
                    
                    top_10 = avg_performance.nlargest(10, 'combined_score')
                    
                    report_lines.append(f"📈 Strategie: {config['description']}")
                    report_lines.append(f"🎯 Threshold: {config['threshold']}")
                    report_lines.append(f"📊 Parameter: Fast EMA / Slow EMA Matrix")
                    report_lines.append(f"🧪 Getestete Kombinationen: {len(avg_performance)}")
                    report_lines.append(f"💎 Tests pro Kombination: Ø {avg_performance['sharpe_ratio_count'].mean():.1f}")
                    report_lines.append("")
                    
                    report_lines.append("🏆 TOP 10 MACD-KOMBINATIONEN:")
                    report_lines.append("-" * 90)
                    
                    for i, (_, row) in enumerate(top_10.iterrows(), 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                        
                        fast = int(row['fast_period'])
                        slow = int(row['slow_period'])
                        ratio = row['ema_ratio']
                        sharpe = row['sharpe_ratio_mean']
                        sharpe_std = row['sharpe_ratio_std']
                        total_return = row['total_return_mean']
                        drawdown = row['max_drawdown_mean']
                        tests = int(row['sharpe_ratio_count'])
                        score = row['combined_score']
                        
                        report_lines.append(
                            f"{medal} Fast({fast:2d}) / Slow({slow:2d}) | "
                            f"Ratio: {ratio:4.1f} | "
                            f"Sharpe: {sharpe:6.3f}±{sharpe_std:.3f} | "
                            f"Return: {total_return:7.1%} | "
                            f"DD: {drawdown:6.1%} | "
                            f"Score: {score:.3f}"
                        )
                
                elif indicator_name in ['APO', 'PPO', 'ADOSC', 'TRENDCONT']:
                    # Generische Matrix-Behandlung für APO, PPO, ADOSC, TRENDCONT
                    # Bestimme Spaltennamen basierend auf Indikator
                    if indicator_name == 'TRENDCONT':
                        fast_col, slow_col = 'fast_hma', 'slow_hma'
                        indicator_label = 'HMA'
                    else:
                        fast_col, slow_col = 'fast_period', 'slow_period'
                        indicator_label = 'Period'
                    
                    if fast_col in results_df.columns and slow_col in results_df.columns:
                        avg_performance = results_df.groupby([fast_col, slow_col]).agg({
                            'sharpe_ratio': ['mean', 'std', 'count'],
                            'total_return': ['mean', 'std'],
                            'max_drawdown': ['mean', 'std'],
                            'sortino_ratio': ['mean', 'std'],
                            'win_rate': ['mean', 'std'],
                            'omega_ratio': ['mean', 'std']
                        }).round(4)
                        
                        avg_performance.columns = ['_'.join(col).strip() for col in avg_performance.columns]
                        avg_performance = avg_performance.reset_index()
                        
                        # Kombinierter Score
                        avg_performance['combined_score'] = (
                            avg_performance['sharpe_ratio_mean'] * 0.3 +
                            avg_performance['sortino_ratio_mean'] * 0.3 +
                            avg_performance['total_return_mean'] * 0.2 +
                            (1 - avg_performance['max_drawdown_mean']) * 0.2
                        )
                        
                        top_10 = avg_performance.nlargest(10, 'combined_score')
                        
                        report_lines.append(f"📈 Strategie: {config['description']}")
                        report_lines.append(f"🎯 Threshold: {config['threshold']}")
                        report_lines.append(f"📊 Parameter: Fast/Slow {indicator_label} Matrix")
                        report_lines.append(f"🧪 Getestete Kombinationen: {len(avg_performance)}")
                        report_lines.append(f"💎 Tests pro Kombination: Ø {avg_performance['sharpe_ratio_count'].mean():.1f}")
                        report_lines.append("")
                        
                        report_lines.append(f"🏆 TOP 10 {indicator_name}-KOMBINATIONEN:")
                        report_lines.append("-" * 90)
                        
                        for i, (_, row) in enumerate(top_10.iterrows(), 1):
                            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                            
                            fast = int(row[fast_col])
                            slow = int(row[slow_col])
                            sharpe = row['sharpe_ratio_mean']
                            sharpe_std = row['sharpe_ratio_std']
                            total_return = row['total_return_mean']
                            drawdown = row['max_drawdown_mean']
                            tests = int(row['sharpe_ratio_count'])
                            score = row['combined_score']
                            
                            report_lines.append(
                                f"{medal} Fast({fast:2d}) / Slow({slow:2d}) | "
                                f"Sharpe: {sharpe:6.3f}±{sharpe_std:.3f} | "
                                f"Return: {total_return:7.1%} | "
                                f"DD: {drawdown:6.1%} | "
                                f"Tests: {tests:2d} | "
                                f"Score: {score:.3f}"
                            )
                
                # Speichere Matrix-Ergebnisse für CSV
                if indicator_name in ['DI', 'MACD', 'APO', 'PPO', 'ADOSC', 'TRENDCONT']:
                    top_10_clean = top_10.copy()
                    top_10_clean['indicator'] = indicator_name
                    top_10_clean['parameter_type'] = 'matrix'
                    all_top_calibrations[indicator_name] = top_10_clean
            
            report_lines.append("")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("🏁 TOP KALIBRIERUNGEN ANALYSE ABGESCHLOSSEN")
        report_lines.append("=" * 80)
        
        # Speichere Text-Report
        report_text = "\n".join(report_lines)
        calibrations_report_path = os.path.join(self.results_dir, 'top_10_calibrations_detailed_report.txt')
        
        with open(calibrations_report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 Top Kalibrierungen Report gespeichert: {calibrations_report_path}")
        
        # Speichere CSV-Daten für jede Indikator
        for indicator_name, top_calibrations_df in all_top_calibrations.items():
            calibrations_csv = os.path.join(self.results_dir, f'{indicator_name.lower()}_top_10_calibrations.csv')
            top_calibrations_df.to_csv(calibrations_csv, index=False)
            print(f"📊 {indicator_name} Top 10 Kalibrierungen CSV gespeichert: {calibrations_csv}")
        
        # Erstelle zusammengefasste CSV mit besten Kalibrierung pro Indikator
        best_calibrations_summary = []
        for indicator_name, top_calibrations_df in all_top_calibrations.items():
            if not top_calibrations_df.empty:
                best = top_calibrations_df.iloc[0]  # Beste Kalibrierung
                
                config = self.indicator_configs[indicator_name]
                summary_row = {
                    'indicator': indicator_name,
                    'description': config['description'],
                    'parameter_type': config['type'],
                    'best_sharpe_ratio': best['sharpe_ratio_mean'],
                    'best_return': best['total_return_mean'],
                    'best_drawdown': best['max_drawdown_mean'],
                    'combined_score': best['combined_score']
                }
                
                if config['type'] == 'single_param':
                    # Spezielle Spaltennamen für manche Indikatoren
                    param_col_map = {
                        'fsvzo': 'vzo_length',
                        'hullsuite': 'hull_length'
                    }
                    param_col = param_col_map.get(indicator_name.lower(), f"{indicator_name.lower()}_length")
                    if param_col in best:
                        summary_row['best_parameter'] = int(best[param_col])
                elif config['type'] == 'matrix':
                    if indicator_name == 'DI':
                        summary_row['best_plus_di'] = int(best['plus_di_length'])
                        summary_row['best_minus_di'] = int(best['minus_di_length'])
                        summary_row['best_parameter'] = f"+DI:{int(best['plus_di_length'])}, -DI:{int(best['minus_di_length'])}"
                    elif indicator_name == 'MACD':
                        summary_row['best_fast_ema'] = int(best['fast_period'])
                        summary_row['best_slow_ema'] = int(best['slow_period'])
                        summary_row['best_parameter'] = f"Fast:{int(best['fast_period'])}, Slow:{int(best['slow_period'])}"
                    elif indicator_name == 'TRENDCONT':
                        # TRENDCONT verwendet fast_hma und slow_hma
                        if 'fast_hma' in best:
                            summary_row['best_fast_hma'] = int(best['fast_hma'])
                            summary_row['best_slow_hma'] = int(best['slow_hma'])
                            summary_row['best_parameter'] = f"Fast HMA:{int(best['fast_hma'])}, Slow HMA:{int(best['slow_hma'])}"
                    elif indicator_name in ['APO', 'PPO', 'ADOSC']:
                        # APO, PPO, ADOSC verwenden fast_period und slow_period
                        if 'fast_period' in best:
                            summary_row['best_fast_period'] = int(best['fast_period'])
                            summary_row['best_slow_period'] = int(best['slow_period'])
                            summary_row['best_parameter'] = f"Fast:{int(best['fast_period'])}, Slow:{int(best['slow_period'])}"
                
                best_calibrations_summary.append(summary_row)
        
        # Speichere Best Calibrations Summary
        if best_calibrations_summary:
            best_summary_df = pd.DataFrame(best_calibrations_summary)
            best_summary_csv = os.path.join(self.results_dir, 'best_calibrations_summary.csv')
            best_summary_df.to_csv(best_summary_csv, index=False)
            print(f"📊 Beste Kalibrierungen Summary gespeichert: {best_summary_csv}")

def main():
    """
    Hauptfunktion für Universal Backtesting System
    """
    print("🌟 UNIVERSAL BACKTESTING SYSTEM - 23 INDIKATOREN")
    print("=" * 80)
    print("Startet ALLE 23 verfügbaren Backtesting-Systeme und erstellt umfassende Vergleiche")
    print("📊 18 Einzelparameter + 5 Matrix-Indikatoren")
    print("🚀 Enhanced Framework inkl. TradingView Indicators (FRAMA, Supertrend, MPT)")
    print()
    
    # Benutzer-Eingabe für Kategorie
    print("Wähle Asset-Kategorie:")
    print("1. 👑 Majors (BTC, ETH, BNB, SOL, etc.)")
    print("2. 🚀 Alts (UNI, AAVE, XMR, etc.)")
    print("3. 🐕 Memes (DOGE, SHIB, PEPE, BONK)")
    category_choice = input("Eingabe (1/2/3): ").strip()
    
    category_map = {"1": "majors", "2": "alts", "3": "memes"}
    category = category_map.get(category_choice, "majors")
    
    print()
    
    # Benutzer-Eingabe für Modus
    mode_choice = input("Wähle Modus:\n1. Quick Mode (schneller, kleinere Ranges)\n2. Full Mode (komplett, alle Parameter)\nEingabe (1/2): ").strip()
    
    quick_mode = mode_choice == "1"
    max_assets = 20
    
    if quick_mode:
        print(f"\n⚡ Quick Mode gewählt - Reduzierte Parameter-Ranges für {category.upper()}")
    else:
        print(f"\n🔥 Full Mode gewählt - Alle Parameter für {category.upper()}")
    
    print()
    
    try:
        # Erstelle Universal System mit Kategorie
        universal_system = UniversalBacktestingSystem(
            max_assets=max_assets,
            quick_mode=quick_mode,
            category=category
        )
        
        # Starte alle Backtests
        universal_system.run_all_backtests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Backtesting durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Fehler im Universal Backtesting System: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()