"""
Gemeinsame Backtesting-Funktionen für verschiedene Trading-Strategien

Dieses Modul enthält die wiederverwendbaren Funktionen für:
- Asset-Daten laden
- Performance-Metriken berechnen
- Backtesting durchführen
- Ergebnisse analysieren und visualisieren
- Berichte generieren

Verwendung in spezifischen Backtesting-Systemen (EMA, CCI, etc.)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BaseBacktestingSystem:
    """
    Basis-Klasse für Backtesting-Systeme mit gemeinsamen Funktionen
    """
    
    def __init__(self, max_assets: int = 20, strategy_name: str = "Unknown", 
                 data_folder: str = None, results_folder: str = None, 
                 assets_csv: str = None, majors_csv: str = None, 
                 _skip_load: bool = False, **kwargs):
        """
        Initialisiert das Basis-Backtesting System
        
        Args:
            max_assets: Maximale Anzahl der Assets für Tests
            strategy_name: Name der Strategie (z.B. "EMA", "CCI")
            data_folder: Ordner mit Preisdaten (None für auto-detection)
            results_folder: Ordner für Ergebnisse
            assets_csv: CSV-Datei mit Assets (neuer Parameter)
            majors_csv: CSV-Datei mit Assets (Legacy-Parameter, Fallback für assets_csv)
            _skip_load: Interner Parameter - überspringt initialen Asset-Load (für manuelle Konfiguration)
            **kwargs: Zusätzliche Parameter (für Kompatibilität ignoriert)
        """
        # Kompatibilität: Verwende majors_csv als Fallback wenn assets_csv nicht angegeben
        if assets_csv is None and majors_csv is None:
            assets_csv = "backtesting_majors.csv"
        elif assets_csv is None:
            assets_csv = majors_csv
        
        self.strategy_name = strategy_name
        self.max_assets = max_assets
        
        # Auto-detect data folder wenn nicht angegeben
        if data_folder is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Prüfe ob wir im backtest/ Unterordner sind
            if os.path.basename(current_dir) == 'backtest':
                # Wir sind im backtest/ Ordner, gehe ein Level hoch
                parent_dir = os.path.dirname(current_dir)
                data_folder = os.path.join(parent_dir, "price_data", "majors")
            else:
                # Wir sind im Hauptordner
                data_folder = os.path.join(current_dir, "price_data", "majors")
        
        # Default results folder
        if results_folder is None:
            # Bestimme automatisch den korrekten Pfad basierend auf dem aktuellen Verzeichnis
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Prüfe, ob wir im backtest/ Unterordner sind
            if os.path.basename(current_dir) == 'backtest':
                # Wir sind im backtest/ Ordner, gehe ein Level hoch zu integrated_backtesting/
                integrated_dir = os.path.dirname(current_dir)
                results_folder = os.path.join(integrated_dir, "details", f"{strategy_name.lower()}_backtesting_results")
            else:
                # Wir sind bereits im integrated_backtesting/ Ordner
                results_folder = os.path.join(current_dir, "details", f"{strategy_name.lower()}_backtesting_results")
        
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.assets_data = {}
        self.results_matrix = {}
        
        # Erstelle Ergebnisordner
        os.makedirs(results_folder, exist_ok=True)
        
        # Lade Assets nur wenn nicht explizit übersprungen
        if not _skip_load:
            # Lade Major Crypto Assets aus CSV
            self.major_assets = self.load_majors_from_csv(assets_csv)
            
            # Kompatibilität: Alias für ältere Code-Versionen
            self.assets = self.major_assets
            
            # Lade Asset-Daten
            self.load_asset_data()
        else:
            # Setze leere Listen wenn Load übersprungen wird
            self.major_assets = []
            self.assets = []
        
        print(f"📊 {strategy_name} Backtesting System initialisiert")
        print(f"📂 Daten-Ordner: {data_folder}")
        print(f"📁 Ergebnis-Ordner: {results_folder}")
        print(f"💎 Assets geladen: {len(self.assets_data)} von {len(self.major_assets)} verfügbar")
    
    def load_majors_from_csv(self, csv_file: str) -> List[str]:
        """
        Lädt Assets aus CSV-Datei
        
        Args:
            csv_file: Pfad zur CSV-Datei mit Assets (kann vollständiger Pfad oder Dateiname sein)
            
        Returns:
            Liste der Asset-Symbole
        """
        assets = []
        csv_path = None
        
        # Prüfe zuerst, ob csv_file ein vollständiger absoluter Pfad ist und existiert
        if os.path.isabs(csv_file):
            if os.path.exists(csv_file):
                csv_path = csv_file
            else:
                print(f"⚠️ Absoluter Pfad existiert nicht: {csv_file}")
                return self._get_fallback_assets()
        else:
            # Suche die CSV-Datei in verschiedenen möglichen Pfaden (relativer Pfad)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            possible_paths = [
                csv_file,  # Direkter Pfad (falls relativ)
                os.path.join(current_dir, csv_file),  # Im aktuellen Ordner (integrated_backtesting/)
                os.path.join("integrated_backtesting", csv_file),  # Im integrated_backtesting Ordner
                os.path.join("..", "integrated_backtesting", csv_file)  # Eine Ebene höher
            ]
            
            # Wenn wir im backtest/ Unterordner sind, füge parent directory hinzu
            if os.path.basename(current_dir) == 'backtest':
                possible_paths.append(os.path.join(os.path.dirname(current_dir), csv_file))
            
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        
            if not csv_path:
                print(f"⚠️ CSV-Datei nicht gefunden: {csv_file}")
                print(f"   Gesuchte Pfade: {possible_paths[:3]}")
                return self._get_fallback_assets()
            
        try:
            df = pd.read_csv(csv_path)
            
            # Verschiedene mögliche Spaltennamen prüfen
            possible_columns = ['asset', 'symbol', 'coin', 'ticker', 'name', 'Symbol', 'Name']
            asset_column = None
            
            for col in possible_columns:
                if col in df.columns:
                    asset_column = col
                    break
            
            if asset_column:
                assets = df[asset_column].str.lower().str.strip().tolist()
                
                # Entferne -USD Endungen und andere Suffixe
                processed_assets = []
                for asset in assets:
                    if asset and isinstance(asset, str) and len(asset) > 0:
                        # Entferne -USD, numbers und andere Suffixe
                        clean_asset = asset.replace('-usd', '').replace('28321-', '').replace('22974-', '').replace('20947-', '')
                        # Entferne trailing numbers
                        import re
                        clean_asset = re.sub(r'\d+$', '', clean_asset)
                        if clean_asset and len(clean_asset) > 1:
                            processed_assets.append(clean_asset)
                
                assets = processed_assets
                # Zeige nur den Dateinamen, nicht den vollständigen Pfad
                csv_display = os.path.basename(csv_path) if csv_path else csv_file
                print(f"✅ {len(assets)} Assets aus CSV geladen: {csv_display}")
                print(f"📋 Geladene Assets: {assets}")
            else:
                print(f"⚠️ Keine Asset-Spalte in CSV gefunden: {list(df.columns)}")
                return self._get_fallback_assets()
                
        except Exception as e:
            print(f"❌ Fehler beim Laden der CSV: {e}")
            return self._get_fallback_assets()
        
        return assets if assets else self._get_fallback_assets()
    
    def _get_fallback_assets(self) -> List[str]:
        """Fallback-Liste mit Major Assets"""
        fallback_assets = [
            'btc', 'eth', 'bnb', 'sol', 'ada', 'avax', 'link', 'pol', 
            'ltc', 'hbar', 'xlm', 'xrp', 'trx', 'leo', 'sui',
            'atom', 'algo', 'tao', 'hype', 'vet'
        ]
        print(f"⚠️ Verwende Fallback-Liste mit {len(fallback_assets)} Assets")
        return fallback_assets
    
    def load_asset_data(self, max_assets: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Lädt Preisdaten für die Assets
        
        Args:
            max_assets: Maximum Anzahl Assets (8-20)
            
        Returns:
            Dictionary mit Asset-Daten
        """
        print(f"\n📈 Lade Asset-Daten (max {max_assets} Assets)...")
        
        loaded_count = 0
        for asset in self.major_assets:
            if loaded_count >= max_assets:
                break
                
            # Suche Asset-Datei
            asset_file = os.path.join(self.data_folder, f"{asset}_1d.csv")
            
            if os.path.exists(asset_file):
                try:
                    df = pd.read_csv(asset_file)
                    
                    # Konvertiere Zeit und sortiere
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.sort_values('time').reset_index(drop=True)
                    
                    # Prüfe ob notwendige Spalten vorhanden sind
                    required_columns = ['close', 'high', 'low']
                    if all(col in df.columns for col in required_columns):
                        self.assets_data[asset] = df
                        loaded_count += 1
                        print(f"✅ {asset.upper()}: {len(df)} Datenpunkte geladen")
                    else:
                        print(f"⚠️ {asset.upper()}: Fehlende Spalten {required_columns}")
                        
                except Exception as e:
                    print(f"❌ {asset.upper()}: Fehler beim Laden - {e}")
            else:
                print(f"⚠️ {asset.upper()}: Datei nicht gefunden")
        
        print(f"\n✅ {len(self.assets_data)} Assets erfolgreich geladen")
        return self.assets_data
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Berechnet umfassende Performance-Metriken
        
        Args:
            returns: Serie mit Strategy Returns
            
        Returns:
            Dictionary mit Performance-Metriken
        """
        if len(returns) == 0 or returns.isna().all():
            return {
                'total_return': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0, 'omega_ratio': 0,
                'calmar_ratio': 0, 'max_drawdown': 0, 'win_rate': 0, 'profit_factor': 0,
                'volatility': 0, 'num_trades': 0
            }
        
        # Bereinige Returns
        returns = returns.dropna()
        
        # Basic metrics - KORRIGIERTE TOTAL RETURN BERECHNUNG
        # Verwende kumulatives Produkt für realistische Backtesting-Returns
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        
        # Cap extreme total returns bei unrealistischen Werten (> 300%)
        if total_return > 3:  # > 300% - deutlich restriktiver
            # Alternative: Verwende geometrischen Mittelwert für stabilere Ergebnisse
            geometric_mean = ((1 + returns).prod() ** (1/len(returns))) - 1
            total_return = ((1 + geometric_mean) ** 252) - 1  # Annualisiert
            
            # Zusätzliche Sicherheit: Cap bei 200% maximum
            if total_return > 2:  # > 200%
                total_return = min(total_return, 2.0)  # Maximum 200%
            
        volatility = returns.std() * np.sqrt(252)  # Annualisiert
        mean_return = returns.mean() * 252  # Annualisiert
        
        # Sharpe Ratio (risk-free rate = 0) - FUNKTIONIERT KORREKT
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation) - KORRIGIERT nach Sharpe-Muster
        # Verwende nur negative Returns für Downside-Volatilität (wie Sharpe, aber nur Downside)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_volatility = downside_returns.std() * np.sqrt(252)  # Annualisiert wie bei Sharpe
            sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else 0
        else:
            # Keine negativen Returns -> perfekte Performance, aber konservativ bewerten
            sortino_ratio = sharpe_ratio * 1.5 if sharpe_ratio > 0 else 0
        
        # Einfacher Extremwert-Schutz wie bei anderen Metriken (konsistent)
        if abs(sortino_ratio) > 5:
            sortino_ratio = 5.0 if sortino_ratio > 0 else -5.0
        
        # Omega Ratio (gains vs losses)
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        omega_ratio = gains / losses if losses > 0 else np.inf if gains > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar Ratio
        calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win Rate (nur für tatsächliche Trades, nicht für 0-Returns von Cash-Positionen)
        active_returns = returns[returns != 0]  # Exclude Cash periods
        winning_trades = (active_returns > 0).sum()
        total_trades = len(active_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'omega_ratio': omega_ratio if omega_ratio != np.inf else 10,  # Cap extreme values
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != np.inf else 10,  # Cap extreme values
            'volatility': volatility,
            'num_trades': total_trades
        }
    
    def find_top_configurations(self, results_df: pd.DataFrame, 
                               metric: str = 'sharpe_ratio', 
                               top_n: int = 5) -> pd.DataFrame:
        """
        Findet die besten Konfigurationen basierend auf einer Metrik
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            metric: Metrik für Ranking
            top_n: Anzahl Top-Konfigurationen
            
        Returns:
            DataFrame mit Top-Konfigurationen
        """
        if results_df.empty or metric not in results_df.columns:
            return pd.DataFrame()
        
        # Sortiere nach Metrik (absteigend)
        top_configs = results_df.nlargest(top_n, metric).copy()
        
        # Füge Ranking hinzu
        top_configs['rank'] = range(1, len(top_configs) + 1)
        
        return top_configs
    
    def create_performance_heatmap(self, results_df: pd.DataFrame, 
                                  metric: str = 'sharpe_ratio',
                                  length_column: str = 'length',
                                  figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Erstellt Heatmap der Performance (Y=Length, X=Asset)
        Jedes Asset hat seine eigene individuelle Farbskalierung
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            metric: Metrik für Heatmap
            length_column: Name der Spalte mit den Längen (z.B. 'ema_length', 'cci_length')
            figsize: Größe der Figur
        """
        if results_df.empty:
            print("❌ Keine Daten für Heatmap")
            return
        
        # Pivot-Tabelle erstellen
        pivot_data = results_df.pivot(index=length_column, columns='asset', values=metric)
        
        # Normalisiere jede Asset-Spalte individuell (0-1 Skalierung pro Asset)
        pivot_normalized = pivot_data.copy()
        for asset in pivot_data.columns:
            asset_data = pivot_data[asset].dropna()
            if len(asset_data) > 0:
                min_val = asset_data.min()
                max_val = asset_data.max()
                if max_val > min_val:
                    pivot_normalized[asset] = (asset_data - min_val) / (max_val - min_val)
                else:
                    pivot_normalized[asset] = 0.5  # Mittlerer Wert wenn alle gleich
        
        # Heatmap erstellen
        plt.figure(figsize=figsize)
        
        # Farbschema basierend auf Metrik
        cmap = 'RdYlGn' if 'ratio' in metric.lower() or 'return' in metric.lower() else 'viridis'
        
        # Y-Achsen-Ticks in 5er-Schritten für bessere Übersicht (5, 10, 15, 20, ...)
        # Finde Positionen die durch 5 teilbar sind (für Längen 5, 10, 15, etc.)
        y_tick_positions = []
        y_tick_labels = []
        for i, length_val in enumerate(pivot_normalized.index):
            if length_val % 5 == 0:  # Nur Längen die durch 5 teilbar sind (5, 10, 15, ...)
                y_tick_positions.append(i)
                y_tick_labels.append(length_val)
        
        # Verwende normalisierte Daten für individuelle Asset-Skalierung
        sns.heatmap(
            pivot_normalized,
            annot=False,
            cmap=cmap,
            vmin=0,  # Normalisierte Werte von 0 bis 1
            vmax=1,
            cbar_kws={'label': f'{metric.replace("_", " ").title()} (Individual Asset Scaling)'},
            xticklabels=True,
            yticklabels=y_tick_labels if len(y_tick_labels) < 25 else False
        )
        
        # Setze Y-Ticks manuell für 10er-Schritte
        plt.yticks(y_tick_positions, y_tick_labels)
        
        plt.title(f'{self.strategy_name} Strategy Performance Heatmap (Individual Scaling)\n{metric.replace("_", " ").title()} by Asset and {self.strategy_name} Length', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Asset', fontsize=12, fontweight='bold')
        plt.ylabel(f'{self.strategy_name} Length', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Speichere Heatmap
        heatmap_path = os.path.join(self.results_folder, f'heatmap_{metric}.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Heatmap gespeichert: {heatmap_path}")
    
    def create_average_heatmap(self, results_df: pd.DataFrame, 
                              length_column: str,
                              metric: str = 'sortino_ratio',
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Erstellt Heatmap der durchschnittlichen Performance pro Länge
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            length_column: Name der Spalte mit den Indikator-Längen
            metric: Metrik für durchschnittliche Berechnung
            length_column: Name der Spalte mit den Längen
            figsize: Größe der Figur
        """
        if results_df.empty:
            print("❌ Keine Daten für Average Heatmap")
            return
        
        # Gruppiere nach Länge und berechne Durchschnittswerte
        avg_performance = results_df.groupby(length_column).agg({
            metric: ['mean', 'std', 'min', 'max', 'count']
        }).round(3)
        
        # Flatten column names
        avg_performance.columns = [f'{metric}_{stat}' for stat in ['mean', 'std', 'min', 'max', 'count']]
        
        # Erstelle eine Matrix für die Heatmap (Length vs. verschiedene Statistiken)
        heatmap_data = avg_performance[[f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max']].T
        
        # Erstelle Heatmap
        plt.figure(figsize=figsize)
        
        # Farbschema
        cmap = 'RdYlGn' if 'ratio' in metric.lower() or 'return' in metric.lower() else 'viridis'
        
        # X-Achsen-Ticks in 5er-Schritten für Längen (5, 10, 15, 20, ...)
        lengths = list(heatmap_data.columns)
        x_tick_positions = []
        x_tick_labels = []
        for i, length_val in enumerate(lengths):
            if length_val % 5 == 0:  # Nur Längen die durch 5 teilbar sind
                x_tick_positions.append(i)
                x_tick_labels.append(str(int(length_val)))
        
        sns.heatmap(
            heatmap_data,
            annot=False,
            cmap=cmap,
            center=0 if 'ratio' in metric.lower() else None,
            cbar_kws={'label': f'Average {metric.replace("_", " ").title()}'},
            xticklabels=False,  # Deaktiviere Standard-Labels
            yticklabels=['Mean', 'Std Dev', 'Min', 'Max']
        )
        
        # Setze X-Ticks manuell für 5er-Schritte
        plt.xticks(x_tick_positions, x_tick_labels, rotation=0)
        
        plt.title(f'{self.strategy_name} Length vs Average {metric.replace("_", " ").title()}\nStatistical Overview Across All Assets', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'{self.strategy_name} Length', fontsize=12, fontweight='bold')
        plt.ylabel('Statistical Measures', fontsize=12, fontweight='bold')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Speichere Heatmap mit korrektem Dateinamen-Format
        heatmap_path = os.path.join(self.results_folder, f'heatmap_{self.strategy_name.lower()}_avg_{metric}.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 {self.strategy_name}-Average Heatmap gespeichert: {heatmap_path}")
        print(f"🔧 DEBUG: strategy_name='{self.strategy_name}', metric='{metric}', length_column='{length_column}'")
        
        # 2. ERSTELLE AUCH LINIEN-GRAFIK (wie in den ursprünglichen Systemen)
        self.create_average_line_plot(results_df, length_column, metric)
    
    def create_average_line_plot(self, results_df: pd.DataFrame, 
                                length_column: str,
                                metric: str = 'sortino_ratio',
                                figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Erstellt Linien-Grafik der durchschnittlichen Performance pro Länge mit Standard Deviation Bands
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            length_column: Name der Spalte mit den Indikator-Längen
            metric: Metrik für durchschnittliche Berechnung
            figsize: Größe der Grafik
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Berechne Statistiken pro Länge
        stats_by_length = results_df.groupby(length_column)[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Filtere nur Längen mit genügend Datenpunkten
        stats_by_length = stats_by_length[stats_by_length['count'] >= 3]
        
        if len(stats_by_length) == 0:
            print(f"⚠️ Nicht genügend Daten für Average Line Plot: {metric}")
            return
            
        # Erstelle Figure
        plt.figure(figsize=figsize)
        
        lengths = stats_by_length[length_column].values
        means = stats_by_length['mean'].values
        stds = stats_by_length['std'].values
        
        # Hauptlinie (Average)
        plt.plot(lengths, means, 'b-', linewidth=2.5, label=f'Average {metric.replace("_", " ").title()}', alpha=0.8)
        
        # Standard Deviation Bands
        plt.fill_between(lengths, means - stds, means + stds, 
                        alpha=0.3, color='blue', label='± 1 Standard Deviation')
        
        # X-Achsen-Ticks in 20er-Schritten für Längen (wie in ursprünglichen Systemen)
        x_min, x_max = lengths.min(), lengths.max()
        x_ticks = []
        for x in range(int(x_min), int(x_max) + 1, 20):
            if x >= x_min and x <= x_max:
                x_ticks.append(x)
        
        if len(x_ticks) > 0:
            plt.xticks(x_ticks)
        
        # Styling
        plt.title(f'{self.strategy_name} Length vs Average {metric.replace("_", " ").title()}\nWith Standard Deviation Bands', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'{self.strategy_name} Length', fontsize=12, fontweight='bold')
        plt.ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        
        # Optimiere Layout
        plt.tight_layout()
        
        # Speichere Linien-Grafik mit korrektem Dateinamen-Format
        line_path = os.path.join(self.results_folder, f'line_{self.strategy_name.lower()}_avg_{metric}.png')
        plt.savefig(line_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 {self.strategy_name}-Average Line Plot gespeichert: {line_path}")

    def generate_comprehensive_report(self, results_df: pd.DataFrame, 
                                     length_column: str = 'length',
                                     strategy_description: str = "") -> None:
        """
        Generiert umfassenden Bericht mit allen Analysen
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            length_column: Name der Spalte mit den Längen
            strategy_description: Beschreibung der Strategie
        """
        if results_df.empty:
            print("❌ Keine Ergebnisse für Bericht")
            return
        
        print(f"\n📋 Generiere umfassenden {self.strategy_name} Bericht...")
        
        # Metriken für Analyse
        metrics = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio', 'calmar_ratio', 'total_return']
        
        # 1. Erstelle Heatmaps für alle wichtigen Metriken
        for metric in metrics:
            if metric in results_df.columns:
                print(f"  📊 Erstelle Heatmap für {metric}...")
                self.create_performance_heatmap(results_df, metric, length_column)
        
        # 1.5. Erstelle Average Heatmaps für wichtige Metriken
        avg_metrics = ['sortino_ratio', 'sharpe_ratio', 'total_return']
        for metric in avg_metrics:
            if metric in results_df.columns:
                print(f"  📈 Erstelle Average Heatmap für {metric}...")
                self.create_average_heatmap(results_df, length_column, metric)
        
        # 2. Erstelle Textbericht
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"📊 {self.strategy_name} BACKTESTING SYSTEM - UMFASSENDER BERICHT")
        report_lines.append("=" * 80)
        report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📈 Assets getestet: {results_df['asset'].nunique()}")
        report_lines.append(f"📏 {self.strategy_name}-Längen getestet: {results_df[length_column].nunique()}")
        report_lines.append(f"🔢 Gesamt-Kombinationen: {len(results_df)}")
        if strategy_description:
            report_lines.append(f"📋 Strategie: {strategy_description}")
        report_lines.append("")
        
        for metric in metrics:
            if metric in results_df.columns:
                report_lines.append(f"\n🏆 TOP 5 KONFIGURATIONEN - {metric.replace('_', ' ').title()}")
                report_lines.append("-" * 60)
                
                top_configs = self.find_top_configurations(results_df, metric, 5)
                
                for _, config in top_configs.iterrows():
                    length_val = config[length_column] if length_column in config else "N/A"
                    report_lines.append(
                        f"#{config['rank']:1d}. {config['asset'].upper():<6} | "
                        f"{self.strategy_name}-{length_val:2.0f} | "
                        f"{metric.replace('_', ' ').title()}: {config[metric]:8.3f} | "
                        f"Return: {config['total_return']:8.1%} | "
                        f"Trades: {config['num_trades']:4.0f}"
                    )
                report_lines.append("")
        
        # 3. Asset-spezifische Analyse
        report_lines.append(f"\n📊 ASSET-SPEZIFISCHE ANALYSE (Beste {self.strategy_name} pro Asset)")
        report_lines.append("-" * 60)
        
        for asset in results_df['asset'].unique():
            asset_results = results_df[results_df['asset'] == asset]
            best_config = asset_results.loc[asset_results['sharpe_ratio'].idxmax()]
            
            length_val = best_config[length_column] if length_column in best_config else "N/A"
            report_lines.append(
                f"{asset.upper():<6}: Beste {self.strategy_name}-{length_val:2.0f} | "
                f"Sharpe: {best_config['sharpe_ratio']:6.2f} | "
                f"Return: {best_config['total_return']:7.1%} | "
                f"DD: {best_config['max_drawdown']:6.1%}"
            )
        
        # 4. Statistische Zusammenfassung
        report_lines.append(f"\n📈 STATISTISCHE ZUSAMMENFASSUNG")
        report_lines.append("-" * 60)
        
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']:
            if metric in results_df.columns:
                stats = results_df[metric].describe()
                report_lines.append(
                    f"{metric.replace('_', ' ').title():<15}: "
                    f"Mean={stats['mean']:7.3f} | "
                    f"Std={stats['std']:7.3f} | "
                    f"Max={stats['max']:7.3f} | "
                    f"Min={stats['min']:7.3f}"
                )
        
        # 5. Beste Gesamt-Performance
        report_lines.append(f"\n💡 {self.strategy_name} EMPFEHLUNGEN")
        report_lines.append("-" * 60)
        
        # Beste Gesamt-Performance
        best_overall = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        length_val = best_overall[length_column] if length_column in best_overall else "N/A"
        report_lines.append(
            f"🥇 Beste Gesamt-Performance: {best_overall['asset'].upper()} mit {self.strategy_name}-{length_val:.0f} "
            f"(Sharpe: {best_overall['sharpe_ratio']:.2f})"
        )
        
        # Konsistenteste Strategie (niedrigste Volatilität bei positivem Sharpe)
        consistent = results_df[results_df['sharpe_ratio'] > 0]
        if not consistent.empty:
            most_consistent = consistent.loc[consistent['volatility'].idxmin()]
            length_val = most_consistent[length_column] if length_column in most_consistent else "N/A"
            report_lines.append(
                f"🎯 Konsistenteste Strategie: {most_consistent['asset'].upper()} mit {self.strategy_name}-{length_val:.0f} "
                f"(Volatilität: {most_consistent['volatility']:.1%})"
            )
        
        # Höchste Returns (bei akzeptabler Volatilität)
        high_return = results_df[results_df['volatility'] < results_df['volatility'].quantile(0.75)]
        if not high_return.empty:
            best_return = high_return.loc[high_return['total_return'].idxmax()]
            length_val = best_return[length_column] if length_column in best_return else "N/A"
            report_lines.append(
                f"💰 Höchste Returns: {best_return['asset'].upper()} mit {self.strategy_name}-{length_val:.0f} "
                f"(Return: {best_return['total_return']:.1%})"
            )
        
        # Speichere Bericht
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.results_folder, f'comprehensive_{self.strategy_name.lower()}_backtest_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Speichere auch CSV mit allen Ergebnissen
        csv_path = os.path.join(self.results_folder, f'all_{self.strategy_name.lower()}_backtest_results.csv')
        results_df.to_csv(csv_path, index=False)
        
        print(f"📄 {self.strategy_name} Bericht gespeichert: {report_path}")
        print(f"📊 CSV-Daten gespeichert: {csv_path}")
        
        # Zeige Zusammenfassung
        print(f"\n" + "=" * 60)
        print(f"✅ {self.strategy_name} BACKTESTING ABGESCHLOSSEN")
        print(f"=" * 60)
        print(f"📊 {len(results_df)} Konfigurationen getestet")
        print(f"🥇 Beste Sharpe Ratio: {results_df['sharpe_ratio'].max():.3f}")
        print(f"💰 Höchster Return: {results_df['total_return'].max():.1%}")
        print(f"📁 Alle Ergebnisse in: {self.results_folder}/")
    
    # =====================================
    # GEMEINSAME FUNKTIONEN FÜR ALLE SYSTEME
    # =====================================
    
    def find_top_configurations(self, results_df: pd.DataFrame, 
                               metric: str = 'sharpe_ratio', 
                               top_n: int = 5) -> pd.DataFrame:
        """
        Findet die besten Konfigurationen basierend auf einer Metrik
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            metric: Metrik für Ranking
            top_n: Anzahl Top-Konfigurationen
            
        Returns:
            DataFrame mit Top-Konfigurationen
        """
        if results_df.empty or metric not in results_df.columns:
            return pd.DataFrame()
        
        # Sortiere nach Metrik (absteigend)
        top_configs = results_df.nlargest(top_n, metric).copy()
        
        # Füge Ranking hinzu
        top_configs['rank'] = range(1, len(top_configs) + 1)
        
        return top_configs
    
    def find_best_average_calibration(self, results_df: pd.DataFrame, 
                                    length_column: str) -> Dict[str, float]:
        """
        Findet die durchschnittlich beste Kalibrierung über alle Assets hinweg
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            length_column: Name der Spalte mit den Indikator-Längen (z.B. 'rsi_length', 'ema_length')
            
        Returns:
            Dictionary mit durchschnittlich besten Längen für verschiedene Metriken
        """
        if results_df.empty or length_column not in results_df.columns:
            return {}
        
        indicator_name = length_column.replace('_length', '')
        
        # Gruppiere nach Länge und berechne Durchschnittswerte
        avg_performance = results_df.groupby(length_column).agg({
            'sharpe_ratio': 'mean',
            'sortino_ratio': 'mean', 
            'omega_ratio': 'mean',
            'calmar_ratio': 'mean',
            'total_return': 'mean',
            'win_rate': 'mean',
            'max_drawdown': 'mean',
            'volatility': 'mean',
            'profit_factor': 'mean'
        }).round(3)
        
        # Finde beste durchschnittliche Längen für jede Metrik
        best_configs = {}
        
        # Positive Metriken (höher = besser)
        positive_metrics = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio', 'calmar_ratio', 
                           'total_return', 'win_rate', 'profit_factor']
        
        for metric in positive_metrics:
            best_length = avg_performance[metric].idxmax()
            best_value = avg_performance[metric].max()
            best_configs[f'best_{indicator_name}_{metric}'] = best_length
            best_configs[f'avg_{metric}'] = best_value
        
        # Negative Metriken (niedriger = besser)
        negative_metrics = ['max_drawdown', 'volatility']
        
        for metric in negative_metrics:
            best_length = avg_performance[metric].idxmin()
            best_value = avg_performance[metric].min()
            best_configs[f'best_{indicator_name}_{metric}'] = best_length
            best_configs[f'avg_{metric}'] = best_value
        
        # Kombinierter Score
        scored_avg = avg_performance.copy()
        
        # Normalisierung für kombinierten Score
        for metric in positive_metrics:
            if metric in scored_avg.columns:
                min_val = scored_avg[metric].min()
                max_val = scored_avg[metric].max()
                if max_val > min_val:
                    scored_avg[f'{metric}_norm'] = (scored_avg[metric] - min_val) / (max_val - min_val)
                else:
                    scored_avg[f'{metric}_norm'] = 0.5
        
        for metric in negative_metrics:
            if metric in scored_avg.columns:
                min_val = scored_avg[metric].min()
                max_val = scored_avg[metric].max()
                if max_val > min_val:
                    scored_avg[f'{metric}_norm'] = 1 - (scored_avg[metric] - min_val) / (max_val - min_val)
                else:
                    scored_avg[f'{metric}_norm'] = 0.5
        
        # Gewichteter kombinierter Score
        weights = {
            'sharpe_ratio_norm': 0.30,
            'sortino_ratio_norm': 0.25,
            'total_return_norm': 0.20,
            'omega_ratio_norm': 0.10,
            'win_rate_norm': 0.05,
            'max_drawdown_norm': 0.05,
            'volatility_norm': 0.05
        }
        
        scored_avg['combined_score'] = 0
        for metric, weight in weights.items():
            if metric in scored_avg.columns:
                scored_avg['combined_score'] += scored_avg[metric] * weight
        
        # Beste kombinierte Konfiguration
        best_combined_length = scored_avg['combined_score'].idxmax()
        best_combined_score = scored_avg['combined_score'].max()
        
        best_configs[f'best_{indicator_name}_combined'] = best_combined_length
        best_configs['avg_combined_score'] = best_combined_score
        
        # Füge Performance-Daten für beste kombinierte Konfiguration hinzu
        best_row = avg_performance.loc[best_combined_length]
        for metric in avg_performance.columns:
            best_configs[f'combined_{indicator_name}_{metric}'] = best_row[metric]
        
        # Erstelle Average Heatmaps und Line Plots (wie in den ursprünglichen Systemen)
        print(f"\n📊 Erstelle Average Analysen für {indicator_name}...")
        avg_metrics = ['sharpe_ratio', 'sortino_ratio', 'total_return']
        for metric in avg_metrics:
            if metric in results_df.columns:
                print(f"  📈 Erstelle Average Heatmap für {metric}...")
                self.create_average_heatmap(results_df, length_column, metric)
        
        return best_configs
    
    def generate_specific_analysis(self, results_df: pd.DataFrame, 
                                 length_column: str,
                                 strategy_description: str,
                                 threshold: Optional[float] = None) -> None:
        """
        Generiert indikator-spezifische Analyse
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            length_column: Name der Spalte mit den Indikator-Längen
            strategy_description: Beschreibung der Strategie
            threshold: Optional - Schwellenwert für Signale
        """
        if results_df.empty:
            print("❌ Keine Daten für spezifische Analyse")
            return
        
        indicator_name = length_column.replace('_length', '').upper()
        
        print(f"\n📋 Erstelle {indicator_name}-spezifische Analyse...")
        
        # Finde beste Kalibrierungen
        best_calibration = self.find_best_average_calibration(results_df, length_column)
        
        if best_calibration:
            # Erstelle spezifischen Bericht
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"🎯 {indicator_name}-SPEZIFISCHE ANALYSE")
            report_lines.append("=" * 80)
            report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Beste kombinierte Länge
            indicator_lower = indicator_name.lower()
            combined_key = f'best_{indicator_lower}_combined'
            if combined_key in best_calibration:
                report_lines.append(
                    f"🥇 Beste Durchschnittliche {indicator_name}-Länge (Kombiniert): "
                    f"{indicator_name}-{best_calibration[combined_key]:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte Länge
                combined_sharpe_key = f'combined_{indicator_lower}_sharpe_ratio'
                combined_return_key = f'combined_{indicator_lower}_total_return'
                combined_dd_key = f'combined_{indicator_lower}_max_drawdown'
                
                if all(key in best_calibration for key in [combined_sharpe_key, combined_return_key, combined_dd_key]):
                    report_lines.append(
                        f"   📊 Avg Sharpe: {best_calibration[combined_sharpe_key]:.3f} | "
                        f"Avg Return: {best_calibration[combined_return_key]:.1%} | "
                        f"Avg DD: {best_calibration[combined_dd_key]:.1%}"
                    )
                    
                    # Sortino und Win Rate
                    combined_sortino_key = f'combined_{indicator_lower}_sortino_ratio'
                    combined_winrate_key = f'combined_{indicator_lower}_win_rate'
                    if combined_sortino_key in best_calibration and combined_winrate_key in best_calibration:
                        report_lines.append(
                            f"   📈 Avg Sortino: {best_calibration[combined_sortino_key]:.3f} | "
                            f"Avg Win Rate: {best_calibration[combined_winrate_key]:.1%} | "
                            f"Score: {best_calibration['avg_combined_score']:.3f}"
                        )
            
            # Top Längen für einzelne Metriken
            report_lines.append("")
            report_lines.append(f"📈 Beste Durchschnitts-{indicator_name}s nach Metriken:")
            
            sharpe_key = f'best_{indicator_lower}_sharpe_ratio'
            return_key = f'best_{indicator_lower}_total_return'
            sortino_key = f'best_{indicator_lower}_sortino_ratio'
            
            if sharpe_key in best_calibration:
                report_lines.append(f"   • Sharpe Ratio: {indicator_name}-{best_calibration[sharpe_key]:.0f} (Ø {best_calibration['avg_sharpe_ratio']:.3f})")
            if return_key in best_calibration:
                report_lines.append(f"   • Total Return: {indicator_name}-{best_calibration[return_key]:.0f} (Ø {best_calibration['avg_total_return']:.1%})")
            if sortino_key in best_calibration:
                report_lines.append(f"   • Sortino Ratio: {indicator_name}-{best_calibration[sortino_key]:.0f} (Ø {best_calibration['avg_sortino_ratio']:.3f})")
            
            # Strategie-spezifische Empfehlungen
            report_lines.append(f"\n💡 {indicator_name} STRATEGIE EMPFEHLUNGEN")
            report_lines.append("-" * 60)
            report_lines.append(f"📋 {indicator_name} STRATEGIE INSIGHTS:")
            report_lines.append(f"   • {strategy_description}")
            
            if threshold is not None:
                report_lines.append(f"   • Schwellenwert: {threshold}")
            
            # Allgemeine Tipps basierend auf Indikatortyp
            if indicator_name in ['RSI', 'CCI', 'WILLR', 'AROON']:
                report_lines.append(f"   • Nur Long-Positionen, keine Short-Positionen")
                report_lines.append(f"   • Momentum-basierte Strategie")
            elif indicator_name == 'EMA':
                report_lines.append(f"   • Trend-Following mit Long/Short Positionen")
                report_lines.append(f"   • Preis-basierte Signalgenerierung")
            
            report_lines.append(f"   • Kürzere {indicator_name}-Perioden: Mehr Trades, höhere Sensitivität")
            report_lines.append(f"   • Längere {indicator_name}-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            report_lines.append(f"\n🎯 {indicator_name} PARAMETER-OPTIMIERUNG")
            report_lines.append("-" * 60)
            report_lines.append(f"📌 OPTIMALE {indicator_name}-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            if combined_key in best_calibration:
                combined_score = best_calibration.get('avg_combined_score', 0.0)
                report_lines.append(f"   🥇 Beste Gesamtperformance: {indicator_name}-{best_calibration[combined_key]:.0f} (Score: {combined_score:.3f})")
            
            if sharpe_key in best_calibration:
                avg_sharpe = best_calibration.get('avg_sharpe_ratio', 0.0)
                report_lines.append(f"   📈 Höchste Sharpe Ratio: {indicator_name}-{best_calibration[sharpe_key]:.0f} ({avg_sharpe:.3f})")
            
            if return_key in best_calibration:
                avg_return = best_calibration.get('avg_total_return', 0.0)
                report_lines.append(f"   💰 Höchste Returns: {indicator_name}-{best_calibration[return_key]:.0f} ({avg_return:.1%})")
            
            dd_key = f'best_{indicator_lower}_max_drawdown'
            if dd_key in best_calibration:
                avg_dd = best_calibration.get('avg_max_drawdown', 0.0)
                report_lines.append(f"   🛡️ Niedrigster Drawdown: {indicator_name}-{best_calibration[dd_key]:.0f} ({avg_dd:.1%})")
            
            report_lines.append(f"\n" + "=" * 68)
            report_lines.append(f"🏁 {indicator_name}-ANALYSE ABGESCHLOSSEN")
            report_lines.append("=" * 68)
            
            # Speichere spezifischen Bericht
            report_text = "\n".join(report_lines)
            report_path = os.path.join(self.results_folder, f'{indicator_name.lower()}_specific_analysis.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📄 {indicator_name}-spezifische Analyse gespeichert: {report_path}")
    
    def print_backtesting_summary(self, indicator_name: str, strategy_description: str, 
                                period_range: range, max_assets: int, threshold: Optional[float] = None):
        """
        Druckt standardisierte Backtesting-Zusammenfassung
        
        Args:
            indicator_name: Name des Indikators
            strategy_description: Beschreibung der Strategie
            period_range: Bereich der getesteten Perioden
            max_assets: Anzahl der Assets
            threshold: Optional - Schwellenwert
        """
        print(f"🚀 {indicator_name.upper()} BACKTESTING SYSTEM START")
        print("=" * 60)
        print(f"⚙️ KONFIGURATION:")
        print(f"   {indicator_name.upper()}-Range: {period_range.start} bis {period_range.stop-1}")
        print(f"   Max Assets: {max_assets} (aus backtesting_majors.csv)")
        print(f"   Strategie: {strategy_description}")
        if threshold is not None:
            print(f"   Schwelle: {threshold}")
    
    def print_quick_preview(self, results_df: pd.DataFrame, length_column: str, 
                          indicator_name: str):
        """
        Druckt Quick-Preview der besten Ergebnisse
        
        Args:
            results_df: DataFrame mit Ergebnissen
            length_column: Name der Längen-Spalte
            indicator_name: Name des Indikators
        """
        # Zeige Quick-Preview der Top-Performer
        print(f"\n🏆 QUICK PREVIEW - TOP 5 SHARPE RATIO:")
        print("-" * 50)
        top_5 = self.find_top_configurations(results_df, 'sharpe_ratio', 5)
        
        for _, config in top_5.iterrows():
            print(f"  {config['rank']:.0f}. {config['asset'].upper()}: "
                  f"{indicator_name.upper()}-{config[length_column]:.0f} | "
                  f"Sharpe: {config['sharpe_ratio']:.3f} | "
                  f"Return: {config['total_return']:.1%}")
        
        # Zeige beste durchschnittliche Kalibrierung
        print(f"\n🎯 BESTE DURCHSCHNITTLICHE {indicator_name.upper()}-KALIBRIERUNG:")
        print("-" * 50)
        best_avg = self.find_best_average_calibration(results_df, length_column)
        
        indicator_lower = indicator_name.lower()
        combined_key = f'best_{indicator_lower}_combined'
        if combined_key in best_avg:
            print(f"  🥇 Optimale {indicator_name.upper()}-Länge: "
                  f"{indicator_name.upper()}-{best_avg[combined_key]:.0f}")
            
            combined_sharpe_key = f'combined_{indicator_lower}_sharpe_ratio'
            combined_return_key = f'combined_{indicator_lower}_total_return'
            if combined_sharpe_key in best_avg and combined_return_key in best_avg:
                print(f"  📊 Avg Sharpe: {best_avg[combined_sharpe_key]:.3f} | "
                      f"Avg Return: {best_avg[combined_return_key]:.1%}")
    
    def run_generic_backtests(self, indicator_range: range, length_param_name: str, 
                             calculate_signals_func, indicator_name: str) -> pd.DataFrame:
        """
        Generische Backtest-Funktion für alle Indikatoren
        
        Args:
            indicator_range: Range der Indikator-Perioden zum Testen (z.B. range(5, 151))
            length_param_name: Name des Längen-Parameters im Result (z.B. 'ema_length', 'cci_length')
            calculate_signals_func: Funktion zur Signalberechnung (z.B. self.calculate_ema_signals)
            indicator_name: Name des Indikators für Ausgaben (z.B. 'EMA', 'CCI')
            
        Returns:
            DataFrame mit allen Backtest-Ergebnissen
        """
        self.print_backtesting_summary(indicator_name, self.strategy_description, 
                                     indicator_range, len(self.assets_data), self.threshold)
        
        all_results = []
        total_combinations = len(indicator_range) * len(self.assets_data)
        current_combination = 0
        
        for length in indicator_range:
            for asset_name, asset_data in self.assets_data.items():
                current_combination += 1
                progress = (current_combination / total_combinations) * 100
                
                # Statische Progress-Anzeige (überschreibt vorherige Zeile)
                print(f"\r📊 Teste {indicator_name}-Länge: {length:3d} | Progress: {progress:5.1f}% ({current_combination}/{total_combinations})     ", end='', flush=True)
                
                # Berechne Signale mit der übergebenen Funktion
                signals = calculate_signals_func(asset_data, length)
                
                if signals.empty:
                    continue
                
                # Berechne Performance-Metriken
                metrics = self.calculate_performance_metrics(signals['strategy_returns'])
                
                # Sammle Ergebnisse
                result = {
                    'asset': asset_name,
                    length_param_name: length,
                    **metrics
                }
                
                all_results.append(result)
        
        # Neue Zeile nach Progress-Ausgabe
        print()
        
        # Konvertiere zu DataFrame
        results_df = pd.DataFrame(all_results)
        
        print(f"\n✅ {indicator_name} Backtests abgeschlossen: {len(results_df)} Kombinationen getestet")
        
        return results_df
    
    def generate_specific_analysis_custom(self, results_df: pd.DataFrame, 
                                        length_column: str,
                                        strategy_description: str = "",
                                        threshold: Optional[float] = None,
                                        strategy_insights: List[str] = None,
                                        classical_periods: List[int] = None,
                                        additional_sections: Dict[str, List[str]] = None) -> None:
        """
        Generiert verallgemeinerte spezifische Analyse für alle Indikatoren
        Behält das Layout der Original-Analysen bei
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            length_column: Name der Spalte mit den Längen (z.B. 'ema_length', 'mom_length')
            strategy_description: Beschreibung der Strategie
            threshold: Schwellenwert der Strategie (falls vorhanden)
            strategy_insights: Liste mit strategie-spezifischen Insights
            classical_periods: Liste mit klassischen Perioden zum Analysieren
            additional_sections: Dict mit zusätzlichen Sektionen {title: [lines]}
        """
        if results_df.empty:
            print(f"❌ Keine Daten für {self.strategy_name}-spezifische Analyse")
            return
        
        print(f"\n📋 Erstelle {self.strategy_name}-spezifische Analyse...")
        
        # Finde beste Kalibrierungen
        best_calibration = self.find_best_average_calibration(results_df, length_column)
        
        if best_calibration:
            # Erstelle zusätzlichen Bericht
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"🎯 {self.strategy_name.upper()}-SPEZIFISCHE ANALYSE")
            report_lines.append("=" * 80)
            report_lines.append(f"🕐 Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Beste kombinierte Länge
            best_combined_key = f'best_{self.strategy_name.lower()}_combined'
            if best_combined_key in best_calibration:
                report_lines.append(
                    f"🥇 Beste Durchschnittliche {self.strategy_name}-Länge (Kombiniert): {self.strategy_name}-{best_calibration[best_combined_key]:.0f}"
                )
                
                # Performance-Metriken für beste kombinierte Länge
                combined_keys = [
                    f'combined_{self.strategy_name.lower()}_sharpe_ratio',
                    f'combined_{self.strategy_name.lower()}_total_return',
                    f'combined_{self.strategy_name.lower()}_max_drawdown'
                ]
                if all(key in best_calibration for key in combined_keys):
                    report_lines.append(
                        f"   📊 Avg Sharpe: {best_calibration[combined_keys[0]]:.3f} | "
                        f"Avg Return: {best_calibration[combined_keys[1]]:.1%} | "
                        f"Avg DD: {best_calibration[combined_keys[2]]:.1%}"
                    )
                
                # Sortino und Win Rate
                sortino_key = f'combined_{self.strategy_name.lower()}_sortino_ratio'
                winrate_key = f'combined_{self.strategy_name.lower()}_win_rate'
                if sortino_key in best_calibration and winrate_key in best_calibration:
                    report_lines.append(
                        f"   📈 Avg Sortino: {best_calibration[sortino_key]:.3f} | "
                        f"Avg Win Rate: {best_calibration[winrate_key]:.1%} | "
                        f"Score: {best_calibration['avg_combined_score']:.3f}"
                    )
            
            # Top Indikatoren für einzelne Metriken
            report_lines.append("")
            report_lines.append(f"📈 Beste Durchschnitts-{self.strategy_name}s nach Metriken:")
            
            metric_keys = [
                (f'best_{self.strategy_name.lower()}_sharpe_ratio', 'avg_sharpe_ratio', 'Sharpe Ratio'),
                (f'best_{self.strategy_name.lower()}_total_return', 'avg_total_return', 'Total Return'),
                (f'best_{self.strategy_name.lower()}_sortino_ratio', 'avg_sortino_ratio', 'Sortino Ratio')
            ]
            
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_calibration and avg_key in best_calibration:
                    avg_val = best_calibration[avg_key]
                    avg_str = f"{avg_val:.3f}" if 'ratio' in metric_name.lower() else f"{avg_val:.1%}"
                    report_lines.append(f"   • {metric_name}: {self.strategy_name}-{best_calibration[best_key]:.0f} (Ø {avg_str})")
            
            # Strategie-spezifische Empfehlungen
            report_lines.append(f"\n💡 {self.strategy_name.upper()} STRATEGIE EMPFEHLUNGEN")
            report_lines.append("-" * 60)
            report_lines.append(f"📋 {self.strategy_name.upper()} LONG-ONLY STRATEGIE INSIGHTS:")
            
            # Standard-Insights oder benutzer-definierte
            if strategy_insights:
                for insight in strategy_insights:
                    report_lines.append(f"   • {insight}")
            else:
                # Standard-Insights basierend auf Strategie-Beschreibung
                if threshold is not None:
                    report_lines.append(f"   • Long-Position wenn {self.strategy_name} > {threshold}")
                    report_lines.append(f"   • Cash-Position wenn {self.strategy_name} <= {threshold}")
                else:
                    report_lines.append(f"   • Strategie: {strategy_description}")
                report_lines.append(f"   • Kürzere {self.strategy_name}-Perioden: Mehr Trades, höhere Sensitivität")
                report_lines.append(f"   • Längere {self.strategy_name}-Perioden: Weniger Trades, stabilere Signale")
            
            # Parameter-Optimierung
            report_lines.append(f"\n🎯 {self.strategy_name.upper()} PARAMETER-OPTIMIERUNG")
            report_lines.append("-" * 60)
            report_lines.append(f"📌 OPTIMALE {self.strategy_name.upper()}-LÄNGEN FÜR VERSCHIEDENE ZIELE:")
            
            # Beste Gesamtperformance
            if best_combined_key in best_calibration:
                combined_score = best_calibration.get('avg_combined_score', 0.0)
                report_lines.append(f"   🥇 Beste Gesamtperformance: {self.strategy_name}-{best_calibration[best_combined_key]:.0f} (Score: {combined_score:.3f})")
            
            # Spezifische Metriken
            for best_key, avg_key, metric_name in metric_keys:
                if best_key in best_calibration and avg_key in best_calibration:
                    avg_val = best_calibration[avg_key]
                    avg_str = f"({avg_val:.3f})" if 'ratio' in metric_name.lower() else f"({avg_val:.1%})"
                    report_lines.append(f"   📈 Höchste {metric_name}: {self.strategy_name}-{best_calibration[best_key]:.0f} {avg_str}")
            
            # Niedrigster Drawdown
            dd_key = f'best_{self.strategy_name.lower()}_max_drawdown'
            avg_dd_key = 'avg_max_drawdown'
            if dd_key in best_calibration and avg_dd_key in best_calibration:
                avg_dd = best_calibration[avg_dd_key]
                report_lines.append(f"   🛡️ Niedrigster Drawdown: {self.strategy_name}-{best_calibration[dd_key]:.0f} ({avg_dd:.1%})")
            
            # Klassische Kalibrierungen (falls angegeben)
            if classical_periods:
                report_lines.append(f"\n📚 KLASSISCHE {self.strategy_name.upper()}-KALIBRIERUNGEN:")
                report_lines.append("-" * 60)
                for period in classical_periods:
                    period_data = results_df[results_df[length_column] == period]
                    if not period_data.empty:
                        avg_sharpe = period_data['sharpe_ratio'].mean()
                        avg_return = period_data['total_return'].mean()
                        avg_dd = period_data['max_drawdown'].mean()
                        report_lines.append(
                            f"   • {self.strategy_name}-{period}: Sharpe {avg_sharpe:.3f} | Return {avg_return:.1%} | DD {avg_dd:.1%}"
                        )
            
            # Zusätzliche Sektionen (falls angegeben)
            if additional_sections:
                for section_title, section_lines in additional_sections.items():
                    report_lines.append(f"\n{section_title}")
                    report_lines.append("-" * 60)
                    for line in section_lines:
                        report_lines.append(f"   {line}")
            
            report_lines.append(f"\n" + "=" * 68)
            report_lines.append(f"🏁 {self.strategy_name.upper()}-ANALYSE ABGESCHLOSSEN")
            report_lines.append("=" * 68)
            
            # Speichere spezifischen Bericht
            report_text = "\n".join(report_lines)
            report_path = os.path.join(self.results_folder, f'{self.strategy_name.lower()}_specific_analysis.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📄 {self.strategy_name}-spezifische Analyse gespeichert: {report_path}")
        else:
            print(f"❌ Keine Kalibrierungsdaten für {self.strategy_name} verfügbar")