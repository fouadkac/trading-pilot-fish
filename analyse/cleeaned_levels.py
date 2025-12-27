#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Accumulation + Support/Résistance + Inflexions
---------------------------------------------------
- Lit un CSV au format :
  Time;Open;High;Low;Close;HighHighest10;LowLowest10;DiffPrevHighCurrentLow
- Détecte les zones d'accumulation (range) & bandes S/R (intervalles)
- Étiquette les points d'inflexion (support / résistance / neutre)
- Entraîne un LSTM TensorFlow **jusqu'à la fin des époques (pas d'EarlyStopping)**
- Interface Tkinter professionnelle (dark theme) :
  * Sélecteur CSV, paramètres, logs, progression, courbes loss/accuracy
  * Boutons : Démarrer, Exporter modèle, Ouvrir dossier modèle

Dépendances : pandas, numpy, scikit-learn, matplotlib, tensorflow, tk
"""

import os
import sys
import json
import threading
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ==========================
# Paramètres & utilitaires
# ==========================

@dataclass
class Config:
    window_bars: int = 48              # fenêtre pour estimer le range
    hist_bins: int = 60                # nb de bacs pour l'histogramme (densité)
    band_width_pct: float = 0.15       # largeur relative des bandes autour des pics
    accumulation_max_width_pct: float = 0.35  # (HighMax-LowMin)/prix_médian
    lookahead: int = 6                 # barres pour confirmer une inflexion
    touch_tolerance_pct: float = 0.10  # tolérance de "touch" bande
    seq_len: int = 32                  # longueur des séquences LSTM
    batch_size: int = 64
    epochs: int = 50                   # ↑ époques; pas d'arrêt anticipé
    learning_rate: float = 1e-3
    val_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    model_dir: str = "models"

# ==========================
# Chargement & Features
# ==========================

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.sort_values('Time', inplace=True)
        df.reset_index(drop=True, inplace=True)
    for c in ['Open','High','Low','Close','HighHighest10','LowLowest10','DiffPrevHighCurrentLow']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    return df


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['HL'] = d['High'] - d['Low']
    d['OC'] = d['Close'] - d['Open']
    d['RET'] = d['Close'].pct_change().fillna(0.0)
    d['MA_10'] = d['Close'].rolling(10).mean()
    d['MA_20'] = d['Close'].rolling(20).mean()
    d['STD_20'] = d['Close'].rolling(20).std()
    d['ATR_14'] = (d['High'] - d['Low']).rolling(14).mean()
    d.fillna(method='bfill', inplace=True)
    d.fillna(method='ffill', inplace=True)
    return d

# ===================================
# Estimation Support / Résistance
# ===================================

def _density_peaks(prices: np.ndarray, bins: int = 60) -> Tuple[Optional[float], Optional[float]]:
    if len(prices) < 10:
        return None, None
    hist, edges = np.histogram(prices, bins=bins)
    top_idx = np.argsort(hist)[-2:]
    if len(top_idx) < 2:
        return None, None
    centers = (edges[:-1] + edges[1:]) / 2.0
    p1, p2 = centers[top_idx[0]], centers[top_idx[1]]
    return (min(p1, p2), max(p1, p2))


def estimate_bands_for_window(sub: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    prices = sub['Close'].values
    lo_peak, hi_peak = _density_peaks(prices, bins=cfg.hist_bins)
    if lo_peak is None or hi_peak is None:
        return {}
    median_price = np.median(prices)
    bw = cfg.band_width_pct * median_price
    return {
        'support_low': lo_peak - bw,
        'support_high': lo_peak + bw,
        'resistance_low': hi_peak - bw,
        'resistance_high': hi_peak + bw,
        'median_price': median_price,
    }


def mark_accumulation_and_bands(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()
    for col in ['support_low','support_high','resistance_low','resistance_high','median_price']:
        d[col] = np.nan
    d['is_accum'] = False

    for i in range(cfg.window_bars, len(d)):
        sub = d.iloc[i-cfg.window_bars:i]
        hi = sub['High'].max()
        lo = sub['Low'].min()
        med = sub['Close'].median()
        width_rel = (hi - lo) / max(med, 1e-9)
        if width_rel <= cfg.accumulation_max_width_pct:
            bands = estimate_bands_for_window(sub, cfg)
            if bands:
                d.loc[d.index[i], 'is_accum'] = True
                for k,v in bands.items():
                    d.loc[d.index[i], k] = v

    d[['support_low','support_high','resistance_low','resistance_high','median_price']] = \
        d[['support_low','support_high','resistance_low','resistance_high','median_price']].fillna(method='ffill')
    d['is_accum'] = d['is_accum'].astype(bool)
    return d

# ===================================
# Étiquetage des inflexions
# ===================================

def label_inflections(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()
    d['label'] = 0  # 0 = neutre, 1 = inflexion support, 2 = inflexion résistance
    for i in range(len(d) - cfg.lookahead - 1):
        price = d.at[i, 'Close']
        sl, sh = d.at[i, 'support_low'], d.at[i, 'support_high']
        rl, rh = d.at[i, 'resistance_low'], d.at[i, 'resistance_high']
        if np.isnan(sl) or np.isnan(sh) or np.isnan(rl) or np.isnan(rh):
            continue
        tol = cfg.touch_tolerance_pct * d.at[i, 'median_price']
        touch_support = (price >= (sl - tol)) and (price <= (sh + tol))
        touch_res = (price >= (rl - tol)) and (price <= (rh + tol))
        if touch_support:
            future_max = d['High'].iloc[i+1:i+1+cfg.lookahead].max()
            atr = d['ATR_14'].iloc[i]
            if future_max >= price + max(atr, 1e-9):
                d.at[i, 'label'] = 1
        elif touch_res:
            future_min = d['Low'].iloc[i+1:i+1+cfg.lookahead].min()
            atr = d['ATR_14'].iloc[i]
            if future_min <= price - max(atr, 1e-9):
                d.at[i, 'label'] = 2
    return d

# ===================================
# Préparation Dataset LSTM
# ===================================

def build_sequences(df: pd.DataFrame, cfg: Config):
    features = [
        'Open','High','Low','Close',
        'HighHighest10','LowLowest10','DiffPrevHighCurrentLow',
        'HL','OC','RET','MA_10','MA_20','STD_20','ATR_14',
        'support_low','support_high','resistance_low','resistance_high','median_price','is_accum'
    ]
    Xdf = df[features].copy()
    Xdf['is_accum'] = Xdf['is_accum'].astype(float)
    y = df['label'].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xdf.values)

    seqs, ys = [], []
    for i in range(cfg.seq_len, len(X_scaled)):
        seqs.append(X_scaled[i-cfg.seq_len:i, :])
        ys.append(y[i])
    X = np.array(seqs, dtype=np.float32)
    Y = tf.keras.utils.to_categorical(np.array(ys, dtype=np.int32), num_classes=3)
    return X, Y, scaler, features

# ===================================
# Modèle LSTM
# ===================================

def build_model(input_shape: Tuple[int,int], cfg: Config) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===================================
# Callbacks (GUI + Checkpoint)
# ===================================

class GuiTrainingCallback(callbacks.Callback):
    def __init__(self, gui_ref, total_epochs: int):
        super().__init__()
        self.gui = gui_ref
        self.total_epochs = total_epochs
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.gui.update_progress(epoch+1, self.total_epochs, logs)

# ===================================
# Pipeline d'entraînement (thread)
# ===================================

def train_pipeline(csv_path: str, cfg: Config, gui_ref=None):
    try:
        df = load_csv(csv_path)
        df = add_basic_indicators(df)
        df = mark_accumulation_and_bands(df, cfg)
        df = label_inflections(df, cfg)

        # Retirer lignes sans bandes
        df = df.dropna(subset=['support_low','support_high','resistance_low','resistance_high']).reset_index(drop=True)
        if len(df) < cfg.seq_len + 100:
            raise ValueError("Pas assez de données après prétraitement pour créer des séquences.")

        X, Y, scaler, feat_names = build_sequences(df, cfg)

        # Split train/val/test (no shuffle pour préserver la temporalité)
        X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=cfg.test_split, shuffle=False)
        val_size_rel = cfg.val_split / (1.0 - cfg.test_split)
        X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=val_size_rel, shuffle=False)

        model = build_model(input_shape=(X.shape[1], X.shape[2]), cfg=cfg)

        os.makedirs(cfg.model_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.model_dir, 'best_model.keras')

        # IMPORTANT : pas d'EarlyStopping -> on va **au bout** des époques
        cbs = [
            GuiTrainingCallback(gui_ref, cfg.epochs) if gui_ref else callbacks.Callback(),
            callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max')
        ]

        hist = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=cbs,
            verbose=0
        )

        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        report = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'features': feat_names,
        }
        with open(os.path.join(cfg.model_dir, 'scaler.json'), 'w') as f:
            json.dump(report, f, indent=2)

        if gui_ref:
            gui_ref.training_done(hist.history, report)

    except Exception as e:
        if gui_ref:
            gui_ref.log(f"Erreur: {e}")
            gui_ref.enable_controls()
        else:
            raise

# ===================================
# Interface Tkinter (dark, pro)
# ===================================

DARK_BG = "#0f172a"      # slate-900
DARK_PANEL = "#111827"   # gray-900
DARK_ACCENT = "#22d3ee"  # cyan-400
DARK_TEXT = "#e5e7eb"    # gray-200
DARK_MUTE = "#94a3b8"    # slate-400

class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LSTM - Accumulation / S&R / Inflexions")
        self.root.geometry("1100x720")
        self.root.minsize(980, 640)
        self.csv_path = tk.StringVar(value="")
        self.model_saved = tk.StringVar(value="")

        # Config par défaut
        self.cfg = Config()

        self._init_style()
        self._build_layout()
        self._build_left_panel()
        self._build_right_panel()

    # ----- Style -----
    def _init_style(self):
        style = ttk.Style()
        # Forcer un thème supportant le style
        try:
            style.theme_use('clam')
        except:
            pass
        style.configure('.', background=DARK_BG, foreground=DARK_TEXT)
        style.configure('TLabel', background=DARK_BG, foreground=DARK_TEXT, font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI Semibold', 14))
        style.configure('Muted.TLabel', foreground=DARK_MUTE)
        style.configure('TButton', font=('Segoe UI', 10), padding=8)
        style.map('TButton', foreground=[('active', DARK_BG)], background=[('active', DARK_ACCENT)])
        style.configure('Card.TLabelframe', background=DARK_PANEL, foreground=DARK_TEXT, padding=10, borderwidth=0)
        style.configure('Card.TLabelframe.Label', background=DARK_PANEL, foreground=DARK_TEXT, font=('Segoe UI Semibold', 11))
        style.configure('TEntry', fieldbackground='#0b1220', foreground=DARK_TEXT)
        style.configure('Horizontal.TProgressbar', troughcolor='#0b1220', background=DARK_ACCENT)

        self.root.configure(bg=DARK_BG)

    # ----- Layout root -----
    def _build_layout(self):
        self.wrapper = ttk.Frame(self.root)
        self.wrapper.pack(fill='both', expand=True)
        self.wrapper.columnconfigure(0, weight=1)
        self.wrapper.columnconfigure(1, weight=2)
        self.wrapper.rowconfigure(0, weight=1)

    # ----- Left panel -----
    def _build_left_panel(self):
        left = ttk.Frame(self.wrapper)
        left.grid(row=0, column=0, sticky='nsew', padx=14, pady=14)
        left.columnconfigure(0, weight=1)

        # Header
        hdr = ttk.Frame(left)
        hdr.grid(row=0, column=0, sticky='ew', pady=(0,10))
        ttk.Label(hdr, text="LSTM - Accumulation / S&R / Inflexions", style='Header.TLabel').pack(anchor='w')
        ttk.Label(hdr, text="Chargez votre CSV et lancez l'entraînement (aucune interruption avant la fin des époques)", style='Muted.TLabel').pack(anchor='w')

        # File card
        file_card = ttk.Labelframe(left, text="Données", style='Card.TLabelframe')
        file_card.grid(row=1, column=0, sticky='ew', pady=8)
        file_card.columnconfigure(1, weight=1)
        ttk.Label(file_card, text="CSV:").grid(row=0, column=0, sticky='w', padx=(2,6))
        self.ent_csv = ttk.Entry(file_card, textvariable=self.csv_path)
        self.ent_csv.grid(row=0, column=1, sticky='ew')
        ttk.Button(file_card, text="Parcourir", command=self.browse_file).grid(row=0, column=2, padx=(8,0))

        # Params card
        params = ttk.Labelframe(left, text="Paramètres", style='Card.TLabelframe')
        params.grid(row=2, column=0, sticky='ew', pady=8)
        grid = [
            ("window_bars", self.cfg.window_bars),
            ("hist_bins", self.cfg.hist_bins),
            ("band_width_pct", self.cfg.band_width_pct),
            ("accumulation_max_width_pct", self.cfg.accumulation_max_width_pct),
            ("lookahead", self.cfg.lookahead),
            ("touch_tolerance_pct", self.cfg.touch_tolerance_pct),
            ("seq_len", self.cfg.seq_len),
            ("batch_size", self.cfg.batch_size),
            ("epochs", self.cfg.epochs),
            ("learning_rate", self.cfg.learning_rate),
        ]
        self.entries = {}
        cols = 2
        for i,(k,v) in enumerate(grid):
            r, c = divmod(i, cols)
            ttk.Label(params, text=k).grid(row=r*2, column=c, sticky='w', padx=4, pady=(2,0))
            e = ttk.Entry(params)
            e.insert(0, str(v))
            e.grid(row=r*2+1, column=c, sticky='ew', padx=4, pady=(0,6))
            params.columnconfigure(c, weight=1)
            self.entries[k] = e

        # Controls card
        ctrl = ttk.Labelframe(left, text="Contrôles", style='Card.TLabelframe')
        ctrl.grid(row=3, column=0, sticky='ew', pady=8)
        self.btn_start = ttk.Button(ctrl, text="▶ Démarrer l'entraînement", command=self.start_training)
        self.btn_start.grid(row=0, column=0, padx=4, pady=4, sticky='w')
        self.btn_export = ttk.Button(ctrl, text="💾 Ouvrir dossier modèle", command=self.open_model_dir)
        self.btn_export.grid(row=0, column=1, padx=4, pady=4, sticky='w')

        self.progress = ttk.Progressbar(ctrl, orient='horizontal', mode='determinate', length=100)
        self.progress.grid(row=1, column=0, columnspan=2, sticky='ew', padx=4, pady=(6,2))
        self.lbl_pct = ttk.Label(ctrl, text="0%", style='Muted.TLabel')
        self.lbl_pct.grid(row=1, column=2, padx=6)

        # Log card
        log_card = ttk.Labelframe(left, text="Journal", style='Card.TLabelframe')
        log_card.grid(row=4, column=0, sticky='nsew', pady=8)
        left.rowconfigure(4, weight=1)
        self.txt = tk.Text(log_card, height=10, bg="#0b1220", fg=DARK_TEXT, insertbackground=DARK_TEXT, relief='flat')
        self.txt.pack(fill='both', expand=True)

    # ----- Right panel (plots) -----
    def _build_right_panel(self):
        right = ttk.Frame(self.wrapper)
        right.grid(row=0, column=1, sticky='nsew', padx=(0,14), pady=14)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        card = ttk.Labelframe(right, text="Courbes d'entraînement", style='Card.TLabelframe')
        card.grid(row=1, column=0, sticky='nsew')

        self.fig = plt.Figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0b1220')
        self.fig.patch.set_facecolor(DARK_PANEL)
        self.ax.tick_params(colors=DARK_MUTE)
        self.ax.spines['bottom'].set_color(DARK_MUTE)
        self.ax.spines['left'].set_color(DARK_MUTE)
        self.ax.set_title('Loss & Accuracy', color=DARK_TEXT)
        self.ax.set_xlabel('Epoch', color=DARK_MUTE)
        self.ax.set_ylabel('Valeur', color=DARK_MUTE)
        self.canvas = FigureCanvasTkAgg(self.fig, master=card)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ----- Helpers -----
    def browse_file(self):
        path = filedialog.askopenfilename(title="Sélectionner le CSV", filetypes=[("CSV", "*.csv"), ("Tous les fichiers","*.*")])
        if path:
            self.csv_path.set(path)

    def open_model_dir(self):
        d = os.path.abspath(Config.model_dir)
        os.makedirs(d, exist_ok=True)
        if sys.platform.startswith('win'):
            os.startfile(d)
        elif sys.platform == 'darwin':
            os.system(f'open "{d}"')
        else:
            os.system(f'xdg-open "{d}"')

    def log(self, msg: str):
        self.txt.insert('end', msg + "\n")
        self.txt.see('end')
        self.root.update_idletasks()

    def update_progress(self, epoch_done: int, total_epochs: int, logs: Dict):
        pct = int(100 * epoch_done / max(total_epochs,1))
        self.progress['value'] = pct
        self.lbl_pct.config(text=f"{pct}%")
        loss = logs.get('loss', float('nan'))
        acc = logs.get('accuracy', float('nan'))
        val_loss = logs.get('val_loss', float('nan'))
        val_acc = logs.get('val_accuracy', float('nan'))
        self.log(f"Epoch {epoch_done}/{total_epochs} - loss={loss:.4f} acc={acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if not hasattr(self, '_hist_cache'):
            self._hist_cache = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self._hist_cache['loss'].append(loss)
        self._hist_cache['accuracy'].append(acc)
        self._hist_cache['val_loss'].append(val_loss)
        self._hist_cache['val_accuracy'].append(val_acc)

        self.ax.clear()
        self.ax.set_facecolor('#0b1220')
        self.fig.patch.set_facecolor(DARK_PANEL)
        self.ax.tick_params(colors=DARK_MUTE)
        self.ax.spines['bottom'].set_color(DARK_MUTE)
        self.ax.spines['left'].set_color(DARK_MUTE)
        self.ax.plot(self._hist_cache['loss'], label='loss')
        self.ax.plot(self._hist_cache['val_loss'], label='val_loss')
        self.ax.plot(self._hist_cache['accuracy'], label='acc')
        self.ax.plot(self._hist_cache['val_accuracy'], label='val_acc')
        self.ax.legend(facecolor=DARK_PANEL, edgecolor=DARK_MUTE, labelcolor=DARK_TEXT)
        self.ax.set_title('Loss & Accuracy', color=DARK_TEXT)
        self.ax.set_xlabel('Epoch', color=DARK_MUTE)
        self.ax.set_ylabel('Valeur', color=DARK_MUTE)
        self.canvas.draw()

    def enable_controls(self, enabled: bool=True):
        state = 'normal' if enabled else 'disabled'
        self.btn_start.config(state=state)

    def _read_params(self) -> bool:
        try:
            self.cfg.window_bars = int(self.entries['window_bars'].get())
            self.cfg.hist_bins = int(self.entries['hist_bins'].get())
            self.cfg.band_width_pct = float(self.entries['band_width_pct'].get())
            self.cfg.accumulation_max_width_pct = float(self.entries['accumulation_max_width_pct'].get())
            self.cfg.lookahead = int(self.entries['lookahead'].get())
            self.cfg.touch_tolerance_pct = float(self.entries['touch_tolerance_pct'].get())
            self.cfg.seq_len = int(self.entries['seq_len'].get())
            self.cfg.batch_size = int(self.entries['batch_size'].get())
            self.cfg.epochs = int(self.entries['epochs'].get())
            self.cfg.learning_rate = float(self.entries['learning_rate'].get())
            return True
        except Exception as e:
            messagebox.showerror("Erreur", f"Paramètre invalide: {e}")
            return False

    def start_training(self):
        path = self.csv_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Erreur", "Veuillez choisir un fichier CSV valide.")
            return
        if not self._read_params():
            return

        self.enable_controls(False)
        self.progress['value'] = 0
        self.lbl_pct.config(text="0%")
        self.txt.delete('1.0','end')
        self._hist_cache = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        self.log("Lancement de l'entraînement… (pas d'arrêt anticipé)")
        t = threading.Thread(target=train_pipeline, args=(path, self.cfg, self), daemon=True)
        t.start()

    def training_done(self, history: Dict, report: Dict):
        self.log("\n✅ Entraînement terminé.")
        self.log(f"Test accuracy: {report['test_acc']:.4f} | Test loss: {report['test_loss']:.4f}")
        out_dir = os.path.abspath(self.cfg.model_dir)
        self.log(f"Modèle & scaler sauvegardés dans: {out_dir}")
        self.model_saved.set(out_dir)
        self.enable_controls(True)

# ==========================
# Main
# ==========================

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.random.set_seed(42)
    np.random.seed(42)
    root = tk.Tk()
    app = TrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
