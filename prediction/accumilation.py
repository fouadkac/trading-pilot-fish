# =====================================================
#  Modèle LSTM pour prédiction de breakout
# =====================================================

# -----------------------------
# Importation des bibliothèques
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import messagebox, filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import chardet
import os

# -----------------------------
# Étape 1: Chargement des données
# -----------------------------
file_path = "export_accumulation.csv"
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read(10000))
encoding = result['encoding']

df = pd.read_csv(file_path, sep=";", encoding=encoding)
df['Time'] = pd.to_datetime(df['Time'])
df.sort_values('Time', inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------
# Étape 2: Identifier les breakouts
# -----------------------------
def detect_breakout(row):
    if row['Close'] > row['RangeTop']:
        return 1   # breakout haussier
    elif row['Close'] < row['RangeBottom']:
        return -1  # breakout baissier
    else:
        return 0   # pas de breakout

df['Breakout'] = df.apply(detect_breakout, axis=1)

# -----------------------------
# Étape 3: Préparer les données
# -----------------------------
features = ['Open', 'High', 'Low', 'Close', 'ATR', 'HMA14']
X = df[features].values
y = df['Breakout'].replace(-1, 2).values  # -1 devient 2 (3 classes)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transformation en séquences pour LSTM
sequence_length = 20  # nombre de pas temporels
X_seq, y_seq = [], []
for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# -----------------------------
# Étape 4: Modèle LSTM
# -----------------------------
def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(3, activation="softmax")  # sortie 3 classes
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model((sequence_length, X_seq.shape[2]))
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

# -----------------------------
# Étape 5: Évaluation
# -----------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Prédiction sur toutes les données
y_all_pred = np.argmax(model.predict(X_seq), axis=1)
df = df.iloc[sequence_length:].copy()
df["BreakoutPred"] = y_all_pred
df["BreakoutPred"] = df["BreakoutPred"].replace({2:-1})

# -----------------------------
# Étape 6: Sauvegarde modèle + scaler
# -----------------------------
model_dir = "trained_model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "breakout_lstm_model.h5")
scaler_path = os.path.join(model_dir, "scaler_accum.gz")

model.save(model_path)
print(f"✅ Modèle LSTM enregistré dans : {model_path}")

joblib.dump(scaler, scaler_path)
print(f"✅ Scaler enregistré dans : {scaler_path}")

# -----------------------------
# Étape 7: Interface graphique
# -----------------------------
class RangeApp:
    def __init__(self, master, df):
        self.master = master
        self.df = df
        self.master.title("Analyse des zones d'accumulation et prédiction LSTM")
        
        # Frame principal
        self.frame = tk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=1)

        # Figure Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
        
        # Sliders pour plage de dates
        tk.Label(master, text="Indice de début").pack()
        self.start_slider = tk.Scale(master, from_=0, to=len(df)-1, orient=tk.HORIZONTAL, length=400)
        self.start_slider.pack()
        tk.Label(master, text="Indice de fin").pack()
        self.end_slider = tk.Scale(master, from_=0, to=len(df)-1, orient=tk.HORIZONTAL, length=400)
        self.end_slider.set(len(df)-1)
        self.end_slider.pack()
        
        # Boutons
        tk.Button(master, text="Tracer le graphique", command=self.plot_graph).pack(pady=5)
        tk.Button(master, text="Sauvegarder le graphique", command=self.save_graph).pack(pady=5)
    
    def plot_graph(self):
        self.ax.clear()
        start_idx = self.start_slider.get()
        end_idx = self.end_slider.get()
        if start_idx >= end_idx:
            messagebox.showerror("Erreur", "Indice début < indice fin")
            return
        
        df_plot = self.df.iloc[start_idx:end_idx+1]
        
        # Close
        self.ax.plot(df_plot["Time"], df_plot["Close"], color="blue", label="Close")
        
        # Zone d'accumulation
        self.ax.fill_between(df_plot["Time"], df_plot["RangeBottom"], df_plot["RangeTop"], 
                             color="yellow", alpha=0.3, label="Zone d’accumulation")
        
        # Breakouts réels
        self.ax.scatter(df_plot[df_plot["Breakout"]==1]["Time"], df_plot[df_plot["Breakout"]==1]["Close"],
                        color="green", marker="^", label="Breakout Up", s=50)
        self.ax.scatter(df_plot[df_plot["Breakout"]==-1]["Time"], df_plot[df_plot["Breakout"]==-1]["Close"],
                        color="red", marker="v", label="Breakout Down", s=50)
        
        # Breakouts prédits
        self.ax.scatter(df_plot[df_plot["BreakoutPred"]==1]["Time"], df_plot[df_plot["BreakoutPred"]==1]["Close"],
                        color="lime", marker="o", label="Pred Up", alpha=0.5, s=40)
        self.ax.scatter(df_plot[df_plot["BreakoutPred"]==-1]["Time"], df_plot[df_plot["BreakoutPred"]==-1]["Close"],
                        color="orange", marker="o", label="Pred Down", alpha=0.5, s=40)
        
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Prix")
        self.ax.legend()
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def save_graph(self):
        file = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files","*.png"),("All files","*.*")])
        if file:
            self.fig.savefig(file)
            messagebox.showinfo("Succès", f"Graphique sauvegardé : {file}")

# -----------------------------
# Lancer l’interface graphique
# -----------------------------
root = tk.Tk()
app = RangeApp(root, df)
root.mainloop()
