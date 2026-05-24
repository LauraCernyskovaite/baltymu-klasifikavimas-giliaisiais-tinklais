"""
================================================================
FERMENTŲ / BALTYMŲ BE EC KLASIFIKATORIUS
Versija su iš anksto paruoštu MMseqs2 klasteriniu padalinimu
================================================================

Naudoja:
   train.csv
   val.csv
   test.csv

Šis scenarijus atitinka pagrindinį bakalauro darbo eksperimentą:
   1) CNN pagal aminorūgščių seką;
   2) MLP pagal 29 biocheminius požymius;
   3) multimodalus CNN ir MLP modelis;
   4) Random Forest bazinis palyginamasis modelis.

CSV stulpeliai:
   sequence, label

label:
   1 = baltymas su EC numeriu
   0 = baltymas be EC numerio po papildomo filtravimo
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import re
import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, Model, regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
    print("Biopython rastas.")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Biopython nerastas - kai kurie pozymiai bus 0.0.")

"""
================================================================
1) NUSTATYMAI
----------------------------------------------------------------
Visi pagrindiniai hiperparametrai surinkti vienoje vietoje, kad
eksperimentą būtų galima atkartoti arba kryptingai pakeisti.

MAX_LEN=1024 — fiksuotas aminorūgščių sekos ilgis CNN įvesčiai.
Ilgesnėms sekoms paliekama pradžia ir pabaiga, o trumpesnės sekos
papildomos nuliais.

EMB_DIM=32 — kiekvienas aminorūgšties indeksas embedding sluoksnyje
paverčiamas 32 matmenų mokomu skaitiniu vektoriumi.

LR=3e-4 — Adam optimizatoriaus mokymosi greitis, naudotas visiems
neuroninių tinklų modeliams šiame eksperimente.
"""

SEED = 42

MAX_LEN = 1024
EMB_DIM = 32
BATCH = 64
EPOCHS = 40
LR = 3e-4

DATA_DIR = "."
RESULTS_DIR = "results_cluster_split_1024"

for sub in ["images", "models", "reports", "eda", "objects"]:
    Path(RESULTS_DIR, sub).mkdir(parents=True, exist_ok=True)

# Fiksuojamos atsitiktinių skaičių sėklos reprodukuojamumui:
# tas pats kodas, tie patys duomenys => tie patys rezultatai.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def find_data_dir():
    if DATA_DIR is not None:
        return Path(DATA_DIR)

    candidates = sorted(
        Path(".").glob("data_uniprot_clean_cluster_alphafold_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        if all((candidate / name).exists() for name in ["train.csv", "val.csv", "test.csv"]):
            return candidate

    fallback = Path("uniprot_clean_cluster_alphafold_dataset")
    if all((fallback / name).exists() for name in ["train.csv", "val.csv", "test.csv"]):
        return fallback

    raise FileNotFoundError("Nerastas data katalogas su train.csv, val.csv ir test.csv.")


DATA_PATH = find_data_dir()

print("=" * 70)
print("FERMENTU / BALTYMU BE EC KLASIFIKATORIUS")
print("Naudojamas MMseqs2 cluster split: train.csv / val.csv / test.csv")
print(f"Duomenu aplankas: {DATA_PATH}")
print(f"MAX_LEN={MAX_LEN}, EMB_DIM={EMB_DIM}, BATCH={BATCH}, EPOCHS={EPOCHS}")
print(f"Rezultatai: {RESULTS_DIR}")
print("=" * 70)

"""
================================================================
2) AMINORŪGŠČIŲ VALYMAS IR KODAVIMAS
----------------------------------------------------------------
Neuroniniai tinklai apdoroja skaitinius duomenis, todėl
aminorūgščių seka paverčiama sveikųjų skaičių masyvu.

clean_aa():  
    Pašalina ne raidinius simbolius, paverčia raides didžiosiomis
    ir nestandartinius aminorūgščių simbolius pakeičia pasirinktais
    standartiniais atitikmenimis. Neatpažinti simboliai žymimi X.

encode_seq():
    Kiekvienai aminorūgščiai priskiria indeksą. Reikšmė 0 paliekama
    sekos papildymui (padding), o X turi atskirą indeksą nežinomoms
    aminorūgštims.
================================================================
"""

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)

AA2IDX = {aa: i + 1 for i, aa in enumerate(AA)}
AA2IDX["X"] = len(AA2IDX) + 1
VOCAB_SIZE = len(AA2IDX) + 1


def clean_aa(seq: str) -> str:
    seq = re.sub(r"[^A-Za-z]", "", str(seq)).upper()
    seq = (
        seq.replace("U", "C")
        .replace("O", "K")
        .replace("B", "D")
        .replace("Z", "E")
        .replace("J", "I")
    )
    return "".join(ch if ch in AA2IDX else "X" for ch in seq)


def encode_seq(seq: str, max_len: int = MAX_LEN) -> np.ndarray:
    seq = str(seq)

    # "Start+end" strategija: išsaugomi abu sekos galai.
    # Tai praktiškas kompromisas fiksuoto ilgio CNN įvesčiai.
    if len(seq) > max_len:
        left = max_len // 2
        right = max_len - left
        seq = seq[:left] + seq[-right:]

    ids = [AA2IDX.get(ch, AA2IDX["X"]) for ch in seq]

    # Papildymas nuliais (zero-padding) iki fiksuoto ilgio —
    # CNN ir Embedding sluoksniai reikalauja vienodo dydžio įvesties.
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))

    return np.array(ids[:max_len], dtype=np.int32)

"""
================================================================
3) BIOCHEMINIAI POZYMIAI
-----------------------------------------------------------------
Kiekvienai sekai apskaičiuojamas 29 biocheminių požymių rinkinys.
Šie požymiai apibūdina ne lokalią aminorūgščių tvarką, o globalias
baltymo fizines ir chemines savybes.

Požymių grupės:
  - 8 požymiai iš Biopython ProteinAnalysis:
    molekulinė masė, izoelektrinis taškas, GRAVY, aromatiškumas,
    nestabilumo indeksas ir 3 numatomos antrinės struktūros dalys.
  - 1 sekos ilgio požymis.
  - 1 cisteino tankio požymis.
  - 9 aminorūgščių grupių santykinės dalys.
  - 10 pasirinktų pavienių aminorūgščių santykinės dalys.

Papildomai sudaromas 27 požymių rinkinys be sekos ilgio ir
molekulinės masės. Jis naudojamas abliacijos tyrime.
"""

GROUPS = {
    "hydrophobic": set("AVILMFYW"),
    "polar": set("STNQCYH"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "charged": set("KRHDE"),
    "tiny": set("AGS"),
    "small": set("AGSCTPVND"),
    "aliphatic": set("ILV"),
    "aromatic": set("FYWH"),
}

INDIVIDUAL_AA = list("CGPWMHKRDE")

BIO_FEATURES = (
    [
        "length",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "aromaticity",
        "instability_index",
        "helix_fraction",
        "turn_fraction",
        "sheet_fraction",
        "cysteine_density",
    ]
    + [f"{g}_fraction" for g in GROUPS]
    + [f"aa_{aa}_fraction" for aa in INDIVIDUAL_AA]
)

# Abliacijos tyrimas: pašalinami sekos ilgis ir molekulinė masė.
# Taip tikrinama, ar modelis nesiremia vien lengvai išmokstamais
# dydžio požymiais.
BIO_WITHOUT_LENGTH_MW = [
    f for f in BIO_FEATURES
    if f not in ["length", "molecular_weight"]
]


def calculate_bio_features(seq: str):
    clean = "".join(aa for aa in str(seq) if aa in AA_SET)

    if len(clean) < 5:
        return None

    n = len(clean)
    feats = {"length": float(n)}

    for grp, members in GROUPS.items():
        feats[f"{grp}_fraction"] = sum(aa in members for aa in clean) / n

    for aa in INDIVIDUAL_AA:
        feats[f"aa_{aa}_fraction"] = clean.count(aa) / n

    # Cisteino tankis aprašo santykinį cisteino liekanų kiekį.
    # Cisteinai gali sudaryti disulfidinius ryšius, todėl šis
    # požymis gali būti susijęs su struktūros stabilumu.
    feats["cysteine_density"] = (clean.count("C") / 2.0) / max(n / 100.0, 1.0)

    feats.update({
        "molecular_weight": 0.0,
        "isoelectric_point": 0.0,
        "gravy": 0.0,
        "aromaticity": 0.0,
        "instability_index": 0.0,
        "helix_fraction": 0.0,
        "turn_fraction": 0.0,
        "sheet_fraction": 0.0,
    })

    if BIOPYTHON_AVAILABLE:
        try:
            pa = ProteinAnalysis(clean)
            h, t, s = pa.secondary_structure_fraction()
            feats.update({
                "molecular_weight": float(pa.molecular_weight()),
                "isoelectric_point": float(pa.isoelectric_point()),
                "gravy": float(pa.gravy()),
                "aromaticity": float(pa.aromaticity()),
                "instability_index": float(pa.instability_index()),
                "helix_fraction": float(h),
                "turn_fraction": float(t),
                "sheet_fraction": float(s),
            })
        except Exception:
            pass

    return {k: float(feats[k]) for k in BIO_FEATURES}

"""
# ================================================================
# 4) PAGALBINĖS FUNKCIJOS
# ----------------------------------------------------------------
# Pagalbiniai įrankiai, naudojami per visą eksperimentą.
#
# load_split_csv()  — nuskaito CSV, patikrina stulpelius,
#   ištrina tuščias eilutes, išfiltruoja per trumpas sekas (< 5 AA).
#
# make_bio_df()     — iteruoja per DataFrame ir skaičiuoja
#   biocheminius požymius kiekvienai sekai.
#
# get_class_weights() — apskaičiuoja klasių svorius.
#   Jei treniravimo rinkinyje yra 60% fermento ir 40% ne fermento,
#   retesnės klasės klaidos baudžiamos labiau. Tai kompensuoja
#   nedidelį klasių disbalansą ir pagerina jautrumą mažesnei klasei.
#
# find_best_threshold() — ieškomas slenkstis tarp 0 ir 1, kuris
#   maksimizuoja F1 rodiklį validacijos rinkinyje.
#   Testavimo rinkinys slenksčiui parinkti nenaudojamas.
#
# evaluate_model()  — apskaičiuoja visas metrikas:
#   ROC-AUC, Accuracy, F1, MCC ir sumaišymo matricą. Kode taip pat
#   išsaugomas Average Precision, tačiau pagrindiniame darbo tekste
#   modeliai pagal šią metriką nelyginami.
# ================================================================
"""

def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"sequence", "label"}.issubset(df.columns), f"{path} turi tureti sequence ir label stulpelius."
    df = df.dropna(subset=["sequence", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    df["clean_seq"] = df["sequence"].apply(clean_aa)
    df["seq_len"] = df["clean_seq"].str.len()
    df = df[df["seq_len"] >= 5].reset_index(drop=True)
    return df


def make_bio_df(df: pd.DataFrame) -> pd.DataFrame:
    bio_list = []
    for seq in df["clean_seq"]:
        feats = calculate_bio_features(seq)
        if feats is None:
            feats = {k: 0.0 for k in BIO_FEATURES}
        bio_list.append(feats)
    return pd.DataFrame(bio_list, columns=BIO_FEATURES)


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_model(model_name, y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "avg_precision": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "threshold": float(threshold),
        "probs": probs,
        "preds": preds,
        "confusion_matrix": confusion_matrix(y_true, preds),
    }


def strip_arrays_for_json(res):
    return {
        "model": res["model"],
        "roc_auc": res["roc_auc"],
        "avg_precision": res["avg_precision"],
        "accuracy": res["accuracy"],
        "f1": res["f1"],
        "mcc": res["mcc"],
        "threshold": res["threshold"],
        "confusion_matrix": res["confusion_matrix"].tolist(),
    }


def plot_training_history(history, model_name, out_path):
    hist = history.history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title(f"{model_name}: Loss")
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("auc", []), label="auc")
    plt.plot(hist.get("val_auc", []), label="val_auc")
    plt.title(f"{model_name}: AUC")
    plt.xlabel("Epocha")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_callbacks(model_name):
    # ----------------------------------------------------------------
    # Treniravimo valdymo callbacks:
    #
    # EarlyStopping — sustabdo mokymąsi, jei val_auc neauga
    #   6 epochas iš eilės, ir atkuria geriausius svorius.
    #   Taip išvengiama permokymosi (overfitting).
    #
    # ReduceLROnPlateau — jei val_loss nemažėja 3 epochas,
    #   mokymosi greitis automatiškai perpus sumažinamas.
    #   Leidžia modeliui "įlipti" į tikslesnį minimumą.
    #
    # ModelCheckpoint — išsaugo geriausią modelį pagal val_auc.
    #   Užtikrina, kad net jei vėliau rezultatai blogėja, turima geriausia versija.
    #
    # CSVLogger — kiekvienos epochos metrikos išsaugomos CSV —
    #   vėliau galima analizuoti mokymosi eigą.
    # ----------------------------------------------------------------
    return [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=f"{RESULTS_DIR}/models/{model_name}.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.CSVLogger(f"{RESULTS_DIR}/reports/{model_name}_training_log.csv"),
    ]


# ================================================================
# 5) MODELIŲ ARCHITEKTŪROS
# ----------------------------------------------------------------
# Trys skirtingos neuroninių tinklų architektūros, leidžiančios
# palyginti, kuri informacija svarbesnė klasifikavimui.
#
# build_seq_branch() — bendra sekos apdorojimo šaka,
#   naudojama tiek grynajame CNN, tiek multimodaliniame modelyje.
#
#   Embedding(VOCAB, EMB_DIM):
#     Kiekvienas skaičius (AA kodas 1–21) paverčiamas
#     EMB_DIM=32 matmenų vektoriumi. Šie vektoriai išmokami
#     mokymo metu ir perduodami tolesniems Conv1D sluoksniams.
#
#   Conv1D(64, 9) — 64 filtrai, kiekvienas "slankioja" per
#     9 gretimų AA langą. Taip aptinkami lokalūs sekos motyvai
#     arba aminorūgščių deriniai.
#
#   Conv1D(64, 5) — antras sluoksnis ieško motyvų jau
#     sutrumpintoje (po MaxPooling) sekoje — trumpesnių šablonų.
#
#   GlobalAveragePooling1D + GlobalMaxPooling1D + Concatenate:
#     Average apibendrina bendrą signalą per visą seką.
#     Max išsaugo stipriausią aptiktą motyvą, nepriklausomai
#     nuo jo pozicijos sekoje.
#
#   L2 regularizacija (1e-4) — baudžia už per dideles svorių
#     reikšmes, mažina permokymosi riziką.
#
# build_bio_mlp() — paprastas daugiasluoksnis perceptronas (MLP)
#   biocheminiams požymiams. Trijų sluoksnių architektūra su
#   BatchNorm ir Dropout. Modelis įvertina, kiek informacijos
#   klasifikavimui suteikia globalūs biocheminiai požymiai.
#
# build_multimodal_cnn_bio() — dviejų šakų (multi-input) modelis:
#   Sekos CNN šaka + Bio MLP šaka => Concatenate => klasifikatorius.
#   Potencialiai naudingas, jei sekos ir biocheminiai požymiai
#   suteikia vienas kitą papildančios informacijos.
# ================================================================

def build_seq_branch(seq_input):
    x = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMB_DIM,
        mask_zero=False,
        name="embedding",
    )(seq_input)

    x = layers.Conv1D(64, 9, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.30)(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    return x


def build_seq_cnn():
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="seq_input")
    x = build_seq_branch(inp)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="Seq_CNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_bio_mlp(n_features, model_name="Bio_MLP"):
    inp = layers.Input(shape=(n_features,), name="bio_input")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name=model_name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_multimodal_cnn_bio(n_features):
    seq_inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="seq_input")
    x1 = build_seq_branch(seq_inp)

    bio_inp = layers.Input(shape=(n_features,), name="bio_input")
    x2 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(bio_inp)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.30)(x2)
    x2 = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x2)
    x2 = layers.Dropout(0.20)(x2)

    # Sujungiamos abi šakos į vieną vektorių.
    # Taip modelis gali naudoti tiek sekos motyvus (x1),
    # tiek globalias biochemines savybes (x2) kartu.
    merged = layers.Concatenate()([x1, x2])
    z = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(merged)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.40)(z)
    z = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(z)
    z = layers.Dropout(0.30)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = Model([seq_inp, bio_inp], out, name="Multimodal_CNN_BioMLP")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ================================================================
# 6) DUOMENŲ ĮKĖLIMAS IR PARUOŠIMAS
# ----------------------------------------------------------------
# Nuskaitomi trys iš anksto paruošti CSV failai (train/val/test),
# sukurti MMseqs2 klasterinio padalinimo metodu.
#
# Kodėl MMseqs2, o ne atsitiktinis padalinimas?
#   Baltymų sekos evoliuciškai konservuotos — žmogaus ir pelės
#   laktatdehidrogenazė gali turėti >90% identišką seką.
#   Atsitiktinis padalinimas leistų tokioms sekoms patekti ir
#   į treniravimo, ir į testavimo rinkinius, dirbtinai
#   padidindamas rezultatus (homologinis nutekėjimas).
#   MMseqs2 grupuoja sekas į klasterius pagal pasirinktus parametrus,
#   tada klasteriai paskirstomi taip, kad vienas klasteris patektų
#   tik į vieną rinkinį. Tai sumažina, bet visiškai nepanaikina,
#   homologinio ar funkcinio panašumo nutekėjimo rizikos.
#
# Biocheminiai požymiai skaičiuojami atskirai kiekvienam
# rinkiniui. StandardScaler PRITAIKOMAS tik treniravimo rinkiniui (fit),
# o val ir test tik transformuojami — taip išvengiama
# informacijos nutekėjimo iš val/test į mokymosi procesą.
#
# Išsaugomi du scaler'iai:
#   bio_scaler.joblib — visų požymių (29)
#   bio_scaler_without_length_mw.joblib — be length/MW (abliacijos)
# ================================================================

df_train = load_split_csv(DATA_PATH / "train.csv")
df_val = load_split_csv(DATA_PATH / "val.csv")
df_test = load_split_csv(DATA_PATH / "test.csv")

print("\nSplit dydziai:")
print(f"Train: {len(df_train)}")
print(f"Val:   {len(df_val)}")
print(f"Test:  {len(df_test)}")
print("\nKlasiu balansas:")
print("Train:")
print(df_train["label"].value_counts().sort_index())
print("Val:")
print(df_val["label"].value_counts().sort_index())
print("Test:")
print(df_test["label"].value_counts().sort_index())

# EDA grafikas: sekų ilgių histograma kiekvienam rinkiniui.
# Leidžia vizualiai patikrinti, ar train/val/test turi
# panašias sekų ilgių distribucijas (svarbu teisingo padalinimo požymis).
plt.figure(figsize=(9, 5))
plt.hist(df_train["seq_len"], bins=40, alpha=0.5, label="Train")
plt.hist(df_val["seq_len"], bins=40, alpha=0.5, label="Val")
plt.hist(df_test["seq_len"], bins=40, alpha=0.5, label="Test")
plt.axvline(MAX_LEN, linestyle="--", label=f"MAX_LEN={MAX_LEN}")
plt.xlabel("Sekos ilgis, aa")
plt.ylabel("Daznis")
plt.title("Seku ilgiu pasiskirstymas pagal split")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/eda/sequence_lengths_by_split.png", dpi=150)
plt.close()

print("\nSkaiciuojami biocheminiai pozymiai...")
bio_train_df = make_bio_df(df_train)
bio_val_df = make_bio_df(df_val)
bio_test_df = make_bio_df(df_test)

X_seq_train = np.vstack([encode_seq(s) for s in df_train["clean_seq"]])
X_seq_val = np.vstack([encode_seq(s) for s in df_val["clean_seq"]])
X_seq_test = np.vstack([encode_seq(s) for s in df_test["clean_seq"]])

X_bio_train_raw = bio_train_df.to_numpy(dtype=np.float32)
X_bio_val_raw = bio_val_df.to_numpy(dtype=np.float32)
X_bio_test_raw = bio_test_df.to_numpy(dtype=np.float32)

X_nolm_train_raw = bio_train_df[BIO_WITHOUT_LENGTH_MW].to_numpy(dtype=np.float32)
X_nolm_val_raw = bio_val_df[BIO_WITHOUT_LENGTH_MW].to_numpy(dtype=np.float32)
X_nolm_test_raw = bio_test_df[BIO_WITHOUT_LENGTH_MW].to_numpy(dtype=np.float32)

y_train = df_train["label"].to_numpy(dtype=np.int32)
y_val = df_val["label"].to_numpy(dtype=np.int32)
y_test = df_test["label"].to_numpy(dtype=np.int32)

# fit() atliekamas tik su treniravimo duomenimis — išvengiama val/test nutekėjimo.
scaler = StandardScaler()
X_bio_train = scaler.fit_transform(X_bio_train_raw)
X_bio_val = scaler.transform(X_bio_val_raw)
X_bio_test = scaler.transform(X_bio_test_raw)

scaler_nolm = StandardScaler()
X_nolm_train = scaler_nolm.fit_transform(X_nolm_train_raw)
X_nolm_val = scaler_nolm.transform(X_nolm_val_raw)
X_nolm_test = scaler_nolm.transform(X_nolm_test_raw)

joblib.dump(scaler, f"{RESULTS_DIR}/objects/bio_scaler.joblib")
joblib.dump(scaler_nolm, f"{RESULTS_DIR}/objects/bio_scaler_without_length_mw.joblib")

class_weights = get_class_weights(y_train)
print(f"\nClass weights: {class_weights}")


# ================================================================
# 7) MODELIŲ MOKYMAS
# ----------------------------------------------------------------
# Penki modeliai treniruojami nuosekliai ir jų rezultatai
# kaupiami žodyne `results`. Kiekvienas modelis:
#   1. Sukuriamas (build_*)
#   2. Treniruojamas su EarlyStopping (fit)
#   3. Geriausias slenkstis randamas iš validacijos
#   4. Įvertinamas testavimo rinkinyje (evaluate_model)
#
# 1 etapas: CNN pagal seką
#   Naudoja tik koduotą aminorūgščių seką. Leidžia įvertinti,
#   kiek informacijos yra grynoje sekoje be papildomų požymių.
#
# 2 etapas: Bio MLP
#   Naudoja 29 biocheminius požymius ir parodo, kiek informacijos
#   suteikia globalios fizinės bei cheminės baltymo savybės.
#
# 2B etapas: Bio MLP be length/MW (abliacijos tyrimas)
#   Pašalinus ilgį ir masę, tikrinama, ar modelis nesiremia
#   vien šiais dviem lengvai atskiriamais požymiais.
#
# 3 etapas: Multimodal CNN + Bio MLP
#   Dviejų šakų modelis, naudojantis abi informacijos rūšis.
#   Gali pranokti atskirus modelius, jei abi informacijos rūšys
#   papildo viena kitą.
#
# Random Forest (bazinis modelis)
#   Klasikinis ansamblio metodas, skirtas palyginti su neuroniniais tinklais.
#   Taip pat treniruojamas be length/MW abliacijos tikslais.
# ================================================================

results = {}

print("\n" + "=" * 70)
print("1 ETAPAS: CNN PAGAL SEKA")
print("=" * 70)
seq_cnn = build_seq_cnn()
history_seq = seq_cnn.fit(
    X_seq_train,
    y_train,
    validation_data=(X_seq_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    callbacks=get_callbacks("seq_cnn"),
    verbose=1,
)
plot_training_history(history_seq, "CNN pagal seka", f"{RESULTS_DIR}/images/history_seq_cnn.png")
val_probs = seq_cnn.predict(X_seq_val, verbose=0).ravel()
test_probs = seq_cnn.predict(X_seq_test, verbose=0).ravel()
thr = find_best_threshold(y_val, val_probs)
results["seq_cnn"] = evaluate_model("CNN pagal seka", y_test, test_probs, thr)

print("\n" + "=" * 70)
print("2 ETAPAS: BIO MLP")
print("=" * 70)
bio_mlp = build_bio_mlp(X_bio_train.shape[1], "Bio_MLP")
history_bio = bio_mlp.fit(
    X_bio_train,
    y_train,
    validation_data=(X_bio_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    callbacks=get_callbacks("bio_mlp"),
    verbose=1,
)
plot_training_history(history_bio, "Bio MLP", f"{RESULTS_DIR}/images/history_bio_mlp.png")
val_probs = bio_mlp.predict(X_bio_val, verbose=0).ravel()
test_probs = bio_mlp.predict(X_bio_test, verbose=0).ravel()
thr = find_best_threshold(y_val, val_probs)
results["bio_mlp"] = evaluate_model("Bio MLP", y_test, test_probs, thr)

print("\n" + "=" * 70)
print("2B ETAPAS: BIO MLP BE LENGTH/MW")
print("=" * 70)
bio_mlp_nolm = build_bio_mlp(X_nolm_train.shape[1], "Bio_MLP_without_length_MW")
history_bio_nolm = bio_mlp_nolm.fit(
    X_nolm_train,
    y_train,
    validation_data=(X_nolm_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    callbacks=get_callbacks("bio_mlp_without_length_mw"),
    verbose=1,
)
plot_training_history(history_bio_nolm, "Bio MLP be length/MW", f"{RESULTS_DIR}/images/history_bio_mlp_without_length_mw.png")
val_probs = bio_mlp_nolm.predict(X_nolm_val, verbose=0).ravel()
test_probs = bio_mlp_nolm.predict(X_nolm_test, verbose=0).ravel()
thr = find_best_threshold(y_val, val_probs)
results["bio_mlp_without_length_mw"] = evaluate_model("Bio MLP be length/MW", y_test, test_probs, thr)

print("\n" + "=" * 70)
print("3 ETAPAS: CNN + BIO MLP")
print("=" * 70)
multi_model = build_multimodal_cnn_bio(X_bio_train.shape[1])
history_multi = multi_model.fit(
    [X_seq_train, X_bio_train],
    y_train,
    validation_data=([X_seq_val, X_bio_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    callbacks=get_callbacks("multimodal_cnn_bio"),
    verbose=1,
)
plot_training_history(history_multi, "CNN + Bio MLP", f"{RESULTS_DIR}/images/history_multimodal_cnn_bio.png")
val_probs = multi_model.predict([X_seq_val, X_bio_val], verbose=0).ravel()
test_probs = multi_model.predict([X_seq_test, X_bio_test], verbose=0).ravel()
thr = find_best_threshold(y_val, val_probs)
results["multimodal"] = evaluate_model("CNN + Bio MLP", y_test, test_probs, thr)

print("\n" + "=" * 70)
print("RANDOM FOREST BAZINIS MODELIS")
print("=" * 70)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_bio_train, y_train)
joblib.dump(rf, f"{RESULTS_DIR}/models/random_forest_bio.joblib")
val_probs = rf.predict_proba(X_bio_val)[:, 1]
test_probs = rf.predict_proba(X_bio_test)[:, 1]
thr = find_best_threshold(y_val, val_probs)
results["rf"] = evaluate_model("Random Forest bio", y_test, test_probs, thr)

rf_nolm = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf_nolm.fit(X_nolm_train, y_train)
joblib.dump(rf_nolm, f"{RESULTS_DIR}/models/random_forest_without_length_mw.joblib")
val_probs = rf_nolm.predict_proba(X_nolm_val)[:, 1]
test_probs = rf_nolm.predict_proba(X_nolm_test)[:, 1]
thr = find_best_threshold(y_val, val_probs)
results["rf_without_length_mw"] = evaluate_model("RF be length/MW", y_test, test_probs, thr)


# ================================================================
# 8) REZULTATAI IR ATASKAITOS
# ----------------------------------------------------------------
# Visi modelių rezultatai surenkami į summary_df DataFrame ir
# išsaugomi. Modeliai skirstomi į dvi grupes:
#
# MAIN_MODEL_KEYS — pagrindiniai 4 modeliai lyginami tarpusavyje:
#   seq_cnn, bio_mlp, multimodal, rf
#
# ABLATION_MODEL_KEYS — abliacijos/artefakto patikros modeliai:
#   bio_mlp_without_length_mw, rf_without_length_mw
#   Jie naudojami patikrinti, ar rezultatai nėra pernelyg priklausomi
#   nuo sekos ilgio ir molekulinės masės.
#
# Grafikai:
#   roc_curves.png       — ROC kreivės visiems pagrindiniams modeliams
#   model_comparison.png — stulpelinė diagrama AUC / Accuracy / F1
#   confusion_matrices.png — 4 klaidų matricos viename paveikslėlyje
#
# Permutacinė svarba (permutation_importance):
#   Kiekvienam požymiui apskaičiuojama, kiek AUC nukrenta, kai
#   to požymio reikšmės sumaišomos atsitiktinai testavimo rinkinyje.
#   Tai papildo Random Forest Gini svarbą, nes vertina požymio
#   poveikį jau apmokyto modelio prognozėms.
#   Skaičiuojama 10 kartų (n_repeats=10) ir imamas vidurkis.
# ================================================================

summary_rows = []
for key, res in results.items():
    summary_rows.append({
        "key": key,
        "model": res["model"],
        "roc_auc": res["roc_auc"],
        "avg_precision": res["avg_precision"],
        "accuracy": res["accuracy"],
        "f1": res["f1"],
        "mcc": res["mcc"],
        "threshold": res["threshold"],
    })

summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
summary_df.to_csv(f"{RESULTS_DIR}/reports/model_metrics.csv", index=False)

MAIN_MODEL_KEYS = ["seq_cnn", "bio_mlp", "multimodal", "rf"]
ABLATION_MODEL_KEYS = ["bio_mlp_without_length_mw", "rf_without_length_mw"]

main_summary_df = summary_df[summary_df["key"].isin(MAIN_MODEL_KEYS)].copy()
ablation_summary_df = summary_df[summary_df["key"].isin(ABLATION_MODEL_KEYS)].copy()

best_row = main_summary_df.iloc[0]

print("\n" + "=" * 70)
print("MODELIU PALYGINIMAS")
print("=" * 70)
print(summary_df)

plt.figure(figsize=(9, 7))
for key, res in results.items():
    if key in ["rf_without_length_mw", "bio_mlp_without_length_mw"]:
        continue
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    plt.plot(fpr, tpr, label=f"{res['model']} AUC={res['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC kreives")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/images/roc_curves.png", dpi=150)
plt.close()

plot_df = summary_df[~summary_df["key"].isin(["rf_without_length_mw", "bio_mlp_without_length_mw"])].copy()
x = np.arange(len(plot_df))
w = 0.25
plt.figure(figsize=(11, 6))
plt.bar(x - w, plot_df["roc_auc"], width=w, label="ROC-AUC")
plt.bar(x, plot_df["accuracy"], width=w, label="Accuracy")
plt.bar(x + w, plot_df["f1"], width=w, label="F1")
plt.xticks(x, plot_df["model"], rotation=15)
plt.ylim(0.4, 1.05)
plt.ylabel("Reiksme")
plt.title("Modeliu palyginimas")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/images/model_comparison.png", dpi=150)
plt.close()

models_for_cm = {
    "CNN pagal seka": results["seq_cnn"],
    "Bio MLP": results["bio_mlp"],
    "CNN + Bio MLP": results["multimodal"],
    "Random Forest": results["rf"],
}

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.ravel()
for ax, (name, res) in zip(axes, models_for_cm.items()):
    ConfusionMatrixDisplay(
        confusion_matrix=res["confusion_matrix"],
        display_labels=["Be EC", "Su EC"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/images/confusion_matrices.png", dpi=150)
plt.close()

# Gini svarba — greitai apskaičiuojama iš medžių struktūros.
feature_importance_df = pd.DataFrame({
    "feature": BIO_FEATURES,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
feature_importance_df.to_csv(f"{RESULTS_DIR}/reports/rf_feature_importance.csv", index=False)

# Permutacinė svarba — patikimesnė, bet lėtesnė (n_repeats=10).
perm = permutation_importance(
    rf,
    X_bio_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=SEED,
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "feature": BIO_FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(f"{RESULTS_DIR}/reports/permutation_importance.csv", index=False)

# Pilna testo rinkinio lentelė su visų modelių tikimybėmis,
# spėjimais ir klaidų žymėmis — skirta gilesnei klaidų analizei.
df_test_out = df_test.copy()
df_test_out["true_label"] = y_test
for key, res in results.items():
    df_test_out[f"prob_{key}"] = res["probs"]
    df_test_out[f"pred_{key}"] = res["preds"]
    df_test_out[f"error_{key}"] = df_test_out[f"pred_{key}"] != df_test_out["true_label"]
df_test_out.to_csv(f"{RESULTS_DIR}/reports/test_predictions.csv", index=False)

best_key = str(best_row["key"])
df_best_errors = df_test_out[df_test_out[f"pred_{best_key}"] != df_test_out["true_label"]].copy()
df_best_errors.to_csv(f"{RESULTS_DIR}/reports/best_model_errors.csv", index=False)

report = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(DATA_PATH),
        "dataset_summary_file": str(DATA_PATH / "dataset_summary.json"),
        "data_source": "UniProt Swiss-Prot reviewed",
        "label_1_definition": "baltymas su EC numeriu",
        "label_0_definition": "baltymas be EC numerio po papildomo filtravimo",
        "split_method": "MMseqs2 cluster split from train.csv/val.csv/test.csv",
        "split_ratio": "75/15/10; exact class balance may vary because splitting was performed by MMseqs2 clusters",
        "cluster_split": True,
        "mmseqs_used": True,
        "train_size": int(len(df_train)),
        "val_size": int(len(df_val)),
        "test_size": int(len(df_test)),
        "seed": int(SEED),
        "max_len": int(MAX_LEN),
        "sequence_encoding_strategy": "start_end",
        "sequence_padding_note": (
            "Sekos trumpesnes uz MAX_LEN papildomos nuliais. "
            "Embedding sluoksnyje mask_zero=False, todel padding nera maskuojamas; "
            "tai yra supaprastinimo apribojimas."
        ),
        "embedding_dim": int(EMB_DIM),
        "batch": int(BATCH),
        "epochs": int(EPOCHS),
        "learning_rate": float(LR),
        "alphafold_used": False, 
    },
    "class_balance": {
        "train": {str(k): int(v) for k, v in df_train["label"].value_counts().sort_index().to_dict().items()},
        "val": {str(k): int(v) for k, v in df_val["label"].value_counts().sort_index().to_dict().items()},
        "test": {str(k): int(v) for k, v in df_test["label"].value_counts().sort_index().to_dict().items()},
    },
    "results": summary_df.to_dict("records"),
    "main_model_results": main_summary_df.to_dict("records"),
    "ablation_results": ablation_summary_df.to_dict("records"),
    "detailed_results": {key: strip_arrays_for_json(res) for key, res in results.items()},
    "best_model": {
        "key": str(best_row["key"]),
        "model": str(best_row["model"]),
        "roc_auc": float(best_row["roc_auc"]),
        "accuracy": float(best_row["accuracy"]),
        "f1": float(best_row["f1"]),
        "mcc": float(best_row["mcc"]),
    },
    "artifact_checks": {
        "bio_mlp_auc_change_without_length_mw": float(
            results["bio_mlp_without_length_mw"]["roc_auc"] - results["bio_mlp"]["roc_auc"]
        ),
        "rf_auc_change_without_length_mw": float(
            results["rf_without_length_mw"]["roc_auc"] - results["rf"]["roc_auc"]
        ),
    },
    "methodological_notes": {
        "label_0": "label=0 reiskia baltyma be EC numerio po papildomo filtravimo, ne absoliuciai patvirtinta nefermenta.",
        "homology_leakage": "Naudojamas MMseqs2 klasterinis splitas, todel panasiu seku nutekejimo rizika sumazinta.",
        "alphafold": "AlphaFold PDB failai siame sekos ir biocheminiu pozymiu eksperimente nenaudojami.",
        "random_forest": "Random Forest naudojamas tik kaip klasikinis bazinis palyginamasis modelis, ne kaip gilusis neuroninis tinklas.",
        "ablation_models": "Bio MLP be length/MW ir RF be length/MW naudojami kaip artefakto/abliacijos patikra.",
    },
}

with open(f"{RESULTS_DIR}/reports/final_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("GALUTINE SANTRAUKA")
print("=" * 70)
print(summary_df)
print("\nGeriausias modelis:")
print(f"{best_row['model']} | AUC={best_row['roc_auc']:.4f} | F1={best_row['f1']:.4f}")
print("\nRezultatai issaugoti:")
print(RESULTS_DIR)
print("=" * 70)
