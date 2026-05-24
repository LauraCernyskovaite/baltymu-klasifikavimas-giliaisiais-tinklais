"""
Fermentų / baltymų be EC klasifikavimas naudojant AlphaFold 3D struktūras.

Skriptas papildo esamą sekos + biocheminių požymių eksperimentą:
1) iš AlphaFold PDB failų paima C-alpha atomų koordinates;
2) sugeneruoja 2D atstumų matricas;
3) apmoko CNN modelį, kuris matricas traktuoja kaip vaizdus;
4) palygina:
   - Bio MLP pagal išplėstus biocheminius požymius;
   - Distance-map CNN pagal AlphaFold struktūrą;
   - Fusion modelį: distance-map CNN + Bio MLP.

Naudoja:
  dataset_cluster_split_with_alphafold_status.csv
  alphafold_structures/*.pdb

Svarbu: AlphaFold struktūros yra prognozuotos, todėl rezultatuose jas reikia
vadinti "AlphaFold prognozuotomis struktūromis", ne eksperimentinėmis PDB.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import json
import random
import re
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, optimizers, regularizers

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


'''
1) NUSTATYMAI

Konfigūracijos blokas - visi svarbūs parametrai vienoje vietoje.

MAP_SIZE=128 - visos atstumų matricos pakeičiamos į vienodą
128×128 dydį, nes CNN įvestis turi būti fiksuotos formos.

DISTANCE_CLIP=32.0 - atstumai didesni nei 32 Å apkarpomi.
Po apkarpymo reikšmės normalizuojamos ir invertuojamos, kad
artimi C-alpha atomų kontaktai matricoje būtų ryškesni.

MAX_STRUCTURES=None - naudojamos visos turimos struktūros.
'''

SEED = 42

DATA_DIR = Path(".")
DATASET_FILE = DATA_DIR / "dataset_cluster_split_with_alphafold_status.csv"
STRUCTURE_DIR = DATA_DIR / "alphafold_structures"

RESULTS_DIR = Path("results_alphafold_distance_maps2")
MAP_SIZE = 128
DISTANCE_CLIP = 32.0

BATCH = 32
EPOCHS = 35
LR = 3e-4

MAX_STRUCTURES = None

for sub in ["images", "models", "reports", "objects", "maps_preview"]:
    (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

'''
2) SEKOS VALYMAS IR BIOCHEMINIAI POŽYMIAI

Šiame eksperimente naudojamas 57 požymių rinkinys:
54 požymiai apskaičiuojami iš aminorūgščių sekos, o 3 požymiai
gaunami iš AlphaFold struktūros failo.

Papildomai įtraukiami sieros turinčių aminorūgščių, pasirinktų
dipeptidų, aminorūgščių įvairovės ir elektrostatinio profilio
požymiai.

Iš AlphaFold struktūros failo naudojami pLDDT vidurkis,
pLDDT standartinis nuokrypis ir Cα atomų skaičius. Šie požymiai
apibūdina struktūros modelio pasitikėjimą ir turimos struktūrinės
informacijos kiekį.
'''

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)

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
    "sulfur": set("CM"),
}

DIPEPTIDES = [
    "GG", "GP", "PG", "PP", "CC",
    "DE", "ED", "KR", "RK", "ST",
]


def clean_aa(seq: str) -> str:
    seq = re.sub(r"[^A-Za-z]", "", str(seq)).upper()
    seq = (
        seq.replace("U", "C")
        .replace("O", "K")
        .replace("B", "D")
        .replace("Z", "E")
        .replace("J", "I")
    )
    return "".join(ch if ch in AA_SET else "" for ch in seq)


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
        "cysteine_fraction",
        "cysteine_density",
        "positive_negative_ratio",
        "charge_balance",
        "aa_entropy",
    ]
    + [f"{g}_fraction" for g in GROUPS]
    + [f"aa_{aa}_fraction" for aa in AA]
    + [f"dipeptide_{dp}_fraction" for dp in DIPEPTIDES]
)


def sequence_entropy(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    probs = np.array([seq.count(aa) / n for aa in AA], dtype=np.float32)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def calculate_bio_features(seq: str) -> dict:
    clean = clean_aa(seq)
    n = len(clean)
    if n < 5:
        return {k: 0.0 for k in BIO_FEATURES}

    feats = {
        "length": float(n),
        "cysteine_fraction": clean.count("C") / n,
        "cysteine_density": (clean.count("C") / 2.0) / max(n / 100.0, 1.0),
        "aa_entropy": sequence_entropy(clean),
    }

    pos = sum(clean.count(aa) for aa in GROUPS["positive"])
    neg = sum(clean.count(aa) for aa in GROUPS["negative"])
    feats["positive_negative_ratio"] = float(pos / max(neg, 1))
    feats["charge_balance"] = float((pos - neg) / n)

    for grp, members in GROUPS.items():
        feats[f"{grp}_fraction"] = sum(aa in members for aa in clean) / n

    for aa in AA:
        feats[f"aa_{aa}_fraction"] = clean.count(aa) / n

    denom = max(n - 1, 1)
    for dp in DIPEPTIDES:
        feats[f"dipeptide_{dp}_fraction"] = clean.count(dp) / denom

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

    return {k: float(feats.get(k, 0.0)) for k in BIO_FEATURES}


'''
3) PDB -> 2D ATSTUMŲ MATRICA

Šiame eksperimente AlphaFold prognozuota 3D struktūra paverčiama
2D atstumų matrica, kuri naudojama kaip CNN įvestis.

resolve_structure_path() suranda PDB failą galimuose kataloguose.

parse_ca_coordinates_from_pdb() iš PDB failo paima Cα atomų
koordinates ir pLDDT reikšmes. Cα atomai naudojami todėl, kad jie
apibūdina aminorūgščių pagrindinės grandinės padėtį ir sumažina
duomenų kiekį.

make_distance_image() apskaičiuoja visų Cα atomų porų atstumus,
juos apkarpo iki DISTANCE_CLIP, normalizuoja, invertuoja ir pakeičia
matricos dydį į 128×128. Taip gaunamas vienodo dydžio vaizdas,
tinkamas konvoliuciniam neuroniniam tinklui.

Pirmieji keli sugeneruoti vaizdai išsaugomi maps_preview kataloge
vizualinei patikrai.
'''

def resolve_structure_path(structure_file: str) -> Path | None:
    if not isinstance(structure_file, str) or not structure_file.strip():
        return None

    raw = Path(structure_file)
    candidates = [
        raw,
        DATA_DIR / raw,
        STRUCTURE_DIR / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_ca_coordinates_from_pdb(pdb_path: Path):
    coords = []
    plddt_values = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b_factor = float(line[60:66])  # AlphaFold: pLDDT balas
            except ValueError:
                continue
            coords.append([x, y, z])
            plddt_values.append(b_factor)

    if len(coords) < 5:
        return None, None

    return np.asarray(coords, dtype=np.float32), np.asarray(plddt_values, dtype=np.float32)


def make_distance_image(coords: np.ndarray, size: int = MAP_SIZE) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    # Apkarpymas ir normalizavimas: [0, CLIP] => [0, 1]
    dist = np.clip(dist, 0.0, DISTANCE_CLIP) / DISTANCE_CLIP

    # Inversija: artimos liekanos tampa šviesesnės (0 atstumas = 1.0).
    # Lokalių kontaktų sritys tampa ryškesnės vaizde.
    image = 1.0 - dist
    image = image[..., None].astype(np.float32)

    image = tf.image.resize(image, (size, size), method="bilinear").numpy()
    return image.astype(np.float32)


def load_dataset_with_structures() -> pd.DataFrame:
    df = pd.read_csv(DATASET_FILE)
    needed = {"sequence", "label", "split", "structure_downloaded", "structure_file"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Truksta stulpeliu: {sorted(missing)}")

    df = df[df["label"].isin([0, 1])].copy()
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    # Paliekami tik tie įrašai, kuriems AlphaFold struktūra atsisiųsta.
    df = df[df["structure_downloaded"].astype(str).str.lower().eq("true")].copy()
    df["resolved_structure_file"] = df["structure_file"].apply(resolve_structure_path)
    df = df[df["resolved_structure_file"].notna()].reset_index(drop=True)

    if MAX_STRUCTURES is not None and len(df) > MAX_STRUCTURES:
        df = (
            df.groupby(["split", "label"], group_keys=False)
            .apply(lambda x: x.sample(
                n=max(1, int(MAX_STRUCTURES * len(x) / len(df))),
                random_state=SEED,
            ))
            .reset_index(drop=True)
        )

    return df

def build_arrays(df: pd.DataFrame):
    '''
    Apdorojami visi įrašai su PDB failais. Kiekvienam įrašui:
        1. Nuskaitomos Cα koordinatės ir pLDDT reikšmės iš PDB failo
        2. Sugeneruojama 128×128 atstumų matrica kaip 2D vaizdas
        3. Apskaičiuojami biocheminiai požymiai
        4. Požymiai papildomi pLDDT statistikomis ir Cα atomų skaičiumi

    Pirmieji 6 vaizdai išsaugomi PNG formatu vizualinei patikrai.
    Progreso žinutė išvedama kas 500 įrašų.
    '''
    bio_rows = []
    maps = []
    labels = []
    splits = []
    accessions = []
    plddt_mean = []
    used_rows = []

    for i, row in df.iterrows():
        coords, plddt = parse_ca_coordinates_from_pdb(Path(row["resolved_structure_file"]))
        if coords is None:
            continue

        image = make_distance_image(coords)
        bio = calculate_bio_features(row["sequence"])

        bio["alphafold_plddt_mean"] = float(np.mean(plddt))
        bio["alphafold_plddt_std"] = float(np.std(plddt))
        bio["ca_atom_count"] = float(len(coords))

        maps.append(image)
        bio_rows.append(bio)
        labels.append(int(row["label"]))
        splits.append(row["split"])
        accessions.append(str(row.get("accession", "")))
        plddt_mean.append(float(np.mean(plddt)))
        used_rows.append(i)

        # Pirmųjų 6 matricų PNG išsaugojimas vizualinei patikrai.
        if len(maps) <= 6:
            out = RESULTS_DIR / "maps_preview" / f"{row.get('accession', i)}_distance_map.png"
            plt.figure(figsize=(4, 4))
            plt.imshow(image.squeeze(), cmap="viridis", vmin=0, vmax=1)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out, dpi=120)
            plt.close()

        if (len(maps) % 500) == 0:
            print(f"Sugeneruota atstumu matricu: {len(maps)}")

    bio_feature_names = BIO_FEATURES + [
        "alphafold_plddt_mean",
        "alphafold_plddt_std",
        "ca_atom_count",
    ]

    X_map = np.asarray(maps, dtype=np.float32)
    X_bio_raw = pd.DataFrame(bio_rows, columns=bio_feature_names).to_numpy(dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    split = np.asarray(splits)

    meta = pd.DataFrame({
        "source_row": used_rows,
        "accession": accessions,
        "split": splits,
        "label": labels,
        "plddt_mean": plddt_mean,
    })

    return X_map, X_bio_raw, y, split, meta, bio_feature_names


'''
4) MODELIŲ ARCHITEKTŪROS

Naudojami trys modeliai, leidžiantys įvertinti skirtingų
informacijos šaltinių indėlį.

build_bio_mlp()
  MLP modelis naudoja 57 biocheminius požymius.

distance_cnn_branch()
  2D CNN šaka apdoroja atstumų matricą. Konvoliuciniai sluoksniai
  mokosi lokalių struktūrinių dėsningumų ir kontaktų šablonų.
  GlobalAveragePooling2D visą matricą paverčia požymių vektoriumi.
  Adam optimizatoriuje naudojamas clipnorm=1.0, kad mokymas būtų
  stabilesnis.

build_fusion_model()
  Dviejų šakų modelis sujungia atstumų matricos CNN šaką ir
  biocheminių požymių MLP šaką. Taip tikrinama, ar struktūrinė
  ir biocheminė informacija papildo viena kitą.
'''

def build_bio_mlp(n_features: int):
    inp = layers.Input(shape=(n_features,), name="bio_input")
    x = layers.Dense(160, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(80, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="Bio_MLP_extended")
    compile_model(model)
    return model


def distance_cnn_branch(map_input):
    x = layers.Conv2D(24, 5, padding="same", activation="relu")(map_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    return x


def build_distance_cnn():
    inp = layers.Input(shape=(MAP_SIZE, MAP_SIZE, 1), name="distance_map_input")
    x = distance_cnn_branch(inp)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="AlphaFold_DistanceMap_CNN")
    compile_model(model)
    return model


def build_fusion_model(n_features: int):
    # Pirmoji šaka apdoroja 128×128 atstumų matricą.
    map_inp = layers.Input(shape=(MAP_SIZE, MAP_SIZE, 1), name="distance_map_input")
    map_x = distance_cnn_branch(map_inp)

    # Antroji šaka apdoroja išplėstą biocheminių požymių vektorių.
    bio_inp = layers.Input(shape=(n_features,), name="bio_input")
    bio_x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(bio_inp)
    bio_x = layers.BatchNormalization()(bio_x)
    bio_x = layers.Dropout(0.30)(bio_x)
    bio_x = layers.Dense(48, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(bio_x)
    bio_x = layers.Dropout(0.20)(bio_x)

    # Šakų išvestys sujungiamos į bendrą požymių vektorių.
    x = layers.Concatenate()([map_x, bio_x])
    x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model([map_inp, bio_inp], out, name="Fusion_DistanceMap_BioMLP")
    compile_model(model)
    return model


def compile_model(model):
    # clipnorm=1.0 - gradientų apkarpymas: jei gradiento norma
    # viršija 1.0, jis proporcingai sumažinamas. Stabilizuoja
    # mokymą su 2D CNN, kur gradientai gali išaugti.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


def get_callbacks(model_name: str):
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
            filepath=str(RESULTS_DIR / "models" / f"{model_name}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.CSVLogger(str(RESULTS_DIR / "reports" / f"{model_name}_training_log.csv")),
    ]

'''
5) VERTINIMAS IR VIZUALIZACIJOS

Šiame bloke apibrėžiamos bendros modelių vertinimo ir rezultatų
vizualizavimo funkcijos.

find_best_threshold()
  Iš validacijos rinkinio parenka F1 rodiklį maksimizuojantį
  slenkstį. Testavimo rinkinys slenksčiui parinkti nenaudojamas.

evaluate_model()
  Apskaičiuoja pagrindines metrikas: ROC-AUC, Accuracy, F1, MCC
  ir sumaišymo matricą.

save_reports()
  Išsaugo modelių rezultatų santrauką, ROC kreives, klaidų
  matricas ir testo prognozių lentelę.
'''

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
        "confusion_matrix": confusion_matrix(y_true, preds),
        "probs": probs,
        "preds": preds,
    }


def plot_history(history, name):
    hist = history.history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.title(f"{name}: Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("auc", []), label="auc")
    plt.plot(hist.get("val_auc", []), label="val_auc")
    plt.xlabel("Epocha")
    plt.ylabel("AUC")
    plt.title(f"{name}: AUC")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "images" / f"history_{name}.png", dpi=150)
    plt.close()


def save_reports(results, y_test, meta_test, feature_names, scaler, bio_model, X_bio_test):
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
    summary_df.to_csv(RESULTS_DIR / "reports" / "model_metrics.csv", index=False)

    plt.figure(figsize=(9, 7))
    for key, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        plt.plot(fpr, tpr, label=f"{res['model']} AUC={res['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC kreives: AlphaFold atstumu matricos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "images" / "roc_curves.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (_, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=res["confusion_matrix"],
            display_labels=["Be EC", "Su EC"],
        ).plot(ax=ax, colorbar=False)
        ax.set_title(res["model"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "images" / "confusion_matrices.png", dpi=150)
    plt.close()

    out = meta_test.copy().reset_index(drop=True)
    out["true_label"] = y_test
    for key, res in results.items():
        out[f"prob_{key}"] = res["probs"]
        out[f"pred_{key}"] = res["preds"]
        out[f"error_{key}"] = out[f"pred_{key}"] != out["true_label"]
    out.to_csv(RESULTS_DIR / "reports" / "test_predictions.csv", index=False)

    try:
        class BioWrapper(ClassifierMixin, BaseEstimator):
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                p = self.model.predict(X, verbose=0).ravel()
                return np.vstack([1.0 - p, p]).T

        perm = permutation_importance(
            BioWrapper(bio_model),
            X_bio_test,
            y_test,
            scoring="roc_auc",
            n_repeats=8,
            random_state=SEED,
            n_jobs=1,
        )
        perm_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(RESULTS_DIR / "reports" / "bio_mlp_permutation_importance.csv", index=False)
    except Exception as exc:
        print(f"Permutacines svarbos nepavyko apskaiciuoti: {exc}")

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_file": str(DATASET_FILE),
            "structure_dir": str(STRUCTURE_DIR),
            "map_size": MAP_SIZE,
            "distance_clip_angstrom": DISTANCE_CLIP,
            "alphafold_used": True,
            "structure_representation": "C-alpha distance matrix resized to fixed 2D image",
            "note": "AlphaFold structures are predicted structures, not experimental PDB structures.",
            "biopython_available": BIOPYTHON_AVAILABLE,
            "seed": SEED,
            "batch": BATCH,
            "epochs": EPOCHS,
            "learning_rate": LR,
        },
        "results": summary_df.to_dict("records"),
        "best_model": summary_df.iloc[0].to_dict(),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }

    with open(RESULTS_DIR / "reports" / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    joblib.dump(scaler, RESULTS_DIR / "objects" / "bio_scaler.joblib")

    print("\nModeliu palyginimas:")
    print(summary_df)
    print(f"\nRezultatai issaugoti: {RESULTS_DIR}")


'''
6) PAGRINDINIS PALEIDIMAS (main)

Pagrindinė eksperimento eiga:

1. Įkeliami tik tie įrašai, kurie turi AlphaFold struktūras.
2. Iš PDB failų sugeneruojamos 128×128 atstumų matricos ir
   apskaičiuojami biocheminiai požymiai.
3. Duomenys padalijami į train, val ir test rinkinius pagal
   iš anksto atliktą MMseqs2 klasterinį padalinimą.
4. StandardScaler pritaikomas tik treniravimo rinkiniui, o
   validacijos ir testavimo rinkiniai tik transformuojami.
5. Nuosekliai treniruojami trys modeliai:
   a) Bio MLP
   b) Distance-map CNN
   c) Fusion modelis
6. Rezultatai išsaugomi ir palyginami.
'''

def main():
    print("=" * 72)
    print("ALPHAFOLD 3D STRUKTURU -> 2D ATSTUMU MATRICU EKSPERIMENTAS")
    print("=" * 72)
    print(f"Dataset: {DATASET_FILE}")
    print(f"Strukturos: {STRUCTURE_DIR}")
    print(f"MAP_SIZE={MAP_SIZE}, EPOCHS={EPOCHS}, BATCH={BATCH}")
    print("=" * 72)

    df = load_dataset_with_structures()
    print(f"Irasu su rastais PDB failais: {len(df)}")
    print(df.groupby(["split", "label"]).size())

    X_map, X_bio_raw, y, split, meta, feature_names = build_arrays(df)
    print(f"\nNaudojama po PDB nuskaitymo: {len(y)}")
    print(pd.Series(split).value_counts())

    train_idx = split == "train"
    val_idx = split == "val"
    test_idx = split == "test"
 
    scaler = StandardScaler()
    X_bio_train = scaler.fit_transform(X_bio_raw[train_idx])
    X_bio_val = scaler.transform(X_bio_raw[val_idx])
    X_bio_test = scaler.transform(X_bio_raw[test_idx])

    X_map_train, X_map_val, X_map_test = X_map[train_idx], X_map[val_idx], X_map[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    class_weights = get_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    results = {}

    print("\n1 ETAPAS: Bio MLP su isplestais biocheminiais pozymiais")
    bio_model = build_bio_mlp(X_bio_train.shape[1])
    hist_bio = bio_model.fit(
        X_bio_train,
        y_train,
        validation_data=(X_bio_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weights,
        callbacks=get_callbacks("bio_mlp_extended"),
        verbose=1,
    )
    plot_history(hist_bio, "bio_mlp_extended")
    val_probs = bio_model.predict(X_bio_val, verbose=0).ravel()
    test_probs = bio_model.predict(X_bio_test, verbose=0).ravel()
    thr = find_best_threshold(y_val, val_probs)
    results["bio_mlp_extended"] = evaluate_model("Bio MLP expanded", y_test, test_probs, thr)

    print("\n2 ETAPAS: AlphaFold atstumu matricos CNN")
    dist_model = build_distance_cnn()
    hist_dist = dist_model.fit(
        X_map_train,
        y_train,
        validation_data=(X_map_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weights,
        callbacks=get_callbacks("distance_map_cnn"),
        verbose=1,
    )
    plot_history(hist_dist, "distance_map_cnn")
    val_probs = dist_model.predict(X_map_val, verbose=0).ravel()
    test_probs = dist_model.predict(X_map_test, verbose=0).ravel()
    thr = find_best_threshold(y_val, val_probs)
    results["distance_map_cnn"] = evaluate_model("AlphaFold distance-map CNN", y_test, test_probs, thr)

    print("\n3 ETAPAS: Fusion modelis, AlphaFold atstumu matrica + bio pozymiai")
    fusion_model = build_fusion_model(X_bio_train.shape[1])
    hist_fusion = fusion_model.fit(
        [X_map_train, X_bio_train],
        y_train,
        validation_data=([X_map_val, X_bio_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weights,
        callbacks=get_callbacks("fusion_distance_bio"),
        verbose=1,
    )
    plot_history(hist_fusion, "fusion_distance_bio")
    val_probs = fusion_model.predict([X_map_val, X_bio_val], verbose=0).ravel()
    test_probs = fusion_model.predict([X_map_test, X_bio_test], verbose=0).ravel()
    thr = find_best_threshold(y_val, val_probs)
    results["fusion_distance_bio"] = evaluate_model("Fusion: distance-map + Bio MLP", y_test, test_probs, thr)

    save_reports(
        results=results,
        y_test=y_test,
        meta_test=meta[test_idx],
        feature_names=feature_names,
        scaler=scaler,
        bio_model=bio_model,
        X_bio_test=X_bio_test,
    )

if __name__ == "__main__":
    main()