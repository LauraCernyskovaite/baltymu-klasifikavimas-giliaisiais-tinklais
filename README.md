# Baltymų klasifikavimas naudojant giliuosius neuroninius tinklus

Šiame projekte pateikiamas bakalauro darbo praktinės dalies kodas. Modeliai klasifikuoja baltymus į dvi klases:

- `label = 1` – baltymai su EC numeriu;
- `label = 0` – baltymai be EC numerio po papildomo anotacijų filtravimo.

## Pagrindiniai failai

- `train_enzyme_classifier_cluster_split.py` – pagrindinis eksperimentas su aminorūgščių sekomis ir biocheminiais požymiais.
- `train_alphafold_distance_map_experiment.py` – papildomas eksperimentas su AlphaFold struktūromis ir C-alpha atomų atstumų matricomis.
- `train.csv`, `val.csv`, `test.csv` – MMseqs2 klasterinio padalijimo duomenų aibės.
- `dataset_cluster_split_with_alphafold_status.csv` – duomenų lentelė su AlphaFold struktūrų būsena.

## Pagrindinis eksperimentas

Paleidimas:

```bash
python train_enzyme_classifier_cluster_split.py
```

Šis scenarijus apmoko ir palygina:

- CNN modelį pagal aminorūgščių seką;
- MLP modelį pagal 29 biocheminius požymius;
- multimodalų CNN ir MLP modelį;
- Random Forest bazinį modelį;
- papildomus modelius be sekos ilgio ir molekulinės masės.

Rezultatai išsaugomi aplanke:

```text
results_cluster_split_1024/
```

Svarbiausia rezultatų lentelė:

```text
results_cluster_split_1024/reports/model_metrics.csv
```

## AlphaFold struktūrinis eksperimentas

Paleidimas:

```bash
python train_alphafold_distance_map_experiment.py
```

Šis scenarijus:

- nuskaito AlphaFold PDB failus;
- atrenka C-alpha atomų koordinates;
- sudaro 128 x 128 atstumų matricas;
- apmoko atstumų matricomis grįstą CNN modelį;
- apmoko išplėstą Bio MLP modelį;
- apmoko sujungtą atstumų matricos ir biocheminių požymių modelį.

Rezultatai išsaugomi aplanke:

```text
results_alphafold_distance_maps2/
```

Svarbiausia rezultatų lentelė:

```text
results_alphafold_distance_maps2/reports/model_metrics.csv
```

## Pastabos dėl duomenų

AlphaFold PDB struktūros ir apmokyti modeliai gali užimti daug vietos, todėl jie nėra būtini kelti į GitHub. Jei struktūrų aplankas neįkeliamas, jį reikia atkurti iš AlphaFold DB pagal `dataset_cluster_split_with_alphafold_status.csv` faile pateiktą informaciją.

## Aplinkos paruošimas

Rekomenduojama naudoti Python 3.10 arba naujesnę versiją.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux arba Google Colab aplinkoje MMseqs2 diegiamas atskirai:

```bash
apt-get update
apt-get install -y mmseqs2
```

