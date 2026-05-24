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

## AlphaFold eksperimentas

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

## Pastabos dėl duomenų

Duomenų atsisiuntimą AlphaFold eksperimentui galite rasti: 
https://colab.research.google.com/drive/1mVwLyr05K-6FsnXtUsAYCjTPZLKcTP6_?usp=sharing

## Eksperimentų paleidimas

Repozitoriją galima atsisiųsti iš GitHub svetainės paspaudus Code ir pasirinkus Download ZIP.
Atsisiųstą ZIP failą reikia išskleisti pasirinktame aplanke ir atidaryti su Visual Studio Code, Jupyter Notebook arba kita Python kodo vykdymui tinkama aplinka.

Pagrindinis eksperimentas, kuriame apmokomi ir palyginami CNN, MLP, multimodalus CNN ir MLP bei Random Forest modeliai, paleidžiamas komanda:

```bash
python train_enzyme_classifier_cluster_split.py
```

Linux ir macOS sistemose gali reikėti naudoti:

```bash
python3 train_enzyme_classifier_cluster_split.py
```

Papildomas AlphaFold struktūrinis eksperimentas paleidžiamas komanda:

```bash
python train_alphafold_distance_map_experiment.py
```

Linux ir macOS sistemose gali reikėti naudoti:

```bash
python3 train_alphafold_distance_map_experiment.py
```
