import subprocess
import os
import pandas as pd
from rdkit import Chem
from meeko import MoleculePreparation

# --- НАСТРОЙКИ ---
VINA_EXE_PATH = r"M:\Документы\Курсач\VAE SMILES gen\doking\vina_1.2.7_win.exe" # Укажите ваш путь
RECEPTOR_PDBQT_PATH = "M:/Документы/Курсач/VAE SMILES gen/doking/3kfa_ABL_native.pdbqt"
OUTPUT_DIR = "./docking_results_3oxz_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- НАСТРОЙКИ ДЛЯ ДОКИНГА (для 3OXZ)---
CENTER_X, CENTER_Y, CENTER_Z = 18, 8, 6 # Центр бокса
SIZE_X, SIZE_Y, SIZE_Z = 31, 23, 23 # Размер бокса в ангстремах
EXHAUSTIVENESS = 100

# --- ВАШИ SMILES ---
ligands_smiles = [
    # ('CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5', 'imatinib'),
    # ('CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5', 'nilotinib'),
    # ('CC1=C(C=C(C=C1)C(=O)NC2=CC(=C(C=C2)CN3CCN(CC3)C)C(F)(F)F)C#CC4=CN=C5N4N=CC=C5', 'ponatinib'),
    ("CC(C(O)c1cccc(Nc3ncc(F)c(Oc4ccccc4)n3)c1F)C2(CC2)C", "lig_1"),
    ("CC(C)Cc1ccc(NC(=O)c2ccnc(Nc3ccccc3C(F)(F)F)n2)cc1O", "lig_2"),
    ("c1ccc(Nc2nccc(C(F)(F)F)n2)cc1-c1cnn(C)c1-c1ccccc1F", "lig_3"),
    ("COC(=O)c1cccc(Nc2nccc(C(=O)Nc3cc(C#N)ccn3)n2)c1C#N", "lig_4"),
    ("CC(=O)Nc1ccc(Nc2ccnc(Nc3cc(NC(C)=O)ccc3OC)n2)cc1OC", "lig_5"),
    ("C=C(O)c1ccccc1NC(=O)c1ccnc(Nc2ccc(Cc3ccccc3)cc2)n1", "lig_6"),
    ("Cc1ccc(Nc2nccc(C(F)(F)F)n2)cc1N(C)C(=O)OC(C)(C)C#N", "lig_7"),
    ("N#CC1CCCC1n1c(=O)ccc2cnc(Nc3ccc(F)cc3)nc2N1C1CCCC1", "lig_8"),
    ("FC(F)(F)c1cccc(Nc2nccc(C(=O)Nc3ccc4ccccc4c3)n2)c1C", "lig_9"),
    ("COc1cc(C)cc(Nc2cc(C)nc(Nc3ccc4c(c3)NC(=O)CO4)n2)c1", "lig_10"),
    ("Cc1c(C)cc(Nc2nccc(-c3cccc4c3ccn4C)n2)cc1CN1CC(O)C1", "lig_11"),
    ("CCOC(=O)Nc1ccc(CNc2ccnc(Nc3ccc(N4CCCC4C)cc3)n2)cc1", "lig_12"),
    ("CS(=O)(=O)Nc1cccc(Nc2nccc(NC3CCCc4ccccc43)n2)c1C#N", "lig_13"),
    ("CC(C)C(=O)c1ccc(Nc2nccc(-c3cccc(OC)c3)n2)cc1NC(=O)", "lig_14"),
    ("C=Cc1ccc(Nc2ncc(C(=O)Nc3cccc4cccnc34)cn2)c(C)c1OCC", "lig_15"),
    ("C=CC(=O)Nc1ccnc(Nc2ccc(N3CCOCC3)nc2N2CCN(C)CC2)c1F", "lig_16"),
    ("CC(=O)c1cncc(N2CCc3cnc(Nc4ccc(F)cc4)nc32)c1C(=O)NC", "lig_17"),
    # ("CN1CCc2nc(ncc2CCc2ccccc2)nc1Nc1ccccc1N1CCN(C)CC1=O", "lig_18"),
    ("CCN(c1nccc(Oc2cccc3c2CCN3)n1)c1ccccc1CN(C)S(C)(=O)", "lig_19"),
    ("CCN(c1ncc(O)c(Nc4ccccc4S(N)(=O)=O)n1)C(c1ccccc1F)N", "lig_20"),
    ("CC(=O)Nc1ccc(Nc2ncc(C#N)c(-c3ccc4c(c3)OCO4)n2)cc1O", "lig_21"),
    ("C=CC(=O)c1cc(F)c(CNc2ccnc(N3CCCc4ccccc43)n2)c(F)c1", "lig_22"),
    ("C#CC(=O)NCCc1ccc(F)cc1Nc1nccc(Nc2ccc3[nH]ccc3c2)n1", "lig_23"),
    ("CNC(=O)c1ccccc1Cn1ccc2cnc(Nc3ccc(N4CCNCC4)cc3)nc21", "lig_24"),
    ("CC(C)Nc1ccc(Nc2nccc(NC3CCCCC3)n2)cc1N1CCN(C)CC1(C)", "lig_25"),
    ("CNCc1cnc(Nc2ccc(F)cc2)nc1N(C)c1ccnc(N2CCC(O)C2)c1C", "lig_26"),
    ("Cc1ccc(Nc2ncc(C(F)(F)F)c(Nc3ccc4[nH]ncc4c3)n2)cc1N", "lig_27"),
    ("CCc1cnc(Nc2cccc(N3CCN(C)CC3)c2)nc1Nc1ccccc1C(=O)OC", "lig_28"),
]

# --- ОСНОВНОЙ БЛОК ---
preparator = MoleculePreparation()

for smi, name in ligands_smiles:
    print(f"Обработка {name}...")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Ошибка: неверный SMILES для {name}")
        continue

    # 1. Готовим лиганд
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
    preparator.prepare(mol)
    lig_pdbqt_string = preparator.write_pdbqt_string()
    
    # 2. Сохраняем PDBQT файл для лиганда
    lig_path = os.path.join(OUTPUT_DIR, f"{name}_ligand.pdbqt")
    with open(lig_path, 'w') as f:
        f.write(lig_pdbqt_string)

    # 3. Формируем команду для Vina
    out_path = os.path.join(OUTPUT_DIR, f"{name}_out.pdbqt")
    
    cmd = [
        VINA_EXE_PATH,
        '--receptor', RECEPTOR_PDBQT_PATH,
        '--ligand', lig_path,
        '--center_x', str(CENTER_X), '--center_y', str(CENTER_Y), '--center_z', str(CENTER_Z),
        '--size_x', str(SIZE_X), '--size_y', str(SIZE_Y), '--size_z', str(SIZE_Z),
        '--exhaustiveness', str(EXHAUSTIVENESS),
        '--num_modes', '9',
        '--energy_range', '3',
        '--out', out_path,
    ]

    # 4. Запускаем докинг
    print(f"Запуск докинга для {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Докинг для {name} завершен. Результат: {out_path}")
    else:
        print(f"Ошибка при докинге {name}:\n{result.stderr}")