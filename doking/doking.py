import subprocess
import os
import pandas as pd
from rdkit import Chem
from meeko import MoleculePreparation

# --- НАСТРОЙКИ ---
VINA_EXE_PATH = r"M:\Документы\Курсач\VAE SMILES gen\doking\vina_1.2.7_win.exe" # Укажите ваш путь
RECEPTOR_PDBQT_PATH = "M:/Документы/Курсач/VAE SMILES gen/doking/3oxz.pdbqt"
OUTPUT_DIR = "./docking_results_3oxz_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- НАСТРОЙКИ ДЛЯ ДОКИНГА (для 3OXZ)---
CENTER_X, CENTER_Y, CENTER_Z = 26.5, 25.0, 19.5 # Центр бокса
SIZE_X, SIZE_Y, SIZE_Z = 20, 20, 20 # Размер бокса в ангстремах
EXHAUSTIVENESS = 8

# --- ВАШИ SMILES ---
ligands_smiles = [
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
    # ('CN1CCc2ccc(NC(=O)c3ccc(Nc4nc(-c5ccccc5)c5ccccc5n4)cc3)cc2C1', 'max_energy_from_dataset')
    # ('CCn1c(=O)ccc2c(C)nc(Nc3ccc(NC(C)=O)cc3)nc21', 'min_energy_from_dataset')
#     ('Nc1cccc(Nc2nccc(-c3cccc(NC(=O)c4ccccn4)c3)n2)c1', 'dataset-1'),
# ('Cc1ccc(Nc2ncc(F)c(-n3cc(C(N)=O)c4ccccc43)n2)cc1N1CCN(C)CC1', 'dataset-2'),
# ('Cc1cc(C)nc(Nc2cccc(C(=O)N3CCN(c4ccc(-c5ccccc5)nn4)CC3)c2)n1', 'dataset-3'),
# ('Cc1nnnn1-c1cc(Nc2ncc(F)c(NCC3CCCN4CCCCC34)n2)ccc1F', 'dataset-4'),
# ('O=C(Nc1ccc(CN2CCCC2=O)cc1)c1ccc(Nc2ncccn2)cc1', 'dataset-5'),
# ('Cc1cc(Nc2nccc(C(F)(F)F)n2)cc(-c2cnn(CC3COC(=O)N3)c2)c1', 'dataset-6'),
# ('CC(=O)Nc1ccc(Nc2nc3c(c(Nc4ccccc4S(=O)(=O)NC(C)C)n2)CCN3)cc1', 'dataset-7'),
# ('Cc1cnc(Nc2ccc(C(=O)O)cc2)nc1-c1ccc(C#N)cc1', 'dataset-8'),
# ('CC(C)Oc1ccccc1Nc1ccnc(Nc2cccc3cccnc23)n1', 'dataset-9'),
# ('Cc1cc(C)c(Nc2ncc(C(=O)Nc3ccccc3)cn2)c(C)c1', 'dataset-10'),
# ('CCN(c1cccc(C)c1)c1ccnc(Nc2ccc(C#N)cc2)n1', 'dataset-11'),
# ('Cc1ccc(Nc2nccc(-c3c(C(F)(F)F)nc4ccccn34)n2)cc1', 'dataset-12'),
# ('CC(C)c1c(O)ccc2cnc(Nc3ccc(F)c(C(F)(F)F)c3)nc12', 'dataset-13'),
# ('c1cc(N2CCNCC2)ccc1Nc1nc(OCC2CCCCC2)c2cn[nH]c2n1', 'dataset-14'),
# ('COc1cc(C(=O)NC2CCN(C)CC2)ccc1Nc1ncc2c(n1)N(C1CCCC1)C1CCCC1C(=O)N2C', 'dataset-15'),
# ('CC(C)(C)c1ccc(NC(=O)c2cnc(Nc3ccccc3C#N)nc2)cc1', 'dataset-16'),
# ('CC(C)(C)Oc1cccc(Nc2nc(Nc3ccc(F)cc3)ncc2C(F)(F)F)c1', 'dataset-17'),
# ('CC(C)(C)c1ccc(Nc2nccc(-n3ccnc3-c3ccccc3)n2)cc1', 'dataset-18'),
# ('NC(=O)c1cccc(Nc2nccc(Nc3cccc(F)c3)n2)c1', 'dataset-19'),
# ('CS(=O)(=O)N(CCO)c1ccccc1Cn1ccc2cnc(Nc3ccc4c(c3)CC(=O)N4)nc21', 'dataset-20'),
# ('O=C(NCCO)c1cccc(Nc2ncc3nnn(-c4cccc(CNC(=O)C5CC5)c4)c3n2)c1', 'dataset-21'),
# ('C=CC(=O)N1CCCC(n2cnc3c(NCc4ccccc4)nc(Nc4ccc(N5CCCCC5)cc4)nc32)C1', 'dataset-22'),
# ('CC(=O)c1cccc(Nc2nc(Nc3ccccc3)cc(-c3ccccc3)n2)c1', 'dataset-23'),
# ('CN1CCCC(c2ccc(Nc3ncc4cc(C5CC5)c(=O)n(Cc5ccsc5C5CCNC5)c4n3)cc2)C1', 'dataset-24'),
# ('Cc1nc2ccccc2n1-c1nc(NCc2ccccc2)c2[nH]cnc2n1', 'dataset-25'),
# ('O=S(=O)(c1ccc(Nc2ncc(C=Cc3cccc(O)c3)cn2)cc1)C1CCNCC1', 'dataset-26'),
# ('COc1ccc(Nc2nc(NCc3ccccc3N(C)S(C)(=O)=O)c3cc[nH]c3n2)cc1', 'dataset-27'),
# ('CC(C)(C)NC(=O)c1cc(Nc2ncc3cccc(OC4CCNCC4)c3n2)cc2[nH]ncc12', 'dataset-28'),
# ('CCc1cc(N2CCN(C)CC2)ccc1Nc1ncc(C(F)(F)F)c(NCCCNC(=O)N2CCC2)n1', 'dataset-29'),
# ('C=CC(=O)N1CCCC(Nc2nc(Nc3cccc(NC(=O)N4CCCC4)c3)nc3[nH]ccc23)C1', 'dataset-30'),
# ('CCS(=O)(=O)Nc1cccc(CNc2nc(Nc3ccc(C(C)O)cc3)ncc2C(F)(F)F)c1', 'dataset-31'),
# ('O=C(O)c1ccc(-c2ccnc(Nc3cccc(CNCCO)c3)n2)s1', 'dataset-32'),
# ('COc1ccccc1Nc1nc(N2CCc3ccccc32)ncc1F', 'dataset-33'),
# ('COc1ccc(C)c(Nc2ncc(C(F)(F)F)c(Nc3ccc4c(c3)OCO4)n2)c1', 'dataset-34'),
# ('CC(O)(c1cccc(-n2c3nc(Nc4ccc5c(c4)CCNC5)ncc3c(=O)n2C2CC2)n1)C(F)(F)F', 'dataset-35'),
# ('Cc1cc(C)c(Nc2nccc(C(=O)Nc3ccc(N(C)C)cc3)n2)c(C)c1', 'dataset-36'),
# ('Cc1cc(Nc2cccc(F)c2)nc(Nc2cccc(C)c2C)n1', 'dataset-37'),
# ('O=c1ccc2cnc(Nc3ccc(N4CCNCC4)cc3)nc2n1Cc1ccc2c(c1)OCO2', 'dataset-38'),
# ('CCN(Cc1ccccc1)c1ccnc(Nc2ccc(F)c(F)c2F)n1', 'dataset-39'),
# ('CCc1cccc(-c2nc(Nc3ccc(N4CCC(C)CC4)c(F)c3)nc3[nH]ccc23)c1', 'dataset-40'),
# ('C=CC(=O)Nc1cccc(-c2c(C)ccc3cnc(Nc4ccc(N5CCN(C(C)=O)CC5)cc4)nc23)c1', 'dataset-41'),
# ('Cc1cc(Nc2ccc(C#N)cc2)nc(Nc2cccc(C#N)c2)n1', 'dataset-42'),
# ('COC(=O)c1cccc(Nc2nccc(Nc3ccc(C(C)=O)cc3)n2)c1', 'dataset-43'),
# ('Cc1ccc(Nc2ncc(C(F)(F)F)c(N(C)C3CCCN4CCCCC34)n2)cc1C#N', 'dataset-44'),
# ('CCc1ccc2nc(Nc3cccc(Nc4nc(C)cc(C)n4)c3)sc2c1', 'dataset-45'),
# ('COc1ccc(C(=CC(C)C)C(N)=O)cc1Nc1ncc(F)c(-c2ccccc2)n1', 'dataset-46'),
# ('CN(C)S(=O)(=O)Cc1cccc(Nc2nccc(-c3ccc(OC4CCOCC4)c(C#N)c3)n2)c1', 'dataset-47'),
# ('COc1ccc(CNc2nc(Nc3ccccc3C)nc3c2ncn3C(C)C)cc1', 'dataset-48'),
# ('CON(C)C(=O)C1CCN(Sc2ccc(Nc3nccc(Nc4ccc(F)c(F)c4)n3)cc2)CC1', 'dataset-49'),
# ('C=CC(=O)N(C)c1cccc(-n2cnc3c(N)nc(Nc4ccc(F)cc4)nc32)c1', 'dataset-50'),
# ('Cc1cc(Nc2nccc(C(F)(F)F)n2)cc(C2=C(F)N(F)C(Cc3ccc(C(=O)O)cc3)S2)c1', 'dataset-51'),
# ('N#CCC(C1CCN(Cc2ccccc2)CC1)n1cc(-c2ccnc(Nc3ccc(N4CCOCC4)cc3)n2)cn1', 'dataset-52'),
# ('C=CC(=O)Nc1cccc(Nc2nc(Nc3cccc4c3CCN4CCC)ncc2OC)c1', 'dataset-53'),
# ('Cc1cc(Nc2nccc(-c3ccccn3)n2)cc2cc(C(=O)C(C)C)[nH]c12', 'dataset-54'),
# ('CC1=c2ccc(N(C)c3ccnc(Nc4ccc(C)c(S(N)(=O)=O)c4)n3)cc2=NC1C', 'dataset-55'),
# ('CCN1C(=O)C(C)(C)Oc2ccc(Nc3nc(Nc4ccc5c(c4)C4NC(=O)OC4C5)ncc3C)cc21', 'dataset-56'),
# ('Oc1cccc(Nc2nc(Nc3ccc4c(c3)OC=CN4)ncc2F)c1', 'dataset-57'),
# ('CCc1ccccc1Nc1nc(Nc2ccc(C#N)cc2)ncc1C(F)(F)F', 'dataset-58'),
# ('Cc1cc2c(C(N)=O)cccc2n1-c1nc2c(c(NCc3cccc(F)c3)n1)NCCC2', 'dataset-59'),
# ('COc1cc(C)cc(Nc2ncc(C)c(N(F)c3ccccc3)n2)c1', 'dataset-60'),
# ('C=CC(=O)Nc1cccc(-n2cnc3c(N)nc(Nc4ccc(F)c(F)c4)nc32)c1', 'dataset-61'),
# ('COC(=O)c1ccc(Nc2nccc(C(=O)Nc3cccc(C#N)c3)n2)cc1', 'dataset-62'),
# ('Cc1cc(Nc2nccc(C(F)(F)F)n2)cc(-c2ccnc(C3(C(=O)O)CCCCC3)c2)c1', 'dataset-63'),
# ('Cc1cc(C)nc(Nc2ccc(NC(=O)C3CC(=O)N(CC(O)c4ccccc4)C3)cc2)n1', 'dataset-64'),
# ('COCCNNc1ccc(-c2ccnc(Nc3ccc(N4CCOCC4)cc3)n2)cc1', 'dataset-65'),
# ('COCCNC(=O)c1ccc(Nc2nccc(-c3cnc4c(c3)C(C)(C)CN4)n2)cc1', 'dataset-66'),
# ('Cc1cc(Nc2ccc(F)c(F)c2F)nc(Nc2ccc(N(C)C)cc2)n1', 'dataset-67'),
# ('COc1ccc(Nc2cc(C)nc(Nc3ccccc3F)n2)cc1OC', 'dataset-68'),
# ('Cc1cnc(Nc2ccc(CC3CCCN(C)CC3)cc2)nc1Nc1cccc(C(C)(C)C)c1', 'dataset-69'),
# ('Cc1cc2c(C(=O)NCCC#N)cccc2n1-c1nc2c(c(NCc3ccccc3)n1)CCCC2', 'dataset-70'),
# ('Cc1ccc(C2CC(=O)c3cnc(Nc4ccc(C)cc4C)nc3C2)cc1', 'dataset-71'),
# ('O=C(NCNc1nc(Nc2ccccc2)nc(-c2ccccc2)c1F)C1CC1', 'dataset-72'),
# ('Cc1ccc(Nc2nc(C)cc(N3CCN(C(=O)c4cccc5ccccc45)CC3)n2)cc1', 'dataset-73'),
# ('Cc1cnc(-c2ccc(Nc3ncc4c(n3)C(C)(c3ccccc3)CCC4)cc2F)[nH]1', 'dataset-74'),
# ('Cc1ccc(NC(=O)c2ccccc2CN2CCN(C)CC2)cc1Nc1nccc(-c2cccnc2)n1', 'dataset-75'),
# ('CC(C)(C)OC(=O)Nc1ccc(Nc2ncc3ccc(=O)n(CC4CC4)c3n2)cc1', 'dataset-76'),
# ('CN(c1ccnc(Nc2cc(N3CCOCC3)cc(N3CCOCC3)c2)n1)c1cc(CO)ccc1F', 'dataset-77'),
# ('Cc1cc(NC(=S)NC2CCCCC2)nc2nc(Nc3ccc(N4CCNCC4)cc3)ncc12', 'dataset-78'),
# ('N#CCNC(=O)c1ccc(-c2ccnc(Nc3ccc(N4CCC5(CCO5)C4)cc3)n2)cc1', 'dataset-79'),
# ('CN1CC(c2cc(Nc3ncc4cc(C5CC5)ccc4n3)cc(OC3CCNC3)c2)C=N1', 'dataset-80'),
# ('CCn1c(-c2ccnc(Nc3ccc(C(=O)N4CCC(N(C)C)C4)cc3)n2)cnc1C(F)F', 'dataset-81'),
# ('NC(=O)c1ccc(Nc2nc(NC3CCNCC3)c3ccn(S(=O)(=O)Cc4ccccc4)c3n2)cc1', 'dataset-82'),
# ('Cc1cc2c(Nc3nccc(NCc4ccc5[nH]ccc5c4)n3)cccc2[nH]1', 'dataset-83'),
# ('Cc1ccc(Nc2ncc3cnn(C4CCNCC4)c3n2)cc1S(N)(=O)=O', 'dataset-84'),
# ('CC(C)(C)NC(=O)Nc1cnc2cnc(Nc3ccc(N4CCC(CCCN5CCNCC5)CC4)cc3)nc2n1', 'dataset-85'),
# ('O=C(NCc1ccc(F)c(F)c1)c1ccc(Nc2nc(NCC(F)(F)F)c3cc[nH]c3n2)cc1', 'dataset-86'),
# ('Cc1ccc(NC(=O)c2cccc(Nc3ncc(-c4ccc(F)cc4)cn3)c2)c(C)c1', 'dataset-87'),
# ('O=C(NCCC1CCCCN1C(=O)c1cccc(Nc2ncccn2)c1)c1ccccc1', 'dataset-88'),
# ('NC1CCC(Nc2nc(Nc3ccc4c(c3)CCC=N4)ncc2C2CC2)CC1', 'dataset-89'),
# ('Cc1cn(-c2ncc(-c3ccccc3F)cn2)c2cc(C(=O)N3CCN(CCO)CC3)ccc12', 'dataset-90'),
# ('CCC(CC)N1c2nc(Nc3cccc(NC(=O)c4ccncc4)c3)ncc2CC1(C)O', 'dataset-91'),
# ('Cc1cnc(Nc2cccc(OCCCN3CCCC3CO)c2)nc1-c1sc(C)c(C#N)c1-c1cccnc1', 'dataset-92'),
# ('COc1cccc(-c2cccc(N(CCCN3CCCN3)c3nccc(N)n3)c2)c1', 'dataset-93'),
# ('COc1cc(-c2cnc(Nc3cc(C(=O)NCCN(C)C)ccc3OC)nc2)ccc1C#N', 'dataset-94'),
# ('N#Cc1ccc(Nc2ncc(C(F)(F)F)c(Nc3ccccc3C3CCCCC3)n2)cc1', 'dataset-95'),
# ('O=C(Nc1ccc(Nc2ncccn2)cc1)c1cc(-c2ccccn2)n[nH]1', 'dataset-96'),
# ('Cc1cccc(NC(=O)c2ccc(Nc3nc(NCC(F)(F)F)c4cc[nH]c4n3)cc2)c1', 'dataset-97'),
# ('O=C(O)c1ccccc1Nc1nccc(-c2cc3ccccc3s2)n1', 'dataset-98'),
# ('CCc1ccc(Nc2ncc(C(F)(F)F)c(Nc3cccc(OCC(C)C)c3)n2)cc1', 'dataset-99'),
# ('COc1ccc2cnc(Nc3ccc(N4CCC(CCCN5CCOCC5)CC4)c(F)c3)nc2c1C1CCCC1', 'dataset-100'),
# ('CCc1ccccc1NC(=O)c1cnc(Nc2ccccc2CC)nc1', 'dataset-101'),
# ('CN1CC2CC1CN2c1ccc(Nc2nccc(-c3cnc4c(c3)OC(C)(C)C(=O)N4)n2)cc1', 'dataset-102'),
# ('CCOC(=O)c1ccccc1Nc1ncc(C(=O)Nc2c(F)cccc2F)cn1', 'dataset-103'),
# ('COc1cc(C(O)NC2CCN(C)CC2)ccc1Nc1ncc(C(F)(F)F)c(OC2CCCC2N)n1', 'dataset-104'),
# ('CS(=O)c1nn(-c2ncc(-c3ccccc3)cn2)c2cc(C3CN(C=O)CCO3)ccc12', 'dataset-105'),
# ('COC(=N)c1cnc(Nc2ccc(C(=O)O)cc2)nc1Nc1ccccc1', 'dataset-106'),
# ('Cc1ccccc1Nc1nc(N(C)c2ccccc2)c2ccsc2n1', 'dataset-107'),
# ('Cc1ccc(Nc2ncc(C)c(N(C)C3CCCN4CCCCC34)n2)cc1C#N', 'dataset-108'),
# ('COc1cc(N2CCOCC2)ccc1Nc1nc2c(c(Nc3ccccc3NC(C)=O)n1)C1(CC1)CN2', 'dataset-109'),
# ('CCOC(=O)c1ccc(Nc2nc(C)cc(NCc3ccccn3)n2)cc1', 'dataset-110'),
# ('Cc1cc(Nc2nccc(C(F)(F)F)n2)cc(-c2cc(C(=O)O)ccc2C#N)c1', 'dataset-111'),
# ('CCOc1ccccc1Nc1nccc(N2CCc3ccccc32)n1', 'dataset-112'),
# ('Cc1cc(Nc2nccc(C(F)(F)F)n2)cc(-c2cnc(-c3ccncc3)s2)c1', 'dataset-113'),
# ('Cc1cc(C(=O)NCCc2ccccc2F)nc(Nc2c(C)cccc2C)n1', 'dataset-114'),
# ('CN1CCC(Oc2ccc(Nc3nc(-c4cccc(CC#N)c4)c4cc[nH]c4n3)cc2F)CC1', 'dataset-115'),
# ('N#Cc1cnc(Nc2ccc(S(=O)(=O)NCCc3ccccc3)cc2)nc1N', 'dataset-116'),
# ('Cc1cccc(Nc2nc(Nc3ccc(N4CCNC(=O)C4)cc3)ncc2C(N)=O)c1', 'dataset-117'),
# ('CC(C)c1cc2c(C(N)=O)cccc2n1-c1nc2c(c(NCc3ccccc3)n1)COCC2', 'dataset-118'),
# ('CC(=O)c1cnc(Nc2ccc3cn[nH]c3c2)nc1Nc1ccc2c(c1)NCCO2', 'dataset-119'),
# ('C#CCCC(CC)S(=O)(=O)c1ccc(-c2ccnc(Nc3ccc(CCO)cc3)n2)s1', 'dataset-120'),
# ('CC1COc2cnc(Nc3ccc(-n4cnc(C5CCC5)c4)c(C#N)c3)nc2N1C', 'dataset-121'),
# ('Cc1cc(C)nc(Nc2ccc(C(=O)NCC3CCCO3)cc2)n1', 'dataset-122'),
# ('CN1CCC(NC(=O)c2ccc(Nc3nccc(-c4cnc5c(c4)C(C)(C)CN5)n3)cc2)CC1', 'dataset-123'),
# ('N#CCN1CCC(c2ccc(Nc3ncc(C(N)=O)c(NC4CC4)n3)cc2)CC1', 'dataset-124'),
# ('CC(Nc1nc(Nc2ccc(N3CCN(C)CC3)cc2)ncc1CN1C(=O)c2ccccc2C1=O)C(=O)O', 'dataset-125'),
# ('COc1cc(Nc2nccc(-c3ccc(C(=O)NCC#N)cc3)n2)cc(OC)c1OC', 'dataset-126'),
# ('Cc1cccc(CNC(=O)c2ccnc(Nc3ccccc3C(F)(F)F)n2)c1', 'dataset-127'),
# ('FC(F)(F)c1ccc(Nc2nc(NCc3ccccc3)c3ccccc3n2)cc1', 'dataset-128'),
# ('C=CC(=O)Nc1cccc(Nc2nc(Nc3ccc(OCC4CC(O)CN4)c(F)c3)ncc2F)c1', 'dataset-129'),
# ('Cc1ccc2nc(Nc3cccc(CNC(=O)c4cccc(N)n4)c3)ncc2c1', 'dataset-130'),
# ('CC(Nc1nc(Nc2ccc(C3CCN(C)CC3)cc2)ncc1F)C(C(N)=O)C1C=CCC1', 'dataset-131'),
# ('COc1ccc(Nc2nccc(NCC(O)c3ccccc3C)n2)cc1', 'dataset-132'),
# ('O=C(Nc1c(F)cccc1F)c1cnc(Nc2ccc3c(c2)OCCO3)nc1', 'dataset-133'),
# ('COc1cc(C)c(S(=O)(=O)Nc2ccc(Nc3nc(C)cc(Nc4ccccc4)n3)cc2)cc1C', 'dataset-134'),
# ('COc1ccccc1CCNc1ccnc(Nc2cccc(NC(C)=O)c2)n1', 'dataset-135'),
# ('Cc1cccc(NC(=O)Nc2cccc(Nc3nc(Nc4ccc(N5CCN(C)CC5)cc4)ncc3C)c2)c1', 'dataset-136'),
# ('CCc1ccccc1-c1ccc(-c2ccnc(Nc3ccc(CCO)cc3)n2)s1', 'dataset-137'),
# ('O=c1ccc2cnc(Nc3ccc4[nH]c5ccccc5c4c3)nc2n1CC1CC1', 'dataset-138'),
# ('CC(=O)N1CCN(c2ccc(Nc3nccc(-c4sc(-c5cccs5)nc4C)n3)cc2)CC1', 'dataset-139'),
# ('CC(=O)c1ccc(-n2ccc3cnc(Nc4ccc(F)cc4)nc32)cc1', 'dataset-140'),
# ('CC1CCN(c2ccc(Nc3nc(Nc4cn(C)nc4SC(C)C)c4sccc4n3)cc2)CC1', 'dataset-141'),
# ('N#CCc1cccc(Nc2nc(Nc3ccc(N4CCOCC4)cc3)nc3[nH]ccc23)c1', 'dataset-142'),
# ('COC(=O)c1cccc(Nc2nc(C)cc(NCc3ccccc3F)n2)c1', 'dataset-143'),
# ('CCCc1cc(N2CCCC(N(C)C3CCCC3)C2)nc(Nc2ccc(C)c(C#N)c2)n1', 'dataset-144'),
# ('Cc1cnc(Nc2cccc(S)c2)nc1Nc1ccc2c(c1)CCN(S(=O)(=O)C1CC1)C2', 'dataset-145'),
# ('Cn1cc(-c2ccnc(Nc3ccc(C4CCCNC4)cc3)n2)c(-c2cccnc2)n1', 'dataset-146'),
# ('Cc1cc(N2CCCC2)nc(Nc2ccc(NC(=O)c3c(C)nc4ccccn34)cc2)n1', 'dataset-147'),
# ('CCCCNC(=O)c1cc2cc(Nc3nccc(-c4ccccc4)n3)ccc2[nH]1', 'dataset-148'),
# ('CC(C)Oc1ccc(COc2nc(N3CCc4ccccc43)ncc2C(=O)NC2CCCCNC2=O)cc1', 'dataset-149'),
# ('CCCOc1cccc(Nc2nc(Nc3ccc(OC4CCCC4)cc3)ncc2C(F)(F)F)c1', 'dataset-150'),
# ('Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)nc1Oc1cccc(S(=O)(=O)NC(C)(C)C)c1', 'dataset-151'),
# ('CCN1CCN(C(=O)c2cnc(Nc3cccc4cccnc34)nc2)CC1', 'dataset-152'),
# ('N#Cc1cc(-c2ccnc(Nc3ccc(N4CCOCC4)cc3)n2)ccc1NC(=O)CC(F)(F)F', 'dataset-153'),
# ('Cc1ccc(Nc2nc(-c3ccccc3)cs2)cc1Nc1nccc(-c2cccnc2)n1', 'dataset-154'),
# ('Cc1cc(N)c(C2CCCC2)c2nc(Nc3ccc(N4CCC(CCCN5CCNCC5)CC4)cc3)ncc12', 'dataset-155'),
# ('CC(C)NC(=O)C1CCCC1CCCc1nc(Nc2ccc(C(=O)OCc3ccccc3)cc2)ncc1C(F)(F)F', 'dataset-156'),
# ('COC(=O)C1(c2ccc(Nc3nc(-c4ccc(F)cc4)cc(N(C)C4CC4)n3)cc2)CCCC1', 'dataset-157'),
# ('CCCc1cc(N2CCC(NCc3ccccc3F)C2)nc(Nc2cccc(C#N)c2)n1', 'dataset-158'),
# ('Cc1cc2c(-c3nn[nH]n3)cccc2n1-c1nc2c(c(NCc3ccccc3)n1)CNCC2', 'dataset-159'),
# ('Cc1cc(Nc2nc(Nc3cc(N)ccc3C)ncc2OCc2ccccc2)n[nH]1', 'dataset-160'),
# ('COc1ccccc1Nc1ccnc(Nc2ccc(N3CCCC3)cc2)n1', 'dataset-161'),
# ('CNc1nc(Nc2ccc(-c3ccc(N(C)C)cc3)cc2)nc(C)c1-c1ccccn1', 'dataset-162'),
# ('Cc1ccc(NC(=O)c2cnc(Nc3ccccc3OC(C)C)nc2)cc1C', 'dataset-163'),
# ('CC1CN(c2ccc(Nc3ncc4ccn(Cc5cccnc5N(C)S(=O)(=O)C5CC5)c4n3)cc2)CCN1', 'dataset-164'),
# ('Nc1nc2ccccc2n1-c1nc2c(c(NCc3ccccc3)n1)CCCC2', 'dataset-165'),
# ('Cc1cc(N2CCCC2)nc(Nc2ccc(Oc3ccccc3)cc2)n1', 'dataset-166'),
# ('Cc1scnc1-c1cc2cnc(Nc3ccc(N4CCNCC4)c(F)c3)nc2n(Cc2cnccc2C2CC2)c1=O', 'dataset-167'),
# ('C=CCNC(=O)c1cnc(Nc2ccc(N3CCN(C)CC3C)cc2)nc1Nc1ccc2c(n1)C(C)CC2', 'dataset-168'),
# ('CC(NC(=O)c1ccc(-c2ccnc(Nc3cccc(N4CCN(C)CC4)c3)n2)s1)c1ccccc1', 'dataset-169'),
# ('N#Cc1cccc(Nc2nccc(N3CCN(c4ccccn4)CC3)n2)c1', 'dataset-170'),
# ('CS(=O)c1cn(-c2ncc(-c3cc(C(N)=O)ccn3)cn2)c2cc(C(=O)N3CCOCC3)ccc12', 'dataset-171'),
# ('Cc1cnc(Nc2cccc(S(N)(=O)=O)c2)nc1Nc1ccc2c(c1)CCC(S(=O)(=O)C1CC1)C2', 'dataset-172'),
# ('COc1cc(Nc2nccc(-c3ncc(C)s3)n2)ccc1-c1ccccc1', 'dataset-173'),
# ('Cc1ccc(C(=O)Nc2ccc(C)c(Nc3nccc(-c4cncc(N5CCOCC5)c4)n3)c2)cc1', 'dataset-174'),
# ('CCCCc1ccc(Nc2nc(Nc3ccc(OC)cc3)c3nccnc3n2)cc1', 'dataset-175'),
# ('CN1CCC(N(C)C(=O)c2ccc(Nc3ncc(C(F)(F)F)c(NC4CCCC4CC#N)n3)cc2)CC1', 'dataset-176'),
# ('NC(=O)c1cc2nc(Nc3ccc(F)cc3)nc(NC3CCC3)c2[nH]1', 'dataset-177'),
# ('Cc1cc(C)c(NC(=O)c2cnc(Nc3c(C)cccc3C)nc2)c(C)c1', 'dataset-178'),
# ('Cc1ccc2c(c1)NC(=O)Cc1cnc(Nc3ccc(C(=O)NCCc4c(C)n[nH]c4C)cc3)nc1-2', 'dataset-179'),
# ('CC1(C)Oc2c(cc(Nc3nccc(C(F)(F)F)n3)cc2-c2cnc(C3(O)CCC3)s2)NC1=O', 'dataset-180'),
# ('CC(C)n1c(=O)c2cnc(Nc3ccc4c(c3)CCNC4)nc2n1-c1ccc(C(C)(C)C)n1C', 'dataset-181'),
# ('CCN(CC)c1cccc(C(=O)Nc2ccc(C)c(Nc3nccc(C4=CNC=CN4)n3)c2)c1', 'dataset-182'),
# ('Cc1cc(C)cc(Nc2ncc(-c3ccc4s[nH]c(=O)c4c3)c(-n3nc(C(F)(F)F)cc3C)n2)c1', 'dataset-183'),
# ('N#Cc1ccc(Nc2nccc(OCc3ccc4ccccc4c3)n2)cc1', 'dataset-184'),
# ('COC(=O)c1ccc(Nc2nc(C)cc(Nc3ccc(C(C)(C)C)cc3)n2)cc1', 'dataset-185'),
# ('c1ccc(COc2ccc(Nc3nc(NC4CCCCC4)c4[nH]cnc4n3)cc2)cc1', 'dataset-186'),
# ('Cc1c(-c2ccnc(Nc3ccc(S(=O)(=O)N4CCCCC4)cc3)n2)c(C)n2c1C(=O)NCC2', 'dataset-187'),
# ('CN(c1ccc(CNC(=O)OC(C)(C)C)cc1)c1ncc2c(n1)CCN(S(C)(=O)=O)C2', 'dataset-188'),
# ('Fc1cc(-c2nc(Nc3ccc(CN4CCOCC4)cc3)ncc2F)cc(N2CCOCC2)c1', 'dataset-189'),
# ('CCCCc1cc(N2CCCC(NC)C2)nc(Nc2ccc(C)c(F)c2)n1', 'dataset-190'),
# ('CN(C)c1cccc(CNc2nc(Nc3cccc(NS(C)(=O)=O)c3)nc3nc[nH]c23)c1', 'dataset-191'),
# ('CCN(CC)C(=O)c1ccnc(Nc2ccc(N3CCC(C)CC3)cc2)n1', 'dataset-192'),
# ('CC(=O)Nc1ccc(Nc2nc(Nc3ccc4ncccc4c3)ncc2C)cc1', 'dataset-193'),
# ('Cc1ccc(CO)cc1NC(=O)c1ccc(Nc2ncc(F)c(-c3ccc(OC(F)(F)F)cc3)n2)c(F)c1', 'dataset-194'),
# ('CCc1ccccc1Nc1nccc(C(=O)Nc2ccccc2C(=O)OC)n1', 'dataset-195'),
# ('O=C(c1cnc(Nc2ccc(F)cc2)nc1)N1CCN(c2ncccn2)CC1', 'dataset-196'),
# ('CC(=O)c1cccc(NC(=O)c2cc(C)nc(Nc3cc(C)ccc3C)n2)c1', 'dataset-197'),
# ('CC1(C)Oc2ccc(Nc3nc(Nc4ccc5c(c4)NSCO5)ncc3F)nc2NC1=O', 'dataset-198'),
# ('O=C(NCCCN1CCCC1=O)c1cccc(Nc2ncc3nnn(-c4ccc5c(c4)CCC5)c3n2)c1', 'dataset-199'),
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