"""
Module for TCGA stuff
"""
import pandas as pd
# TCGA_CODE_TABLE = pd.DataFrame([
#   ['01', 'Primary Solid Tumor', 'TP'],
#   ['02', 'Recurrent Solid Tumor', 'TR'],
#   ['03', 'Primary Blood Derived Cancer - Peripheral Blood', 'TB'],
#   ['04', 'Recurrent Blood Derived Cancer - Bone Marrow', 'TRBM'],
#   ['05', 'Additional - New Primary', 'TAP'],
#   ['06', 'Metastatic', 'TM'],
#   ['07', 'Additional Metastatic', 'TAM'],
#   ['08', 'Human Tumor Original Cells', 'THOC'],
#   ['09', 'Primary Blood Derived Cancer - Bone Marrow', 'TBM'],
#   ['10', 'Blood Derived Normal', 'NB'],
#   ['11', 'Solid Tissue Normal', 'NT'],
#   ['12', 'Buccal Cell Normal', 'NBC'],
#   ['13', 'EBV Immortalized Normal', 'NEBV'],
#   ['14', 'Bone Marrow Normal', 'NBM'],
#   ['15', 'sample type 15', '15SH'],
#   ['16', 'sample type 16', '16SH'],
#   ['20', 'Control Analyte', 'CELLC'],
#   ['40', 'Recurrent Blood Derived Cancer - Peripheral Blood', 'TRB'],
#   ['50', 'Cell Lines', 'CELL'],
#   ['60', 'Primary Xengraft Tissue', 'XP'],
#   ['61', 'Cell Line Derived Xenograft Tissue', 'XCL'],
#   ['99', 'sample type 99', '99SH'],
#   ], columns=['Code', 'Definition', 'Short Letter Code']
#   )

CODE, DEFINITION, SHORT_CODE = list(zip(*[
    ['01', 'Primary Solid Tumor', 'TP'],
    ['02', 'Recurrent Solid Tumor', 'TR'],
    ['03', 'Primary Blood Derived Cancer - Peripheral Blood', 'TB'],
    ['04', 'Recurrent Blood Derived Cancer - Bone Marrow', 'TRBM'],
    ['05', 'Additional - New Primary', 'TAP'],
    ['06', 'Metastatic', 'TM'],
    ['07', 'Additional Metastatic', 'TAM'],
    ['08', 'Human Tumor Original Cells', 'THOC'],
    ['09', 'Primary Blood Derived Cancer - Bone Marrow', 'TBM'],
    ['10', 'Blood Derived Normal', 'NB'],
    ['11', 'Solid Tissue Normal', 'NT'],
    ['12', 'Buccal Cell Normal', 'NBC'],
    ['13', 'EBV Immortalized Normal', 'NEBV'],
    ['14', 'Bone Marrow Normal', 'NBM'],
    ['15', 'sample type 15', '15SH'],
    ['16', 'sample type 16', '16SH'],
    ['20', 'Control Analyte', 'CELLC'],
    ['40', 'Recurrent Blood Derived Cancer - Peripheral Blood', 'TRB'],
    ['50', 'Cell Lines', 'CELL'],
    ['60', 'Primary Xenograft Tissue', 'XP'],
    ['61', 'Cell Line Derived Xenograft Tissue', 'XCL'],
    ['99', 'sample type 99', '99SH'],
    ]))

def barcode2sampcode(barcode):
    return barcode.split('-')[3][:2]

CODE2DEF = {code: defi for code, defi in zip(CODE, DEFINITION)}
def barcode2samptype(barcode):
    return CODE2DEF[barcode2sampcode(barcode)]


TUMOR_SAMPLE_CODES = { '01', '02', '03', '04', '05', '09', '40' }
def select_tumor_samples(df):
    tumor_samples = [e for e in df.index if barcode2sampcode(e) in TUMOR_SAMPLE_CODES]
    return df.loc[tumor_samples]

NORMAL_SAMPLE_CODES = { '10', '11', '12', '13', '14' }
def select_normal_samples(df):
    normal_samples = [e for e in df.index if barcode2sampcode(e) in NORMAL_SAMPLE_CODES]
    return df.loc[normal_samples]
