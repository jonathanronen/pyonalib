"""
Module for reading in VCF files.

@jonathanronen 2015/10
"""

import csv
import pandas as pd

class VCFDictReader:
    """
    Class to read VCF files and return a dict per record.

    # Metadata
    with open('file.vcf') as infile:
        reader = VCFDictReader(infile)
        reader.metadata

    # Iteration
    for rec in reader:
        ...
    >
    rec is { '#CHROM': '1',
             'ALT': 'A',
             'FILTER': '.',
             'ID': 'ENST00000335137:c.849G>A',
             'INFO': {
                'AA_MAF': '',
                'MetaLR_score': '',
                'Protein_position': '283',
                'cDNA_position': '849'
                ...
             },
             'POS': '69939',
             'QUAL': '.',
             'REF': 'G'
             ...
           }
    """
    def __init__(self, infile):
        """
        `infile` is a file-like object that supports `next(infile)` and `infile.seek()`
        """
        # Read in metadata lines
        self.metadata = list()
        for line in infile:
            if line.startswith('#'):
                self.metadata.append(line)
            else:
                break
        del self.metadata[-1] #The last one was the headers
        
        # Get columns list for INFO column
        try:
            self._info_columns = self.metadata[-1].split('Format: ')[1].strip().strip('>"').split('|')
        except:
            self._info_columns = []
        
        # Initialize a csv.DictReader from the header line
        infile.seek(0)
        for i in range(len(self.metadata)):
            next(infile)
        self.csv_dict_reader = csv.DictReader(infile, delimiter='\t')
    
    def __next__(self):
        rec = next(self.csv_dict_reader)
        info = ''.join(rec['INFO'].split('=')[1:])
        rec['INFO'] = dict(zip(self._info_columns, info.split('|')))
        return rec
    
    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

class VEPDictReader:
    """
    """
    def __init__(self, infile, extra_column_name='Extra'):
        """
        `infile` is a file-like object that supports `next(infile)` and `infile.seek()`
        """
        # Read in metadata lines
        self.metadata = list()
        for line in infile:
            if line.startswith('#'):
                self.metadata.append(line)
            else:
                break
        del self.metadata[-1] #The last one was the headers
        self._extra_column_name = extra_column_name
        
        # Get columns list for Extra column
        colnames, coldescs = zip(*[[k.strip() for k in e.strip('# \n').split(':')] for e in reader.metadata[reader.metadata.index('## Extra column keys:\n')+1:]])
        self._extra_columns = colnames
        self.extra_column_descriptions = dict(zip(colnames, coldescs))

        
        # Initialize a csv.DictReader from the header line
        infile.seek(0)
        for i in range(len(self.metadata)):
            next(infile)
        self.csv_dict_reader = csv.DictReader(infile, delimiter='\t')
    
    def __next__(self):
        rec = next(self.csv_dict_reader)
        if self._extra_column_name:
            info = dict([e.split('=') for e in rec['Extra'].split(';')])
            rec.update(info)
            del rec['Extra']
        return rec
    
    def next(self):
        return self.__next__()
    
    def __iter__(self):
        return self

def read_vep(filename):
    with open(filename) as f:
        return pd.concat([pd.Series(e) for e in VEPDictReader(f)], axis=1).T
