## a python library

#### Capabilities

#### Preprocessing

##### Quantile normalization
```python
from pyonalib.preprocessing import quantile_normalize
import numpy as np

test_matrix = np.array([[5,4,3], [2,1,4], [3,4,6], [4,2,8]], dtype=float)
expected_normalized = np.array([[5.67, 4.67, 2], [2,2,3], [3, 4.67, 4.67], [4.67, 3, 5.67]])
qnormalized = quantile_normalize(test_matrix)
assert(np.allclose(qnormalized, expected_normalized, atol=.1))
```

##### VCFReader
```python
    from pyonalib.vcf import VCFDictReader
    
    # Metadata
    with open('file.vcf') as infile:
        reader = VCFDictReader(infile)
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
```

##### Logging
```python
from pyonalib import y_logging

logger = y_logging.getLogger(__file__)
```
This returns a logger with a format I like, and set to the `logging.INFO` level. Specifically,
```python
basicConfig(format='%(asctime)s\t%(pathname)s:%(lineno)s--%(levelname)s: %(message)s', level=INFO)
```


---------------------
[@jonathanronen](https://github.com/@jonathanronen) 2015
