# Amplicon sequencing dataset

Amplicon sequencing dataset of potato rhizosphere microbiomes collected from major production regions in Southwest China, including raw sequences, zOTU tables, taxonomic assignments, and associated metadata.

![Figure 1](./Figures/Figure%201.png)

Dataset contents:

* Raw FASTQ files (Illumina MiSeq platform) targeting the bacterial 16S rRNA gene, deposited in the SRA database under accession number PRJNA1449797 [https://www.ncbi.nlm.nih.gov/sra/?term=PRJNA1449797](https://www.ncbi.nlm.nih.gov/sra/?term=PRJNA1449797)
* Processed zOTU tables generated using the USEARCH pipeline
* Representative sequences and corresponding taxonomic assignments
* Sample metadata, including soil type, cropping system, disease occurrence, potato cultivar, and other relevant variables

# Install dependence

```bash
git clone https://github.com/zhaoy2020/AlphaMicrobiome.git
cd AlphaMicrobiome
pip install -e .
```

# Usage

```python
from microbiome import amplicon, diversity, stats

help(amplicon)
```

# Cite

