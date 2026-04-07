# Amplicon sequencing dataset

Amplicon sequencing dataset of potato rhizosphere microbiomes collected from major production regions in Southwest China, including raw sequences, zOTU tables, taxonomic assignments, and associated metadata.

Dataset contents:

- Raw FASTQ files (Illumina MiSeq platform) targeting the 16S rRNA gene (bacteria)
- Processed zOTUtables ( USEARCH pipelines)
- Representative sequences and taxonomic assignments
- Sample metadata including soil type, potato variety, disease status and *etc*.

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
