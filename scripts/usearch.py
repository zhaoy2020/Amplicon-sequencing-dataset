
    # Author: Yu Zhao
    # Copyright (c) 2025 Yu Zhao
    

#@save: imported packages
from loguru import logger 
from pathlib import Path

import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams['svg.fonttype'] = 'none'  # To keep fonts editable in Illustrator
import seaborn as sns 
sns.set_theme(
    context='paper', style='ticks', palette='husl', 
    font='arial',
    rc={
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    },
)

from deepspore import run_cmd, num_thread, parallel 


#@save: directory setup
home_dir: str = input('Enter the home directory path: ')

# home_dir: str = ''

home_dir: Path = Path(home_dir)
data_dir: Path = home_dir.joinpath('datas')
results_dir: Path = home_dir.joinpath('results')

logger.info(f'Home directory set to {home_dir}')
if not results_dir.exists():
    results_dir.mkdir(parents=True)
    logger.info(f'Created results directory at {results_dir}')


#@save show data files
logger.info(f'Data directory is set to: {data_dir}')
for read_file in data_dir.iterdir():
    print(read_file)


#@save: quality control step
logger.info('Quality control step starting...')
def quality_control(data_dir: Path, results_dir: Path):
    qc_dir: Path = results_dir.joinpath('001_quality_control')
    if not qc_dir.exists():
        qc_dir.mkdir(parents=True)
        logger.info(f'Created quality control directory at {qc_dir}')


#@save
# Quality control
quality_control(data_dir, results_dir)   


#@save: merging paired-end reads step
logger.info('Merging paired-end reads step starting...')

def task(file_dir: Path, file_name:str, output_dir: Path) -> None:
    '''Qualitify Merged reads, such as merge reads, trimi primer, and etc.'''

    # Merge the read files
    file_1, file_2 = tuple(f"{file_dir}/{file_name}_{i}.fastq" for i in [1, 2])
    if Path(file_1).exists() and Path(file_2).exists():
        logger.info(f'Dected paired-end files with _1.fastq and _2.fastq suffixes for {file_name}. Proceeding with merging.')
    else:
        file_1, file_2 = tuple(f"{file_dir}/{file_name}.{i}.fq" for i in [1, 2])
        if Path(file_1).exists() and Path(file_2).exists():
            logger.info(f'Detected paired-end files with .1.fq and .2.fq suffixes for {file_name}. Proceeding with merging.')
        else:
            logger.warning(f'No paired-end files found for {file_name} with expected suffixes. Skipping merging for this file name.')
            raise FileNotFoundError(f'No paired-end files found for {file_name} with expected suffixes.')
        
    merge_file = output_dir.joinpath(f"{file_name}.fastq")
    ## @ 很重要
    merge_reads = f"usearch11 -fastq_mergepairs {file_1} -reverse {file_2} -fastqout {merge_file} -relabel @"
    # Remove the primers
    # f"usearch11 -fastx_trim_primer {merge_file} -db primers.fa -strand both -maxdiffs 1 -width 0 -fastqout ./results/merge_read/{file_name}_trim.fastq"    
    # run cmd via subprocess module
    run_cmd(merge_reads)
    
def merge_paired_end_reads(data_dir: Path, results_dir: Path):
    merge_dir: Path = results_dir.joinpath('002_merge_paired_end_reads')
    if not merge_dir.exists():
        merge_dir.mkdir(parents=True)
        logger.info(f'Created merging paired-end reads directory at {merge_dir}')
    
    print("Starting the merging process...")
    # Get all file_name to a tuple.
    # file_name_list = {file.stem.split("_")[0] for file in data_dir.iterdir() if file.suffix in {'.fastq', '.fq'}}
    file_name_list = {file.stem.split(".")[0] for file in data_dir.iterdir() if file.suffix in {'.fastq', '.fq'}}
    # Build the tasks consist of tuple which contains function and its paramters.
    tasks = [(task, (data_dir, file_name, merge_dir)) for file_name in file_name_list]
    # Directly using Pool, with stop manually.
    parallel(tasks= tasks, num_threads= num_thread())    
    print("All finished.")


#@save
# Merging paired-end reads
merge_paired_end_reads(data_dir, results_dir)


#@save: merge all samples
logger.info('Merging all samples step starting...')

def merge_all_samples(results_dir: Path):
    merge_dir: Path = results_dir.joinpath('002_merge_paired_end_reads')
    all_samples_file: Path = results_dir.joinpath('003_all_samples_merged')
    if not all_samples_file.exists():
        all_samples_file.mkdir(parents=True)
        logger.info(f'Created all samples merged directory at {all_samples_file}')
    merge_cmd = f'cat {merge_dir}/*.fastq > {all_samples_file.joinpath("all_samples.fastq")}'
    run_cmd(merge_cmd)
    
    # Count total reads
    print("Total reads in all samples:")
    run_cmd(f'grep @ {all_samples_file.joinpath("all_samples.fastq")} | wc -l')


#@save
# Merging all samples
merge_all_samples(results_dir)


#@save: filtering low-quality reads step
logger.info('Filtering low-quality reads step starting...')

def filter_low_quality_reads(input_dir: Path):
    all_samples_file: Path = input_dir.joinpath('all_samples.fastq')
    all_samples_filtered: Path = input_dir.joinpath('all_samples_filtered.fastq')
    filter_cmd = f'usearch11 -fastq_filter {all_samples_file} -fastaout {all_samples_filtered} -fastq_maxee 1.0 -relabel Filt -threads {num_thread()}'
    print(filter_cmd)
    run_cmd(filter_cmd)
    
    # Count total reads after filtering
    print("Total reads after filtering:")
    run_cmd(f'grep ">" {all_samples_filtered} | wc -l')


#@save
# Filtering low-quality reads
filter_low_quality_reads(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: dereplication step
logger.info('Dereplication step starting...')

def dereplication(input_dir: Path):
    all_samples_filtered: Path = input_dir.joinpath('all_samples_filtered.fastq')
    derep_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated.fasta')
    derep_cmd = f'usearch11 -fastx_uniques {all_samples_filtered} -fastaout {derep_fasta} -relabel Uniq -sizeout -threads {num_thread()}'
    print(derep_cmd)
    run_cmd(derep_cmd)
    
    # Count total unique reads after dereplication
    print("Total unique reads after dereplication:")
    run_cmd(f'grep ">" {derep_fasta} | wc -l')


#@save
# Dereplication
dereplication(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: remove singletons step
logger.info('Remove singletons step starting...')

def remove_singletons(input_dir: Path):
    derep_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated.fasta')
    no_singleton_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton.fasta')
    remove_cmd = f'usearch11 -sortbysize {derep_fasta} -fastaout {no_singleton_fasta} -minsize 2 '
    print(remove_cmd)
    run_cmd(remove_cmd)
    
    # Count total unique reads after removing singletons
    print("Total unique reads after removing singletons:")
    run_cmd(f'grep ">" {no_singleton_fasta} | wc -l')


#@save
# Remove singletons
remove_singletons(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: preorder step
logger.info('Preorder step starting...')

def preorder(input_dir: Path, minisize: int):
    no_singleton_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton.fasta')
    preorder_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder.fasta')
    preorder_cmd = f'usearch11 -sortbysize {no_singleton_fasta} -fastaout {preorder_fasta} -minsize {minisize}'
    print(preorder_cmd)
    run_cmd(preorder_cmd)
    
    # Count total OTUs after preorder
    print("Total OTUs after preorder:")
    run_cmd(f'grep ">" {preorder_fasta} | wc -l')


#@save
# Preorder
preorder(input_dir=results_dir.joinpath('003_all_samples_merged'), minisize=8)


#@save: clustering to OTUs step
logger.info('Clustering to OTUs step starting...')

def cluster_to_otus(input_dir: Path):
    preorder_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder.fasta')
    otu_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus.fasta')
    otu_fasta_log: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus.log')
    cluster_cmd = f'usearch11 -cluster_otus {preorder_fasta} -otus {otu_fasta} -relabel Otu -threads {num_thread()} > {otu_fasta_log} 2>&1'
    print(cluster_cmd)
    run_cmd(cluster_cmd)
    
    # Count total OTUs after clustering
    print("Total OTUs after clustering:")
    run_cmd(f'grep ">" {otu_fasta} | wc -l')


#@save
# Clustering to OTUs
# cluster_to_otus(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: denosing step
logger.info('Denoising to zOTU step starting...')

def denoising(input_dir: Path):
    no_singleton_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder.fasta')
    denoised_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus.fasta')
    denoised_fasta_log: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus.log')
    denoise_cmd = f'usearch11 -unoise3 {no_singleton_fasta} -zotus {denoised_fasta} -threads {num_thread()} > {denoised_fasta_log} 2>&1'
    print(denoise_cmd)
    run_cmd(denoise_cmd)
    
    # Count total ZOTUs after denoising
    print("Total ZOTUs after denoising:")
    run_cmd(f'grep ">" {denoised_fasta} | wc -l')


#@save
# Denoising
denoising(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: building the feature table step
logger.info('Building the OTU feature table step starting...')

def build_otu_feature_table(input_dir: Path):
    all_samples: Path = input_dir.joinpath('all_samples.fastq')
    otu_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus.fasta')
    feature_table: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus_feature_table.txt')
    feature_table_log: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus_feature_table.log')
    feature_table_map: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus_feature_table_map.txt')
    # build_cmd = f'nohup usearch11 -otutab {all_samples} -otus {otu_fasta} -otutabout {feature_table} -mapout {feature_table_map} -threads {num_thread()} > {feature_table_log} 2>&1 &'
    build_cmd = f'usearch11 -otutab {all_samples} -otus {otu_fasta} -otutabout {feature_table} -mapout {feature_table_map} -threads {num_thread()} > {feature_table_log} 2>&1'
    print(build_cmd)
    run_cmd(build_cmd)
    
    # # Show first 10 lines of the feature table
    # print("First 10 lines of the feature table:")
    # run_cmd(f'head -n 10 {feature_table}')


#@save
# Building the feature table
# build_otu_feature_table(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: building zOTU step
logger.info('Building the zOTU feature table step starting...')

def build_zotu_feature_table(input_dir: Path):
    all_samples: Path = input_dir.joinpath('all_samples.fastq')
    zotu_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus.fasta')
    zotu_feature_table: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus_feature_table.txt')
    zotu_feature_table_log: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus_feature_table.log')
    zotu_feature_table_map: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus_feature_table_map.txt')
    # build_cmd = f'nohup usearch11 -otutab {all_samples} -zotus {zotu_fasta} -otutabout {zotu_feature_table} -mapout {zotu_feature_table_map} -threads {num_thread()} > {zotu_feature_table_log} 2>&1 &'
    build_cmd = f'usearch11 -otutab {all_samples} -zotus {zotu_fasta} -otutabout {zotu_feature_table} -mapout {zotu_feature_table_map} -threads {num_thread()} > {zotu_feature_table_log} 2>&1'
    print(build_cmd)
    run_cmd(build_cmd)
    
    # # Show first 10 lines of the zOTU feature table
    # print("First 10 lines of the zOTU feature table:")
    # run_cmd(f'head -n 10 {zotu_feature_table}')


#@save
# Building the zOTU feature table
build_zotu_feature_table(input_dir=results_dir.joinpath('003_all_samples_merged'))


#@save: nbc taxonomy assignment step

def nbc_taxonomy_assignment(fasta_file: Path, db_file: Path, output_file: Path):
    logger.info('nbc taxonomy assignment step starting...')
    # taxonomy_cmd = f'nohup usearch11 -nbc_tax {fasta_file} -db {db_file} -strand plus -tabbedout {output_file} > {output_file}.log 2>&1 &'
    taxonomy_cmd = f'usearch11 -nbc_tax {fasta_file} -db {db_file} -strand plus -tabbedout {output_file} > {output_file}.log 2>&1'
    print(taxonomy_cmd)
    run_cmd(taxonomy_cmd)


#@save: sintax taxonomy assignment step

def sintax_taxonomy_assignment(fasta_file: Path, db_file: Path, output_file: Path, sintax_cutoff: float = 0.8):
    logger.info('Sintax taxonomy assignment step starting...')
    # taxonomy_cmd = f'nohup usearch11 -sintax {fasta_file} -db {db_file} -tabbedout {output_file} -strand both -sintax_cutoff 0.8 -threads {num_thread()} > {output_file}.log 2>&1 &'
    taxonomy_cmd = f'usearch11 -sintax {fasta_file} -db {db_file} -tabbedout {output_file} -strand both -sintax_cutoff {sintax_cutoff} -threads {num_thread()} > {output_file}.log 2>&1'
    print(taxonomy_cmd)
    run_cmd(taxonomy_cmd)


#@save: taxonomy assignment step

def taxonomy_assignment(input_dir: Path, db_file: Path):
    logger.info('Taxonomy of OTU')
    # OTU
    otu_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus.fasta')
    otu_taxonomy_output: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_otus_taxonomy.txt')
    ## nbc
    # nbc_taxonomy_assignment(fasta_file=otu_fasta, db_file=db_file, output_file=otu_taxonomy_output)
    ## sintax
    sintax_taxonomy_assignment(fasta_file=otu_fasta, db_file=db_file, output_file=otu_taxonomy_output)
    
    logger.info('Taxonomy of zOTU')
    # zOTU
    zotu_fasta: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus.fasta')
    zotu_taxonomy_output: Path = input_dir.joinpath('all_samples_filtered_dereplicated_no_singleton_preorder_zotus_taxonomy.txt')
    ## nbc 
    # nbc_taxonomy_assignment(fasta_file=zotu_fasta, db_file=db_file, output_file=zotu_taxonomy_output)
    ## sintax
    sintax_taxonomy_assignment(fasta_file=zotu_fasta, db_file=db_file, output_file=zotu_taxonomy_output)


#@save 
# Taxonomy assignment
taxonomy_assignment(
    input_dir=results_dir.joinpath('003_all_samples_merged'), 
    # db_file=Path('../dbs/rdp_16s_v18_usearch.udb')              # For sintax
    db_file=Path('../dbs/silva_16s_v123_usearch.udb'),
)