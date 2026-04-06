
    # Author: Yu Zhao
    # Copyright (c) 2025 Yu Zhao
    

#@save

# Basic packages
import importlib
from loguru import logger
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm 

# Scientific computing packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pingouin as pg
from statannotations.Annotator import Annotator

# Diversity and network analysis packages
import skbio
import networkx as nx 

import deepspore
importlib.reload(deepspore)

from microbiome import amplicon, diversity, stats
importlib.reload(amplicon)
importlib.reload(diversity)
importlib.reload(stats)


#@save 
logger.info('Setting seaborn theme...')
# sns.set_style('whitegrid')
# sns.set_context('paper')
# sns.set_palette() # sns.color_palette()

# sns.set_theme?
sns.set_theme(
    context='paper',
    palette='husl',
    style='darkgrid',
    # font="times new roman",
    font="Arial",
    rc={
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    }
)


#@save 
# Check the palette
sns.color_palette()


#@save 
rel_otu_table: pd.DataFrame = amplicon.OTUQC.normalize(otu_table_df=otu_table, method='rel')
rel_otu_table.shape


rel_otu_table.head(3)


rel_otu_table.sum(axis=0) # check if sum to 1


# taxa: str = 'Domain'
taxa: str = 'Phylum'
# taxa: str = 'Class'
# taxa: str = 'Order'
# taxa: str = 'Family'
# taxa: str = 'Genus'
# taxa: str = 'Species'


rel_otu_tax_table: pd.DataFrame = pd.merge(
    left=rel_otu_table,
    right=taxonomy_table[[taxa]],
    left_index=True,
    right_index=True,
    how='left'
).groupby(by=[taxa]).sum() # sum aggregation

print(rel_otu_tax_table.shape)
rel_otu_tax_table.head(3)


rel_otu_tax_table.sum(axis=0) # check sum, should be 1.0


rel_otu_tax_long_table = rel_otu_tax_table.reset_index().melt(
    id_vars=[taxa],
    var_name='SampleID',
    value_name='Relative_Abundance'
)
rel_otu_tax_long_table.head(3)


rel_otu_tax_metadata_table: pd.DataFrame = pd.merge(
    left=rel_otu_tax_long_table,
    right=metadata_table,
    left_on='SampleID',
    right_index=True,
    how='left'
)

print(rel_otu_tax_metadata_table.shape)
rel_otu_tax_metadata_table.head(3)


# group_name: str = 'Soil'
# group_name: str = 'Status'
# group_name: str = 'Planting'
group_name: str = 'Breeding'


topN: int = 15

top_tax_df = (
    rel_otu_tax_metadata_table
    .groupby(by=[taxa])['Relative_Abundance']
    .mean()                                                         # get "mean" relative abundance across all samples
    .reset_index()
    .sort_values(by='Relative_Abundance', ascending=False)
    .head(topN + 1)
)
print('Top taxa shape (including possible Unclassified): ', top_tax_df.shape)
top_tax_df.head(3)


if not top_tax_df[taxa].str.startswith('Unclassified').any():
    top_tax_df = top_tax_df.iloc[:topN]                                         # get topN only from topN + 1
    print('No Unclassified found in top taxa, shape: ', top_tax_df.shape)
else:
    top_tax_df = top_tax_df[top_tax_df[taxa] != f"Unclassified_{taxa}"]         # remove Unclassified if present
    print('Unclassified found in top taxa, removed, shape: ', top_tax_df.shape)

print('Final top taxa:', top_tax_df[taxa].to_list())
top_tax_df.head()


rel_otu_tax_metadata_table[f'{taxa}_TopN'] = rel_otu_tax_metadata_table[taxa].where(rel_otu_tax_metadata_table[taxa].isin(top_tax_df[taxa]), other='Other', inplace=False)
# rel_otu_tax_metadata_table[f'{taxa}_TopN'].unique()
rel_otu_tax_metadata_table.head(3)


df = (
    rel_otu_tax_metadata_table
    .groupby(by=['SampleID', f'{taxa}_TopN'])
    .agg({'Relative_Abundance': 'sum', group_name: 'first'}) # sum for Other per sample
    .reset_index()
    .groupby(by=[group_name, f'{taxa}_TopN'])
    .agg({'Relative_Abundance': 'mean'})                     # mean across samples in the group
    .reset_index()
)
df.head()


# sns.set_style('ticks')

pivot_df = (
    df
    .sort_values(by='Relative_Abundance', ascending=True)
    .pivot(index=group_name, columns=f'{taxa}_TopN', values='Relative_Abundance')
)
cols = pivot_df.sum(axis=0).sort_values(ascending=False).index.to_list() # sort columns by total abundance
if 'Other' in cols:
    cols.remove('Other')
    cols.append('Other') # move Other to the end
print('Final columns order: ', cols)

pivot_df = pivot_df[cols]
# pivot_df
pivot_df.plot(
    kind='bar',
    stacked=True,
    # figsize=(5, 5),
    figsize=(8, 5),
    colormap='tab20',
    # colormap= sns.set_palette('Set1', n_colors=topN + 1),  # +1 for 'Other'
    # colormap= sns.set_palette(colors.COLORS[::-1])
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Relative Abundance')
plt.title(f'Relative Abundance of Top {topN} {taxa} across {group_name}')
plt.tight_layout()


pivot_df.to_csv(
    results_dir.joinpath('005_composition_analysis', f'top_{topN}_{taxa}_across_{group_name}.csv'),
)

plt.savefig(
    results_dir.joinpath('005_composition_analysis', f'top_{topN}_{taxa}_across_{group_name}.svg'),
    format='svg',
)


# There is an error bar.
plt.figure(figsize=(5, 5))
sns.barplot(
    data= df,
    x=group_name,
    y='Relative_Abundance',
    hue=f'{taxa}_TopN',
    dodge=True, # False to make it stacked
    palette='tab20',
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Relative Abundance')
plt.xticks(rotation=45, ha='right')


group_mean = (
    df
    .groupby(by=[group_name, f'{taxa}_TopN'])['Relative_Abundance']
    .mean()
    .reset_index()
)
group_mean.head(3)


# no error bar.
sns.barplot(
    data=group_mean,
    x=group_name,
    y='Relative_Abundance',
    hue=f'{taxa}_TopN',
    dodge=True, # False to make it stacked
    palette='tab20',
)
plt.ylabel('Relative Abundance')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


plt.figure(figsize=(5, 5))
sns.boxplot(
    data= df,
    x=group_name,
    y='Relative_Abundance',
    hue=f'{taxa}_TopN',
    legend=False,
)
plt.xticks(rotation=45, ha='right')


rel_otu_tax_table.head(3)


topN: int = 20

top_taxa = (
    rel_otu_tax_table
    [rel_otu_tax_table.index != f"Unclassified_{taxa}"]             # don't show Unclassified
    .mean(axis=1).sort_values(ascending=False).head(topN).index     # ascending=False, from high to low.
)
print('from high to low:', top_taxa.to_list())

# top_taxa
heat_df = rel_otu_tax_table.loc[top_taxa].copy()
heat_df.head(3)


sns.clustermap(
    data=heat_df,
    cmap='viridis',
    figsize=(5, 5),
    row_cluster=True,
    col_cluster=True,
    standard_scale=0, # min-max normalize, 0 for row, 1 for column.
    # z_score=0, # z_score normalization, 0 for row, 1 for column.
    # center=0,
)


heat_grouped_df = (
    heat_df
    .reset_index()
    .melt(id_vars=taxa, var_name='SampleID', value_name='Relative_Abundance') # wide to long
    .merge(
        right=metadata_table,
        left_on='SampleID',
        right_index=True,
        how='left')                                                              # merge with metadata
    .groupby(by=[group_name, taxa])['Relative_Abundance']
    .mean()
    .reset_index()                                                               # group by group_name and Genus
    .pivot_table(index=taxa, columns=group_name, values='Relative_Abundance')    # pivot to wide format
)
heat_grouped_df


sns.clustermap(
    data=heat_grouped_df,
    cmap='viridis',
    figsize=(5, 5),
    row_cluster=False,
    col_cluster=False,
    # standard_scale=0,
    # z_score=0, # normalize rows
    # center=0,
)


summary = df.groupby(by=[group_name, f'{taxa}_TopN'])['Relative_Abundance'].agg(['mean', 'std']).reset_index()
summary['size'] = summary['mean'] * 200  # scale for better visualization
print('Summary shape: ', summary.shape)
summary.head(3)


plt.figure(figsize=(3, 3))
sns.scatterplot(
    data=summary,
    x=group_name,
    y=f'{taxa}_TopN',
    size='size',
    hue='mean',
    sizes=(20, 200),
    palette='viridis',
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')


rank = (
    rel_otu_table.mean(axis=1)
    .sort_values(ascending=False)
)

plt.figure(figsize=(3, 3))
sns.lineplot(
    x=range(1, len(rank) + 1),
    y=rank.values
)
plt.xscale('log')
plt.xlabel('OTU Rank')
plt.ylabel('Mean Relative Abundance')
plt.title('Rank Abundance Curve', fontsize='large')
plt.tight_layout()


temp_dir: Path = results_dir.joinpath('005_composition_analysis')

temp_list: list = []
for csv_file in temp_dir.glob('*.csv'):
    df = pd.read_csv(csv_file)
    df = df.set_index(df.columns[0]) # set the first column as index
    print(f'CSV file: {csv_file.name}, shape: {df.shape}')
    temp_list.append(df)

temp_list[2]


summary_df = pd.concat(temp_list, axis=0, join='outer')
summary_df = summary_df.drop(index='Unclear')
# pd.concat?

fig, ax = plt.subplots(figsize=(6, 6/2), layout='constrained')
summary_df.plot(
    kind='bar',
    stacked=True,
    figsize=(6, 6/2),
    colormap='tab20',
    ax=ax,
)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, frameon=False)
ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=6, frameon=False)
ax.set_ylabel('Relative Abundance', fontsize=8)
ax.set_xlabel('Group', fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top', fontsize=7)

fig.savefig(
    results_dir.joinpath('005_composition_analysis', f'top_{topN}_all_taxa_across_all_groups.svg'),
    format='svg',
)


print('otu_table shape:', otu_table.shape)
# filtered_otu_table = amplicon.OTUQC.filter_otu_table_by_sample(otu_table_df=otu_table, method='abundance', min_max_threshold=(3961.4, 75284.5), show=True)
filtered_otu_table = amplicon.OTUQC.filter_otu_table_by_sample(otu_table_df=otu_table, method='iqr', show=True)
filtered_otu_table = amplicon.OTUQC.filter_otu_table_by_otu(otu_table_df=filtered_otu_table, min_prevalence=0.01, min_abundance=0.000001, show=True)
print('filtered_otu_table shape:', filtered_otu_table.shape)


# ar = diversity.AutoRarefaction(otu_table=otu_table.iloc[:, :]) # easy debug with all samples
ar = diversity.AutoRarefaction(otu_table=filtered_otu_table.iloc[:, :]) # easy debug with all samples


all_rarefaction_table_df = ar.rarefaction_curves(max_depth=20000, steps=30, repeats=30, alpha_metric_name='chao1')
all_rarefaction_table_df.shape


# Save
# all_rarefaction_table_df.to_csv(
#     results_dir.joinpath('004_diversity_analysis', 'rarefaction_curves', 'filtered_otu_table_rarefaction_curves_chao1_20k.csv')
# )

# Load
all_rarefaction_table_df = pd.read_csv(
    results_dir.joinpath('004_diversity_analysis', 'rarefaction_curves', 'filtered_otu_table_rarefaction_curves_chao1_20k.csv'),
    index_col=0
)
all_rarefaction_table_df.shape


class Annotator:

    def __init__(self):
        self.pointer = 97 # ASCII code for 'a'

    def reset(self):
        self.pointer = 97 # reset to 'a'

    def show(self):
        current_letter = chr(self.pointer).upper()
        self.pointer += 1
        return current_letter
    
    
annotator = Annotator()


nrows: int = 2
ncols: int = 4
fig, axs = plt.subplots(nrows, ncols, layout='constrained', figsize=(3*1.2*ncols, 3*1.2*nrows))
# fig, axs = plt.subplots(nrows, ncols, layout='constrained', figsize=(6, 6/1.6))
axs = axs.flat
for i, group_name in enumerate(['PartName', 'Group', 'Batch', 'AreaName', 'Soil', 'Breeding', 'PreviousCrop', 'Mode']):
    ar.visualization_rarefaction_curves_with_group(
        rarefaction_table_df=all_rarefaction_table_df,
        metadata_table=metadata_table,
        group_name=group_name,
        alpha_metric_name='chao1',
        ax=axs[i],
        show_legend= False if group_name in ['Group', 'Breeding', 'PreviousCrop'] else True,
    )
    axs[i].set_yscale('log')
    axs[i].text(x=-0.2, y=1.0, s=annotator.show(), transform=axs[i].transAxes, fontsize=10, fontweight='bold')


# metadata_table['Status'] = metadata_table['LateBlight'].apply(lambda x: 'LateBlight' if not str(x).startswith('0') else 'Healthy')
# metadata_table['Planting'] = metadata_table['Mode'].apply(lambda x: 'Single' if x.startswith('Single') else 'Rotation')

# metadata_table


rel_filtered_otu_table_rank = amplicon.OTUQC.normalize(otu_table_df=filtered_otu_table, method='rel')
rank = rel_filtered_otu_table_rank.mean(axis=1).sort_values(ascending=False)


annotator.reset() # reset annotator for the next figure

fig, axs = plt.subplots(2, 3, figsize=(6, 6/1.4), layout='constrained')
axs = axs.flatten()

# histogram of sequencing depth
i = 0
sns.histplot(
    data=filtered_otu_table.sum(axis=0).reset_index(name='ReadsPerSample'), 
    x='ReadsPerSample', 
    color='blue', 
    kde=True,
    ax=axs[0],
)
axs[i].text(x=-0.2, y=1.02, s=annotator.show(), transform=axs[i].transAxes, fontsize=10)
axs[i].set_xlabel('Sequencing depth', fontsize=8)
axs[i].set_ylabel('Count', fontsize=8)
axs[i].tick_params(labelsize=7)

# rank abundance curve
i = 1
sns.lineplot(
    x=range(1, len(rank) + 1),
    y=rank.values,
    ax=axs[i],
    color='green',
)
axs[i].set_xscale('log')
axs[i].set_xlabel('zOTU rank', fontsize=8)
axs[i].set_ylabel('Mean relative abundance', fontsize=8)
axs[i].text(x=-0.2, y=1.02, s=annotator.show(), transform=axs[i].transAxes, fontsize=10)
axs[i].tick_params(labelsize=7)

# rarefaction curves
i = 2
line_ax = sns.lineplot(
    data=all_rarefaction_table_df,
    x='Depth',
    y='chao1',
    hue='SampleID',
    ax=axs[i],
    legend=False,
    color='lightgray',
)
axs[i].set_ylabel('Chao1 index', fontsize=8)
axs[i].set_xlabel('Sampling depth', fontsize=8)
for line in line_ax.lines:
    line.set_color('gray')
axs[i].text(x=-0.2, y=1.02, s=annotator.show(), transform=axs[i].transAxes, fontsize=10)
axs[i].set_yscale('log')
axs[i].tick_params(labelsize=7)

# rarefaction curves with group
i = 3
ar.visualization_rarefaction_curves_with_group(
        rarefaction_table_df=all_rarefaction_table_df,
        metadata_table=metadata_table,
        group_name='Soil',
        alpha_metric_name='chao1',
        ax=axs[i],
        show_legend=True,
)
axs[i].set_ylabel('Chao1 index', fontsize=8)
axs[i].set_xlabel('Sampling depth', fontsize=8)
axs[i].text(x=-0.2, y=1.02, s=annotator.show(), transform=axs[i].transAxes, fontsize=10)
axs[i].set_yscale('log')
axs[i].tick_params(labelsize=7)
axs[i].legend(fontsize=6)

# rarefaction curves with group
i = 4
ar.visualization_rarefaction_curves_with_group(
        rarefaction_table_df=all_rarefaction_table_df,
        metadata_table=metadata_table,
        group_name='Planting',
        alpha_metric_name='chao1',
        ax=axs[i],
        show_legend=True,
)
axs[i].set_ylabel('Chao1 index', fontsize=8)
axs[i].set_xlabel('Sampling depth', fontsize=8)
axs[i].text(x=-0.2, y=1.02, s=annotator.show(), transform=axs[i].transAxes, fontsize=10)
axs[i].set_yscale('log')
axs[i].tick_params(labelsize=7)
axs[i].legend(fontsize=6)

# rarefaction curves with group
i = 5
ar.visualization_rarefaction_curves_with_group(
        rarefaction_table_df=all_rarefaction_table_df,
        metadata_table=metadata_table,
        group_name='Status',
        alpha_metric_name='chao1',
        ax=axs[i],
        show_legend=True,
)
axs[i].set_ylabel('Chao1 index', fontsize=8)
axs[i].set_xlabel('Sampling depth', fontsize=8)
axs[i].text(x=-0.2, y=1.02, s=annotator.show(), transform=axs[i].transAxes, fontsize=10)
axs[i].set_yscale('log')
axs[i].tick_params(labelsize=7)
axs[i].legend(fontsize=6)


# # Save
# fig.savefig(
#     results_dir.joinpath('004_diversity_analysis', 'rarefaction_curves', 'rarefaction_curves_summary.svg'),
#     format='svg',
# )


import subprocess


def usearch_rarefy(otu_table_path: Path, sample_sizes: list[int], randseed: int, output_dir: Path, prefix: str = 'otu_table_rarefied', run_method: str = 'joblib'):
    for sample_size in sample_sizes:
        cmds = [
            f'usearch11 -otutab_rare {otu_table_path} -sample_size {sample_size} -randseed {randseed} -output {output_dir.joinpath(f"{prefix}_{sample_size}.txt")}' for sample_size in sample_sizes
        ]

    if run_method == 'run':
        for cmd in cmds:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    elif run_method == 'popen':
        for cmd in cmds:
            subprocess.Popen(cmd, shell=True)

    elif run_method == 'concurrent':
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(subprocess.run, cmd, shell=True, check=True, capture_output=True, text=True) for cmd in cmds]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e}")

    elif run_method == 'joblib':
        from joblib import Parallel, delayed
        def run_cmd(cmd):
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        tasks = [delayed(run_cmd)(cmd) for cmd in cmds]
        parallel = Parallel(n_jobs=-1)
        parallel(tasks)

    else:
        raise ValueError(f"Invalid run_method: {run_method}. Use 'ordered' or 'parallel'.")


# Batch rarefaction with usearch, can be run in parallel with different methods.
usearch_rarefy(
    otu_table_path= results_dir.joinpath('003_all_samples_merged', 'all_samples_filtered_dereplicated_no_singleton_preorder_zotus_feature_table.txt'),
    sample_sizes=[5_000, 10_000, 20_000],
    randseed=123,
    output_dir= results_dir.joinpath('004_diversity_analysis', 'alpha'),
    prefix='otu_table_rarefied',
    # run_method='run',
    # run_method='popen',
    # run_method='concurrent',
    run_method='joblib',
)


usearch_otu_rarefied_table: pd.DataFrame = pd.read_csv(
    results_dir.joinpath('004_diversity_analysis', 'alpha', 'otu_table_rarefied_10000.txt'),
    sep='\t', index_col=0,
)
usearch_otu_rarefied_table.index.name = 'OTU_ID'
usearch_otu_rarefied_table.shape


print(usearch_otu_rarefied_table.shape)
usearch_otu_rarefied_table.head(3)


# # skbio.diversity.alpha_diversity?
skbio.diversity.get_alpha_diversity_metrics()[:5]


# define alpha diversity metrics
alpha_metric_names: List[str] = [
    'shannon', 'simpson_d',             # diversity
    'observed_features', 'chao1',       # richness
    'pielou_e', 'simpson_e',            # evenness
    'gini_index',                       # 优势物种
    'goods_coverage',                   # coverage
]

# skbio.diversity.get_alpha_diversity_metrics()
alpha_metrics: Dict[str, pd.Series] = {
    name: skbio.diversity.alpha_diversity(
        metric= name,

        counts= usearch_otu_rarefied_table.T.values,     # Sample X OTU_ID, np.ndarray
        ids= usearch_otu_rarefied_table.T.index.tolist() # SampleID

        # counts= otu_rarefied_table.T.values,     # Sample X OTU_ID, np.ndarray
        # ids= otu_rarefied_table.T.index.tolist() # SampleID
        
        # counts= otu_table.T.values,     # Sample X OTU_ID, np.ndarray
        # ids= otu_table.T.index.tolist() # SampleID
        
    ) for name in alpha_metric_names
}

# transform to DataFrame
alpha_metrics_df = pd.DataFrame(alpha_metrics)
alpha_metrics_df.index.name = 'SampleID'

alpha_metrics_df.shape


print(alpha_metrics_df.shape)
alpha_metrics_df.head(3)


metadata_table['Status'] = metadata_table['LateBlight'].apply(lambda x: 'LateBlight' if not str(x).startswith('0') else 'Healthy')
metadata_table['Planting'] = metadata_table['Mode'].apply(lambda x: 'Monoculture' if x.startswith('Single') else 'Intercropping')
metadata_table.head(3)


# merge alpha diversity with metadata.

alpha_metadata_table: pd.DataFrame = pd.merge(
    left= alpha_metrics_df,
    right= metadata_table,
    left_index= True,
    right_index= True,
    how='left',
)

print(alpha_metadata_table.shape)
alpha_metadata_table.head(3)


from statannotations.Annotator import Annotator
from itertools import combinations


def significance_annotation(value: float) -> Tuple[float, str]:
    """Convert p-value to significance annotation.
    Args:
        value (float): p-value
    Returns:
        str: significance annotation
    """
    
    if value < 0.0001:
        label = '****'
    elif value < 0.001:
        label = '***'
    elif value < 0.01:
        label = '**'
    elif value < 0.05:
        label = '*'
    else:
        label = 'ns'  # not significant

    return (f'{value:.2e}', label)


def show(ax: plt.Axes, alpha_metadata_table: pd.DataFrame, group: str, alpha_name: str, method: str = 'auto', text_format: str = 'star', show_differences: bool = True, letter: str = 'A') -> plt.Axes:
    sns.boxenplot(
        data=alpha_metadata_table,
        # x='Status',
        y=alpha_name,
        x = group,
        hue= group,
        palette='husl',
        legend=False,
        ax=ax,
    )

    if show_differences:
        cats = alpha_metadata_table[group].unique()
        pairs = list(combinations(cats, 2))
        annotator = Annotator(ax=ax, pairs=pairs, data=alpha_metadata_table, y=alpha_name, x=group, hue=group)
        if method == 'auto':
            method = 'Mann-Whitney' if len(pairs) <= 2 else 'Kruskal'
        else:
            raise ValueError(f"Invalid method: {method}. Use 'auto', 'Mann-Whitney', or 'Kruskal'.")
        
        annotator.configure(
            test=method, 
            text_format=text_format, loc='inside', fontsize=6, line_width=0.5, verbose=False, hide_non_significant=False,
        )    
        annotator.apply_and_annotate()
    ax.text(x=-0.02, y=1.02, s=letter, transform=ax.transAxes, fontsize=10)
    
    return ax



pg.kruskal(data=alpha_metadata_table, dv='shannon', between='Soil')
pg.pairwise_tests(data=alpha_metadata_table, dv='shannon', between='Soil', parametric=False, padjust='fdr_bh')

# pg.kruskal(data=alpha_metadata_table, dv='shannon', between='Breeding')
# pg.pairwise_tests(data=alpha_metadata_table, dv='shannon', between='Breeding', parametric=False, padjust='fdr_bh')


def add_significance_letters(ax: plt.Axes, group: str, metric: str, alpha_metadata_table: pd.DataFrame, offset: float = 0.2) -> None:
    al = stats.AddLetter()
    _, letter_dict = al.show_letter(df=alpha_metadata_table, group=group, metric=metric)

    xticks = ax.get_xticks()
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    group_max = alpha_metadata_table.groupby(group)[metric].max()

    for x, label in zip(xticks, xlabels):
        y = group_max[label]
        ax.text(x, y + offset, letter_dict.get(label, ''), ha='center', va='bottom', fontsize=6, color='black')

    return None 


# Show on FIgure
fig = plt.figure(figsize=(6, 6/1.4), layout='constrained')
gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, 0])
show(ax=ax, alpha_metadata_table=alpha_metadata_table, group='Soil', alpha_name='shannon', method='auto', text_format='star', letter='A', show_differences=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
add_significance_letters(ax=ax, group='Soil', metric='shannon', alpha_metadata_table=alpha_metadata_table)
ax.set_ylim([6, 8.5])

ax = fig.add_subplot(gs[0, 1])
show(ax=ax, alpha_metadata_table=alpha_metadata_table, group='Status', alpha_name='shannon', method='auto', text_format='star', letter='B')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

ax = fig.add_subplot(gs[0, 2])
show(ax=ax, alpha_metadata_table=alpha_metadata_table, group='Planting', alpha_name='shannon', method='auto', text_format='star', letter='C')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

ax = fig.add_subplot(gs[1, :])
show(ax=ax, alpha_metadata_table=alpha_metadata_table.query('Breeding != "Unclear"'), group='Breeding', alpha_name='shannon', method='auto', text_format='star', show_differences=False, letter='D')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
add_significance_letters(ax=ax, group='Breeding', metric='shannon', alpha_metadata_table=alpha_metadata_table)
ax.set_ylim([6, 8.5])


fig.savefig(
    results_dir.joinpath('004_diversity_analysis', 'alpha_differences', 'alpha_diversity_summary.svg'),
    format='svg',
)


filtered_sample_otu_table = amplicon.OTUQC.filter_otu_table_by_sample(otu_table_df=otu_table, method='iqr')
# filtered_sample_otu_table = amplicon.OTUQC.filter_otu_table_by_sample(otu_table_df=otu_table, method='abundance', min_max_threshold=(3961.4, 75284.5))
filtered_sample_otu_table.shape


filtered_otu_table: pd.DataFrame = amplicon.OTUQC.filter_otu_table_by_otu(otu_table_df=filtered_sample_otu_table, min_prevalence=0.01, min_abundance=0.000001, show=False)
filtered_otu_table.shape 


rel_otu_table: pd.DataFrame = amplicon.OTUQC.normalize(otu_table_df=filtered_otu_table, method='rel')
rel_otu_table.shape


rel_otu_table.head(3)


rel_otu_table.sum(axis=0) # check sum, should be 1.0


# skbio.diversity.beta_diversity?
# ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'mahalanobis', 'manhattan',
#  'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'unweighted_unifrac', 'weighted_unifrac', 'yule']


# Define beta diversity metrics.

beta_metric_names: List[str] = [
    'braycurtis',           # 基于相对丰度
    'jaccard',              # 基于存在/缺失
    # 'unweighted_unifrac',     # 基于存在/缺失, tree required; 关注物种组成本质差异（如新物种入侵
    # 'weighted_unifrac',       # 基于相对丰度, tree required; 若关注生态功能变化（如关键物种丰度波动对生态系统的影响
]


# skbio.diversity.beta_diversity
beta_metrics: Dict[str, skbio.stats.distance.DistanceMatrix] = {
    name: skbio.diversity.beta_diversity(
        metric= name,

        counts= rel_otu_table.T.values,     # Sample X OTU_ID, np.ndarray
        ids= rel_otu_table.T.index.tolist(), # SampleID

        **{'tree': tree, 'taxa': rel_otu_table.T.columns.tolist()} if name in ['weighted_unifrac', 'unweighted_unifrac'] else {}  # provide tree for UniFrac,
        
    ) for name in beta_metric_names
}



# save the distance matrix to csv.
for name in beta_metric_names:
    
    beta_matrix = beta_metrics[name]
    logger.info(f'Beta diversity metric: {name}, distance matrix shape: {beta_matrix.shape}')

    # DistanceMatrix to DataFrame, then save to csv.
    beta_matrix.to_data_frame().to_csv(
        results_dir.joinpath('004_diversity_analysis', 'beta', f'{name}_distance_matrix.csv'),
    )


# read 
beta_metrics: Dict[str, skbio.stats.distance.DistanceMatrix] = {
    name: skbio.stats.distance.DistanceMatrix.read(
        results_dir.joinpath('004_diversity_analysis', 'beta', f'{name}_distance_matrix.csv')
    ) for name in beta_metric_names
}


# compare the two distance matrix [braycurtis vs jaccard].
r, p, n = skbio.stats.distance.mantel(
    x = beta_metrics['braycurtis'],
    y = beta_metrics['jaccard'],
    method= 'pearson',
    permutations= 9999,
    seed = 123,
)
print(f'Mantel test between braycurtis and jaccard: r={r:.4f}, p={p:.4f}, n={n}')

# Mantel test between braycurtis and jaccard: r=0.9404, p=0.0001, n=3910


distance_matrix: skbio.stats.distance.DistanceMatrix = beta_metrics['braycurtis']


distance_pcoa = skbio.stats.ordination.pcoa(
    distance_matrix=distance_matrix,
    seed = 123,
)

type(distance_pcoa)


distance_pcoa.plot(
    df= metadata_table,
    column= 'Status',
    title= 'PCoA based on Bray-Curtis distance matrix',
    axis_labels= ('PC1', 'PC2', 'PC3'),
    cmap= 'tab20',
    # s= 50,
)
plt.grid(True)


distance_pcoa_metadata_table = (
    distance_pcoa.samples.iloc[:, :3]
    .merge(
        right=metadata_table,
        left_index=True,
        right_index=True,
        how='left')
)
distance_pcoa_metadata_table.head(3)

# distance_pcoa_metadata_table
fig, axs = plt.subplots(2, 3, layout='constrained', figsize=(6, 6/1.2))
axs = axs.flatten()
for i, group_name in enumerate(['Soil', 'Status', 'Planting', 'Breeding']):
    sns.scatterplot(
        data=distance_pcoa_metadata_table,
        x='PC1',
        y='PC2',
        hue=group_name,
        palette='husl',    
        ax= axs[i],
        s= 5,
    )
    axs[i].set_title(f'PCoA colored by {group_name}', fontsize='medium')
    axs[i].set_aspect('equal')
    axs[i].set_xlabel(f'PC1 ({distance_pcoa.proportion_explained["PC1"]*100:.2f}%)', fontsize=8)
    axs[i].set_ylabel(f'PC2 ({distance_pcoa.proportion_explained["PC2"]*100:.2f}%)', fontsize=8)
    axs[i].text(x=-0.2, y=1.05, s=chr(97 + i).upper(), transform=axs[i].transAxes, fontsize=10)
    if i == 3:
        axs[i].legend(fontsize=6, ncols=2, bbox_to_anchor=(-0.2, -0.2), loc='upper left', markerscale=1.5)
        # axs[i].legend_.remove()
    else:
        axs[i].legend(fontsize=6, markerscale=1.5)

fig.savefig(
    results_dir.joinpath('004_diversity_analysis', 'beta', 'pcoa_braycurtis_summary.svg'),
    format='svg',
)



# Distance-based statistics:
# 1. Categorical Variable Stats
def anosim(distance_matrix: skbio.stats.distance.DistanceMatrix, metadata_table: pd.DataFrame, column: str, permutations: int = 999):
    """Perform ANOSIM test on a distance matrix with respect to a categorical variable.
    Args:
        distance_matrix (skbio.stats.distance.DistanceMatrix): Distance matrix of samples.
        metadata_table (pd.DataFrame): Metadata table with sample information.
        column (str): Column name in metadata_table to group by for ANOSIM.
        permutations (int, optional): Number of permutations for significance testing. Defaults to 999.
    Returns:
        skbio.stats.distance.ANOSIMResults: Results of the ANOSIM test, including R statistic and p-value.
    """

    anosim_results = skbio.stats.distance.anosim(
        distance_matrix= distance_matrix,
        grouping= metadata_table,
        column= column,
        permutations= permutations,
    )

    return anosim_results


# ANOSIM test for each categorical variable.
for group_name in ['Soil', 'Status', 'Planting', 'Breeding']:
    print(f'ANOSIM test for {group_name}:')
    anosim_result = anosim(distance_matrix=distance_matrix, metadata_table=metadata_table, column=group_name, permutations=999)
    print(f'R statistic: {anosim_result["test statistic"]:.4f}, group numbs: {anosim_result["number of groups"]}, p-value: {anosim_result["p-value"]:.4f}\n')
    # break


# Distance-based statistics:
# 1. Categorical Variable Stats
def adonis(distance_matrix: skbio.stats.distance.DistanceMatrix, metadata_table: pd.DataFrame, column: str, permutations: int = 999):
    """Perform PERMANOVA (Adonis) test on a distance matrix with respect to a categorical variable.
    Args:
        distance_matrix (skbio.stats.distance.DistanceMatrix): Distance matrix of samples.
        metadata_table (pd.DataFrame): Metadata table with sample information.
        column (str): Column name in metadata_table to group by for PERMANOVA.
        permutations (int, optional): Number of permutations for significance testing. Defaults to 999.
    Returns:
        skbio.stats.distance.PERMANOVAResults: Results of the PERMANOVA test, including pseudo-F statistic and p-value.
    """

    adonis_results = skbio.stats.distance.permanova(
        distance_matrix= distance_matrix,
        grouping= metadata_table,
        column= column,
        permutations= permutations,
    )

    return adonis_results 


# PERMANOVA (Adonis) test for each categorical variable.
for group_name in ['Soil', 'Status', 'Planting', 'Breeding']:
    print(f'PERMANOVA (Adonis) test for {group_name}:')
    adonis_result = adonis(distance_matrix=distance_matrix, metadata_table=metadata_table, column=group_name, permutations=999)
    print(f'pseudo-F statistic: {adonis_result["test statistic"]:.4f}, group numbs: {adonis_result["number of groups"]}, p-value: {adonis_result["p-value"]:.4f}\n')


def pairwise_permanova(distance_matrix: skbio.stats.distance.DistanceMatrix, grouping: pd.DataFrame, column: str, permutations: int = 999, p_adjust_method: str = 'fdr_bh') -> pd.DataFrame:
    """Perform pairwise PERMANOVA on distance matrix.
    Princple: For each pair of groups, subset the distance matrix and grouping,
    then perform PERMANOVA on the subsetted data. Finally, adjust p-values for multiple testing.

    Args:
        distance_matrix (skbio.stats.distance.DistanceMatrix): Distance matrix.
        grouping (pd.DataFrame): Metadata table.
        column (str): Column name in metadata table for grouping.
        permutations (int, optional): Number of permutations. Defaults to 999.

    Returns:
        pd.DataFrame: Pairwise PERMANOVA results.
    """

    from itertools import combinations

    unique_groups = grouping[column].unique()
    results = []

    for group1, group2 in combinations(unique_groups, 2):
        # Subset distance matrix and grouping
        mask = grouping[column].isin([group1, group2])
        sub_dm = distance_matrix.filter(grouping[mask].index)
        sub_grouping = grouping[mask]

        # Perform PERMANOVA
        adonis_res = skbio.stats.distance.permanova(
            distance_matrix=sub_dm,
            grouping=sub_grouping,
            column=column,
            permutations=permutations,
        )
        results.append({
            'Group1': group1,
            'Group2': group2,
            "N1": sub_grouping[sub_grouping[column] == group1].shape[0],
            "N2": sub_grouping[sub_grouping[column] == group2].shape[0],
            'F-value': adonis_res['test statistic'],
            'p-value': adonis_res['p-value'],
            "permutations": permutations,
        })

    results_df = pd.DataFrame(results)

    # Adjust p-values for multiple testing
    # 多重比较校正
    from statsmodels.stats.multitest import multipletests
    if p_adjust_method == 'fdr_bh':
        adjusted = multipletests(results_df['p-value'], method='fdr_bh')
        results_df['p-adjusted'] = adjusted[1]
    elif p_adjust_method == 'bonferroni':
        adjusted = multipletests(results_df['p-value'], method='bonferroni')
        results_df['p-adjusted'] = adjusted[1]
    else:
        raise ValueError(f'Unsupported p_adjust_method: {p_adjust_method}')

    return results_df


pairwise_permanova(
    distance_matrix= distance_matrix,
    grouping= metadata_table[metadata_table.index.isin(distance_matrix.ids)],
    column= 'Soil',
    permutations= 999,
    p_adjust_method= 'fdr_bh'
)


# Distance-based statistics:
def permdisp(distance_matrix: skbio.stats.distance.DistanceMatrix, metadata_table: pd.DataFrame, column: str, permutations: int = 999, test: str = 'centroid'):
    """Perform PERMDISP (permutational analysis of multivariate dispersions) on a distance matrix with respect to a categorical variable.
    Args:
        distance_matrix (skbio.stats.distance.DistanceMatrix): Distance matrix of samples.
        metadata_table (pd.DataFrame): Metadata table with sample information.
        column (str): Column name in metadata_table to group by for PERMDISP.
        permutations (int, optional): Number of permutations for significance testing. Defaults to 999.
        test (str, optional): Method for calculating distances to centroids ('centroid' or 'median'). Defaults to 'centroid'.
    Returns:
        skbio.stats.distance.PermDispResults: Results of the PERMDISP test, including F statistic and p-value.
    """

    permdisp_results = skbio.stats.distance.permdisp(
        distance_matrix= distance_matrix,
        grouping= metadata_table,
        column= column,
        permutations= permutations,
        test = test, # or 'median'
    )

    return permdisp_results


# PERMDISP test for each categorical variable.
for group_name in ['Soil', 'Status', 'Planting', 'Breeding']:
    print(f'PERMDISP test for {group_name}:')
    permdisp_result = permdisp(distance_matrix=distance_matrix, metadata_table=metadata_table, column=group_name, permutations=999, test='centroid')
    print(f'F statistic: {permdisp_result["test statistic"]:.4f}, group numbs: {permdisp_result["number of groups"]}, p-value: {permdisp_result["p-value"]:.4f}\n')


filtered_otu_table = amplicon.OTUQC.filter_otu_table_by_sample(otu_table_df=otu_table, method='abundance', min_max_threshold=(3961.4, 75284.5))
filtered_otu_table.shape


filtered_otu_table = amplicon.OTUQC.filter_otu_table_by_otu(otu_table_df=filtered_otu_table, min_prevalence=0.01, min_abundance=0.000001, show=False)
filtered_otu_table.shape


filtered_otu_table.head(3)


# Phylum, Genus, OTU

# taxa: str = 'Phylum'
taxa: str = 'Genus'
# taxa: str = 'OTU'


# groupby at different taxa levels.
if taxa != 'OTU':
    filtered_otu_table: pd.DataFrame = (
        filtered_otu_table.merge(
            right=taxonomy_table[[taxa]],
            left_index=True,
            right_index=True,
            how='left'
        )
        .groupby(by=[taxa]).sum() # sum aggregation for each taxa
    )
    print(f'Filtered OTU table shape after {taxa} aggregation: ', filtered_otu_table.shape)
    
filtered_otu_table.head(3)


# get the Top taxas.
filtered_otu_table.sum(axis=1).sort_values(ascending=False).head(10)


# The input table for ANCOM-BC should be samples in rows and features in columns.
# without 0, so replace 0 with pseudocount calculated by multi_replace function.
ancombc_input_table: pd.DataFrame = pd.DataFrame(
    data= skbio.stats.composition.multi_replace(filtered_otu_table.T), # sample X OTU_ID, with pseudocount that replaces zeros.
    index= filtered_otu_table.T.index,
    columns= filtered_otu_table.T.columns,
)
ancombc_input_table.head(3)


ancombc_input_table.sum(axis=1) # check, sample sums are 1.0


# Calculate ANCOM-BC.
def ancombc(
    table: pd.DataFrame, 
    metadata: pd.DataFrame, 
    formula: str, 
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Perform ANCOM-BC analysis on the input table with respect to the specified formula.
    Args:
        table (pd.DataFrame): Input table with samples in rows and features in columns, with pseudocounts replacing zeros.
        metadata (pd.DataFrame): Metadata table with sample information, indexed by SampleID.
        formula (str): Formula specifying the model for ANCOM-BC, e.g., 'Group' or 'Group + Batch'.
        alpha (float, optional): Significance level for determining differentially abundant features. Defaults to 0.05.
    Returns:
        pd.DataFrame: Results of the ANCOM-BC analysis.
    """

    return skbio.stats.composition.ancombc(
        table= table,
        metadata= metadata,
        formula= formula,
        alpha= alpha
    )


# Calculate ANCOM-BC results for the specified formula.
for group_name in ['Soil', 'Status', 'Planting', 'Breeding']:
    print(f'ANCOM-BC analysis for {group_name}:')
    ancombc_results = ancombc(table=ancombc_input_table, metadata=metadata_table.loc[ancombc_input_table.index][[group_name]], formula=group_name, alpha=0.05)
    break 


# ancombc_results = ancombc(
#     table= ancombc_input_table,
#     metadata= metadata_table.loc[ancombc_input_table.index][['PartName']],
#     formula= 'PartName',
#     # grouping= 'PartName',
#     alpha= 0.05,
# )


print(ancombc_results.__len__())
ancombc_results.head()


# just tow groups.
(
    ancombc_results.query('Covariate != "Intercept" and Signif == True').sort_values(by='W', ascending=False)
    .reset_index()
)


# multiple groups pairwise comparison results

# ancombc_results[0].query('Covariate != "Intercept" and Signif == True')
# ancombc_results[1].query('Signif == True')


# Calculate structural zeros.
res_zero = skbio.stats.composition.struc_zero(
    table = filtered_otu_table.T, 
    metadata = metadata_table.loc[filtered_otu_table.T.index][['PartName', 'Soil']],
    grouping = 'PartName',
)


res_zero


# res_zero.query('Bulk == True or Rhizosphere == True')


def func(row, threshold: float = 1.0):
    '''qvalue < 0.05 and |Log2(FC)| > 1.0'''

    if row['Signif']:
        if row['Log2(FC)'] > threshold:
            return 'Enriched'
        elif row['Log2(FC)'] < -threshold:
            return 'Depleted'
        else:
            return 'Not Significant'
    else:
        return 'Not Significant'
    
sig_ancombc_results_df = ancombc_results.query('Covariate != "Intercept"')

sig_ancombc_results_df['Significance'] = sig_ancombc_results_df.apply(func, axis=1)
sig_ancombc_results_df['-log10(qvalue)'] = sig_ancombc_results_df['qvalue'].map(lambda x: -np.log10(1e-300) if x == 0 else -np.log10(x)) # avoid log10(0)
sig_ancombc_results_df['Signif'] = sig_ancombc_results_df['Signif'].astype(str)

sig_ancombc_results_df


plt.figure(figsize=(3*1.2, 3))
ax = sns.scatterplot(
    data= sig_ancombc_results_df,
    x='Log2(FC)',
    y='-log10(qvalue)',
    hue='Significance',
    palette={'Enriched': 'red', 'Depleted': 'blue', 'Not Significant': 'gray'}, # #F77189 #36ADA4
    # size='W',
)
ax.axvline(x=-1, color='gray', linestyle='--')
ax.axvline(x=1, color='gray', linestyle='--')
ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--') # significance threshold line
plt.xlabel('Log2(Fold Change)')
plt.ylabel('-Log10(q-value)')
plt.title(f'ANCOM-BC Volcano Plot at {taxa} Level')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.yscale('log')
plt.tight_layout()

# plt.savefig(
#     home_dir.joinpath('results/different_taxa/ancombc', f'{taxa}_ancombc_volcano_plot.svg'),
#     format='svg',
# )


# # save to local csv file.
# logger.info(f'Save significant ANCOM-BC results of [{taxa}] to csv file.')
# (
#     sig_ancombc_results_df
#     .sort_values(by='W', ascending=False)
#     .to_csv(home_dir.joinpath('results/different_taxa/ancombc', f'{taxa}_ancombc_diff_taxa.csv'))
# )


# # Combine all rank ancombc results.
# tem_dir = home_dir.joinpath('results/different_taxa/ancombc')

# tem_df_list: list = []
# for file in tem_dir.iterdir():
#     if file.name.endswith(('csv')):
#         print(file.name)
#         rank = file.name.split("_")[0]
#         tem_df = pd.read_csv(file)
#         tem_df['rank'] = rank
#         print(tem_df.shape)
#         tem_df_list.append(tem_df)
# combined_ancombc_df = pd.concat(tem_df_list, axis=0)
# combined_ancombc_df.to_csv(
#     tem_dir.joinpath('combined_ancombc_results_all_ranks.csv')
# )



# Normalize the filtered OTU table to relative abundance.
rel_filtered_otu_table = amplicon.OTUQC.normalize(otu_table_df=filtered_otu_table, method='rel')
# rel_filtered_otu_table.sum(axis=0) # check sum, should be 1.0
rel_filtered_otu_table.head(3)


# group: str = 'Soil'
# group: str = 'Status'
# group: str = 'Planting'
group: str = 'Breeding'


# Build LEfSe input table.
lefse_input = rel_filtered_otu_table.T.merge(
    right=metadata_table[[group]],
    left_index=True,
    right_index=True,
    how='left',
).reset_index().set_index(['index', group]).reorder_levels([group, 'index']).T

lefse_input.columns.names = [group, 'SampleID']

logger.info(f'Save LEfSe input table of [{taxa}] to txt file.')
lefse_input.to_csv(
    results_dir.joinpath('006_different_abundance', 'lefse', f'{taxa}_{group}_lefse_input.txt'),
    sep='\t',
)

lefse_input.head(3)


# group = 'Status'
# group = 'Planting'
group = 'Breeding'


# /bmp/backup/zhaosy/miniconda3/envs/lefse/lib/python3.11/site-packages/lefse/lefse_plot_res.py

RESULTS_DIR="/bmp/backup/zhaosy/ws/china_16s_pipeline/results_southwest/results/006_different_abundance/lefse"

!echo lefse_format_input.py \
    {RESULTS_DIR}/{taxa}_{group}_lefse_input.txt \
    {RESULTS_DIR}/{taxa}_{group}_lefse_input.in \
    -c 1 -u 2 -o 1000000

# -c for class (grouping variable)
# -s for subclass (optional)
# -u for subject
# -o for normalization (default 1e6) 

!echo lefse_run.py \
    {RESULTS_DIR}/{taxa}_{group}_lefse_input.in \
    {RESULTS_DIR}/{taxa}_{group}_lefse_input.res \
    -l 2
    
    # -l 3 \
    # -a 0.01 \
    # -w 0.01
 
!echo lefse_plot_res.py \
    {RESULTS_DIR}/{taxa}_{group}_lefse_input.res \
    {RESULTS_DIR}/{taxa}_{group}_lefse_input.svg \
    --format svg

# # lefse_plot_cladogram.py \
# #     /bmp/backup/zhaosy/ws/china_16s_pipeline/results/different_taxa/LEfSe/Genus_lefse_input.res \
# #     /bmp/backup/zhaosy/ws/china_16s_pipeline/results/different_taxa/LEfSe/Genus_lefse_input_cladogram.png \
# #     --format png --dpi 300



lefse_dir: Path = results_dir.joinpath('006_different_abundance', 'lefse')

lefse_df_list: list = []
for file in lefse_dir.iterdir():
    if file.name.endswith('res'):
        print(file)
        rank = file.name.split("_")[0]
        group = file.name.split("_")[1]
        lefse_df = pd.read_csv(file, sep='\t', header=None, names=['Taxa', 'log10(MaxLDAScore)', 'Class', 'LDAScore(log10)', 'p-value'])
        lefse_df['Rank'] = rank
        lefse_df['Group'] = group

        lefse_df_list.append(lefse_df)

combined_lefse_df = pd.concat(lefse_df_list, axis=0, ignore_index=True)
combined_lefse_df


fig = plt.figure(layout='constrained', figsize=(6, 6/1.4))
gs = fig.add_gridspec(1, 2)

ax0 = fig.add_subplot(gs[0])
sns.barplot(
    data= combined_lefse_df.query('`LDAScore(log10)` > 4'),
    x='Taxa',
    y='LDAScore(log10)',
    hue='Group',
    palette='husl',
    ax=ax0,
)
ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right', va='top', fontsize=7)

ax1 = fig.add_subplot(gs[1], projection='polar')
sns.barplot(
    data= combined_lefse_df.query('`LDAScore(log10)` > 4'),
    x='Taxa',
    y='LDAScore(log10)',
    hue='Group',
    palette='husl',
    ax=ax1,
)


def plot_spiral(
    data: pd.DataFrame,
    value_col: str = 'LDAScore(log10)',
    label_col: str = 'Taxa',
    group_col: str = 'Group',
    angle_range: tuple = (0, 4 * np.pi),       # 螺旋角度范围 (start, end)
    sort_by_value: bool = True,                # 是否按 value_col 降序排序
    filter_expr: str = '`LDAScore(log10)` > 4',# 数据过滤表达式（使用 query 语法）
    plot_type: str = 'line_scatter',           # 'line_scatter', 'bar', 'fill', 'scatter'
    cmap: str = 'viridis',                     # 分组或数值颜色映射
    ax=None,
    figsize: tuple = (6, 6),
    title: str = 'Spiral Plot of LDA Scores',
    show_labels: bool = False,                 # 是否显示每个点的标签（拥挤时慎用）
    label_fontsize: int = 6,
    **kwargs
):
    """
    在极坐标轴上绘制螺旋图（基于数据点的角度和半径映射）。

    参数
    ----------
    data : pd.DataFrame
        包含待映射数据的 DataFrame。
    value_col : str
        用于确定半径的数值列名（例如 LDA 得分）。
    label_col : str
        标签列名（例如 Taxa 名称），仅在 show_labels=True 时使用。
    group_col : str
        分组列名，用于颜色映射（若 plot_type 为 'line_scatter' 或 'scatter' 且无 hue 参数时）。
    angle_range : tuple
        螺旋的角度范围（弧度），例如 (0, 4*np.pi) 表示两圈。
    sort_by_value : bool
        是否按 value_col 降序排序（影响角度分配顺序）。
    filter_expr : str or None
        用 data.query() 过滤数据，例如 '`LDAScore(log10)` > 4'。若为 None 则不过滤。
    plot_type : str
        绘图类型：
        - 'line_scatter' : 连线 + 散点（默认）
        - 'scatter'      : 仅散点
        - 'bar'          : 螺旋条形图（每个点从中心向外延伸一条 bar）
        - 'fill'         : 连线并填充内部区域
    cmap : str or dict
        颜色映射。若为字符串，则按数值（value_col）或分组（group_col）映射；
        若为字典，则需提供 {group: color}。
    ax : matplotlib.axes.Axes, optional
        已有的极坐标轴。若为 None 则自动创建。
    figsize : tuple
        当 ax=None 时，新创建图形的尺寸。
    title : str
        图形标题。
    show_labels : bool
        是否在每个点旁标注 label_col 文本。
    label_fontsize : int
        标签字体大小。
    **kwargs
        传递给底层绘图函数（如 ax.plot, ax.scatter, ax.bar）的关键字参数。

    返回
    -------
    ax : matplotlib.axes.Axes
        极坐标轴对象。
    """
    # 数据预处理
    if filter_expr is not None:
        plot_df = data.query(filter_expr).copy()
    else:
        plot_df = data.copy()

    if plot_df.empty:
        raise ValueError("过滤后数据为空，请调整 filter_expr 或检查数据。")

    # 排序（决定角度顺序）
    if sort_by_value:
        plot_df = plot_df.sort_values(value_col, ascending=False)
    else:
        plot_df = plot_df.reset_index(drop=True)

    N = len(plot_df)
    theta = np.linspace(angle_range[0], angle_range[1], N)
    r = plot_df[value_col].values

    # 创建或获取轴
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='polar')
    else:
        fig = ax.figure

    # 颜色处理
    if 'c' in kwargs or 'color' in kwargs:
        # 用户指定了固定颜色
        colors = kwargs.pop('c', kwargs.pop('color', None))
    else:
        # 默认按 group_col 分类着色
        groups = plot_df[group_col].unique()
        if isinstance(cmap, str):
            palette = plt.get_cmap(cmap, len(groups))
            color_dict = {g: palette(i) for i, g in enumerate(groups)}
        elif isinstance(cmap, dict):
            color_dict = cmap
        else:
            color_dict = {g: f'C{i}' for i, g in enumerate(groups)}
        colors = [color_dict[g] for g in plot_df[group_col]]

    # 根据 plot_type 绘图
    if plot_type == 'line_scatter':
        # 连线
        ax.plot(theta, r, color='gray', linewidth=0.8, alpha=0.6, **kwargs)
        # 散点
        ax.scatter(theta, r, c=colors, s=30, alpha=0.8, edgecolors='w', linewidth=0.5, **kwargs)
    elif plot_type == 'scatter':
        ax.scatter(theta, r, c=colors, s=40, alpha=0.8, **kwargs)
    elif plot_type == 'bar':
        width = (theta[1] - theta[0]) * 0.8 if N > 1 else 0.2
        for i, (th, rad, col) in enumerate(zip(theta, r, colors)):
            ax.bar(th, rad, width=width, bottom=0, color=col, alpha=0.7, **kwargs)
    elif plot_type == 'fill':
        ax.plot(theta, r, color='gray', linewidth=1, **kwargs)
        ax.fill(theta, r, alpha=0.3, color=colors[0] if len(set(colors)) == 1 else 'skyblue', **kwargs)
    else:
        raise ValueError(f"plot_type 不支持 '{plot_type}'，可选 'line_scatter', 'scatter', 'bar', 'fill'")

    # 添加标签（可选）
    if show_labels:
        for i, (th, rad, lbl) in enumerate(zip(theta, r, plot_df[label_col])):
            ax.text(th, rad + 0.2, lbl, fontsize=label_fontsize, ha='center', va='bottom', rotation=0)

    ax.set_title(title, pad=20)
    ax.grid(True)
    return ax


# 假设 combined_lefse_df 已经准备好
fig = plt.figure(layout='constrained', figsize=(6, 6/1.4))
gs = fig.add_gridspec(1, 2)

# 左侧：普通条形图（保持不变）
ax0 = fig.add_subplot(gs[0])
sns.barplot(
    data=combined_lefse_df.query('`LDAScore(log10)` > 4'),
    x='Taxa',
    y='LDAScore(log10)',
    hue='Group',
    palette='husl',
    ax=ax0,
)
ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right', va='top', fontsize=7)

# 右侧：使用封装函数绘制螺旋图
ax1 = fig.add_subplot(gs[1], projection='polar')
plot_spiral(
    data=combined_lefse_df,
    value_col='LDAScore(log10)',
    group_col='Group',
    filter_expr='`LDAScore(log10)` > 4',
    angle_range=(0, 4 * np.pi),
    sort_by_value=True,
    plot_type='bar',   # 可改为 'bar', 'scatter', 'fill'
    cmap='viridis',                # 按 Group 着色使用 'husl' 调色板
    ax=ax1,
    title='Spiral Plot of LDA Scores'
)



# Redine taxa level here:

# taxa: str = "Phylum"
# taxa: str = "Genus"


# load significant different taxa results from ANCOM-BC
ancombc_diff_taxa_df = pd.read_csv(
    home_dir.joinpath('results/different_taxa/ancombc', f'{taxa}_ancombc_diff_taxa.csv'), 
    index_col=0
)
print(f'results of ancombc at {taxa}: {ancombc_diff_taxa_df.shape}')
ancombc_diff_taxa_df.head(3)


# Load LEfSe results
lefse_diff_taxa_df = pd.read_csv(
    home_dir.joinpath('results/different_taxa/LEfSe', f'{taxa}_lefse_input.res'), 
    sep='\t', 
    header=None, 
    names=['Taxa', 'log10(MaxLDAScore)', 'Class', 'LDAScore(log10)', 'p-value']
)
print(f'results of LEfSe at {taxa}: {lefse_diff_taxa_df.shape}')

lefse_diff_taxa_df = lefse_diff_taxa_df.dropna() # remove not significant taxa with NaN LDA score

def func(row):
    x = row['Class']
    y = row['LDAScore(log10)']
    if x == "Rhizosphere":
        return y
    elif x == "Bulk":
        return -y
    else:
        raise f"Error"

# for seaborn barplot
lefse_diff_taxa_df['LDA SCORE (log 10)'] = lefse_diff_taxa_df[['Class', 'LDAScore(log10)']].apply(func, axis=1)

print(f'significant results of LEfSe at {taxa}: {lefse_diff_taxa_df.shape}')
lefse_diff_taxa_df.head()


# visualize LEfSe results with 3.0 threshold.
threshold = 3.0
tem_df = lefse_diff_taxa_df.query('abs(`LDA SCORE (log 10)`) >= @threshold').sort_values(by='LDA SCORE (log 10)', ascending=False).copy()
print(f'Number of significant different taxa at LDA score threshold {threshold}: {tem_df.shape[0]}')

sns.barplot(
    data=tem_df,
    x='LDA SCORE (log 10)',
    # order=lefse_diff_taxa_df.sort_values(by='LDA SCORE (log 10)', ascending=False)['Taxa'],  # order of x
    y='Taxa',
    hue='Class',
    hue_order=['Bulk', 'Rhizosphere'],
    palette='husl',
)
sns.despine()
plt.tight_layout()


# |logFC| >= 1.0 and qvalue < 0.05
# ancombc_significant_diff_taxa_names: set = set(ancombc_diff_taxa_df.query('Significance != "Not Significant"').index.tolist())

# qvalue < 0.05 equal to Signif == True
ancombc_significant_diff_taxa_names: set = set(ancombc_diff_taxa_df.query('Signif').index.tolist())


# |LDA SCORE (log 10)| >= threshold and not NaN.
threshold: float = 3.5
# threshold: float = 3.0
lefse_significant_diff_taxa_names: set = set(lefse_diff_taxa_df.query('abs(`LDA SCORE (log 10)`) >= @threshold')['Taxa'].tolist())


print(f'ancombc_diff shape: {len(ancombc_significant_diff_taxa_names)}')
print(f'lefse_diff shape: {len(lefse_significant_diff_taxa_names)}')


# diff_taxa: set = ancombc_significant_diff_taxa_names.union(lefse_significant_diff_taxa_names)
diff_taxa: set = set(ancombc_significant_diff_taxa_names).intersection(set(lefse_significant_diff_taxa_names))
# diff_taxa: set = ancombc_significant_diff_taxa_names.difference(lefse_significant_diff_taxa_names)

print(f'overlap shape: {len(diff_taxa)}')
# diff_taxa


def mark_func(row):
    if row.name in diff_taxa:
        if row['Signif']:
            if row['Log2(FC)'] > 0.0:
                return 'Enriched'
            elif row['Log2(FC)'] < 0.0:
                return 'Depleted'
            else:
                return 'Not Significant'
        else:
            return 'Not Significant'
    else:
        return 'Not Significant'

# Mark overlap significance in ancombc results dataframe.
ancombc_diff_taxa_df['overlap_significance'] = ancombc_diff_taxa_df.apply(mark_func, axis=1)
ancombc_diff_taxa_df = ancombc_diff_taxa_df.sort_values(by='W', ascending=False)

print('overlap significance marked ancombc results: ')
print(ancombc_diff_taxa_df.query("overlap_significance != '''Not Significant'''").shape)
ancombc_diff_taxa_df.head(3)


# Volcano plot for overlap significant taxa.
plt.figure(figsize=(3*1.2, 3))
ax = sns.scatterplot(
    data= ancombc_diff_taxa_df,
    x='Log2(FC)',
    y='-log10(qvalue)',
    hue='overlap_significance',
    palette={'Enriched': 'red', 'Depleted': 'blue', 'Not Significant': 'gray'}, # #F77189 #36ADA4
    # size='W',
)
ax.axvline(x=-1, color='gray', linestyle='--')
ax.axvline(x=1, color='gray', linestyle='--')
ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--') # significance threshold line
plt.xlabel('Log2(Fold Change)')
plt.ylabel('-Log10(q-value)')
plt.title(f'ANCOM-BC Volcano Plot at {taxa} Level')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.yscale('log')
plt.tight_layout()

plt.savefig(
    home_dir.joinpath('results/different_taxa/lefse_and_ancombc_overlap', f'{taxa}_lefse_ancombc_overlap_diff_volcano_plot.svg'),
    format='svg',
)


tem_df = lefse_diff_taxa_df.query('Taxa in @diff_taxa').sort_values(by='LDA SCORE (log 10)', ascending=False).copy()

sns.barplot(
    data= tem_df,
    x='LDA SCORE (log 10)',
    y='Taxa',
    # order=,  # order of y
    hue='Class',
    hue_order=['Bulk', 'Rhizosphere'],
    palette='husl',
)
plt.yticks(fontsize='small')
sns.despine()
plt.tight_layout()

plt.savefig(
    home_dir.joinpath('results/different_taxa/lefse_and_ancombc_overlap', f'{taxa}_lefse_ancombc_overlap_diff_barplot.svg'),
    format='svg',
)
tem_df.to_csv(
    home_dir.joinpath('results/different_taxa/lefse_and_ancombc_overlap', f'{taxa}_lefse_ancombc_overlap_diff_taxa.csv')
)


# for scatterplot size mapping (relative abundance).
tem_rel_otu_taxa_table = (
    # rel_otu_tax_table # from compositional normalization step
    rel_filtered_otu_table # from lefse step
    .reset_index()
    .melt(
        id_vars=[taxa],
        var_name='SampleID',
        value_name='Relative Abundance'
    )
    .merge(
        right=metadata_table[['PartName']],
        left_on='SampleID',
        right_index=True,
        how='left'
    )
    .groupby(by=[taxa, 'PartName'])['Relative Abundance'].mean().reset_index()
    # .groupby(by='PartName')['Relative Abundance'].sum().reset_index()
)

def func(row, taxa):
    taxa_name = row['Taxa']
    class_name = row['Class']
    tem_subset = tem_rel_otu_taxa_table.query(f'{taxa} == @taxa_name and PartName == @class_name')
    if not tem_subset.empty:
        return tem_subset['Relative Abundance'].values[0]
    else:
        return 0.0
    
tem_df['Relative Abundance'] = tem_df[['Taxa', 'Class']].apply(
    func,
    axis=1,
    taxa=taxa,
)

tem_df.head(3)


# plt.figure(figsize=(4, 6))
sns.scatterplot(
    data= tem_df,
    x='LDA SCORE (log 10)',
    y='Taxa',
    # order=,  # order of y
    hue='Class',
    hue_order=['Bulk', 'Rhizosphere'],
    palette='husl',
    size='Relative Abundance',
    sizes=(100, 400),
)
plt.xticks(fontsize='large')
plt.yticks(fontsize='x-large')
plt.legend(fontsize='large')
plt.tight_layout()
plt.savefig(
    home_dir.joinpath('results/different_taxa/lefse_and_ancombc_overlap', f'{taxa}_lefse_ancombc_overlap_diff_scatterplot.svg'),
    format='svg',
    bbox_inches='tight',
)


overlap_dir = home_dir.joinpath('results/different_taxa/lefse_and_ancombc_overlap')
overlap_df_list: list = []
for file in overlap_dir.iterdir():
    if file.name.endswith(('csv')):
        print(file.name)
        rank = file.name.split("_")[0]
        overlap_df = pd.read_csv(file)
        overlap_df['rank'] = rank
        print(overlap_df.shape)
        overlap_df_list.append(overlap_df)
combined_overlap_df = pd.concat(overlap_df_list, axis=0)
combined_overlap_df.to_csv(overlap_dir.joinpath('combined_lefse_ancombc_overlap_results_all_ranks.csv'))


# %%bash 
!echo "without stratified version"
!echo picrust2_pipeline.py \
    -s /bmp/backup/zhaosy/ws/china_16s_pipeline/results/feature_table_from_Jiahe/asv.fa \
    -i /bmp/backup/zhaosy/ws/china_16s_pipeline/results/feature_table_from_Jiahe/asvtab.txt \
    -o /bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified \
    -p 60 --verbose

# !echo "with stratified version, 极其耗时。"
# !echo picrust2_pipeline.py \
#     -s /bmp/backup/zhaosy/ws/china_16s_pipeline/results/feature_table_from_Jiahe/asv.fa \
#     -i /bmp/backup/zhaosy/ws/china_16s_pipeline/results/feature_table_from_Jiahe/asvtab.txt \
#     -o /bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_stratified \
#     -p 60 --stratified --verbose


nsti = pd.read_csv(
    '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified/combined_marker_predicted_and_nsti.tsv.gz',
    # '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified/bac_marker_predicted_and_nsti.tsv.gz',
    # '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified/arc_marker_predicted_and_nsti.tsv.gz',
    sep='\t',
    index_col=0,
    compression='gzip'
)
print(f'Average NSTI: {nsti["metadata_NSTI"].mean():.4f}')
print(f'Ratio of samples with NSTI <= 0.15: {(nsti["metadata_NSTI"] <= 0.15).mean():.4%}')
nsti.head(3)


plt.figure(figsize=(3*1.2, 3))
sns.histplot(data=nsti, x='metadata_NSTI', kde=True)
plt.yscale('log')


# EC_predicted = pd.read_csv(
#     filepath_or_buffer='/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_01/EC_predicted.tsv.gz',
#     sep='\t',
#     index_col=0,
#     compression='gzip'
# )

# KO_predicted = pd.read_csv(
#     filepath_or_buffer='/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_01/KO_predicted.tsv.gz',
#     sep='\t',
#     index_col=0,
#     compression='gzip')

# EC_predicted.shape, KO_predicted.shape


ko_unstrat_table = pd.read_csv(
    # '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_01/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz',
    '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz',
    sep='\t',
    index_col=0,
    compression='gzip'
)

ec_unstrat_table = pd.read_csv(
    # '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_01/EC_metagenome_out/pred_metagenome_unstrat.tsv.gz',
    '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified/EC_metagenome_out/pred_metagenome_unstrat.tsv.gz',
    sep='\t',
    index_col=0,
    compression='gzip'
)

# 这个通路的功能基因总量
path_unstrat_table = pd.read_csv(
    # '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_01/pathways_out/path_abun_unstrat.tsv.gz',
    '/bmp/backup/zhaosy/ws/china_16s_pipeline/results/functional_analysis/picrust_results_263_nostratified/pathways_out/path_abun_unstrat.tsv.gz',
    sep='\t',
    index_col=0,
    compression='gzip'
)

# Pathway coverage（非常重要）: 这个通路是否完整？

ec_unstrat_table.shape, ko_unstrat_table.shape, path_unstrat_table.shape


metacyc_pathways_info = pd.read_csv(
    # home_dir.joinpath('scripts/picrust2-2.6.3/picrust2/default_files/description_mapfiles/metacyc_pathways_info.txt.gz'),
    home_dir.joinpath('scripts/picrust2-2.6.3/picrust2/default_files/description_mapfiles/metacyc-pwy_name.txt.gz'),
    header=None, names=['ID', 'Metacyc_pathway'], index_col=0,
    sep='\t', compression='gzip',
)
print(metacyc_pathways_info.shape)
metacyc_pathways_info.head(3)


from microbiome import metacyc_utils
importlib.reload(metacyc_utils)


metacyc_utils.get_metacyc_pathway_maps(
    html_file_path = home_dir.joinpath('results/functional_analysis/deseq2_diff/annotation_files/MetaCyc Pathways.html'), 
    # html_file_path = home_dir.joinpath('results/functional_analysis/deseq2_diff/annotation_files/MetaCyc_Pathways_v26011601.html'), 
    output_csv_path = home_dir.joinpath('results/functional_analysis/deseq2_diff/annotation_files/metacyc_pathway_maps.csv'), 
)


# 下载并读取 top-level mapping
map_df = pd.read_csv(
    home_dir.joinpath("results/functional_analysis/deseq2_diff/annotation_files", "metacyc_pathways_info_prokaryotes_top_level.tsv"),
    sep="\t",
    header=None,
    names=["Pathway_ID", "Top_level"],
)
print(map_df.shape)
map_df.head(3)


metacyc_path_info = (
    metacyc_pathways_info
    .reset_index()
    .merge(
        right=map_df,
        left_on='ID',
        right_on='Pathway_ID',
        how='left'
    )
)

metacyc_path_info.head(3)


ko_unstrat_table.head(3)


# ec_unstrat_table.head(3)
path_unstrat_table.head(3)


top20 = path_unstrat_table.mean(axis=1).sort_values(ascending=False).index[:20]
kegg_top20 = path_unstrat_table.loc[top20]

# replace with pathway annotations (name or top level)

# kegg_top20
sns.clustermap(
    data=kegg_top20,
    z_score=1, # normalize rows
    cmap='coolwarm',
    figsize=(7, 5),
)

plt.title('Top 20 KEGG Pathways across Samples')


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


tem_path_unstrat_table = path_unstrat_table.copy()
tem_path_unstrat_table = tem_path_unstrat_table.T
tem_path_unstrat_table[tem_path_unstrat_table.columns] = StandardScaler().fit_transform(tem_path_unstrat_table[tem_path_unstrat_table.columns])
tem_path_unstrat_table.head()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(tem_path_unstrat_table)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=tem_path_unstrat_table.index)
pca_df = pca_df.merge(metadata_table[['PartName']], left_index=True, right_index=True, how='left')

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='PartName',
    palette='husl',
)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.title('PCA of Standardized Predicted Pathway Presence/Absence')


from statsmodels.multivariate.manova import MANOVA
maov = MANOVA.from_formula('PC1 + PC2 ~ PartName', data=pca_df)
results = maov.mv_test()
print(results)


# path_unstrat_table

tem_path_unstrat_table = path_unstrat_table.copy()
tem_path_unstrat_table = (tem_path_unstrat_table > 0).astype(float)

# bc_dm = skbio.diversity.beta_diversity('braycurtis', path_unstrat_table.T.values, ids=path_unstrat_table.T.index.tolist())
dm = skbio.diversity.beta_diversity('jaccard', tem_path_unstrat_table.T.values, ids=tem_path_unstrat_table.T.index.tolist())
dm_pc = skbio.stats.ordination.pcoa(dm)
# bc_pc.proportion_explained
sns.scatterplot(
    data=dm_pc.samples.merge(
        right=metadata_table[['PartName']],
        left_index=True,
        right_index=True,
        how='left'
    ),
    x='PC1',
    y='PC2',
    hue='PartName',
    palette='husl',
)
plt.xlabel(f'PC1 ({dm_pc.proportion_explained[0] * 100:.2f}%)')
plt.ylabel(f'PC2 ({dm_pc.proportion_explained[1] * 100:.2f}%)')
plt.title('PCoA of Jaccard Distance on Predicted Pathway Presence/Absence')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

plt.savefig(
    home_dir.joinpath('results/functional_analysis/differential_analysis', 'pcoa_PartName.svg'),
    format='svg',
)


from skbio.stats.distance import permanova
permanova_results = permanova(
    distmat=dm,
    grouping=metadata_table[['PartName']],
    column='PartName',
    permutations=9999,
)


permanova_results


# 距离矩阵
D = dm.data  # n x n
n = D.shape[0]

# 分组信息
groups = metadata_table.loc[list(dm.ids), 'PartName'].values
unique_groups = np.unique(groups)

# 1. 总平方和 (Total SS)
SS_total = np.sum(D**2) / n

# 2. 组内平方和 (Within SS)
SS_within = 0
for g in unique_groups:
    idx = np.where(groups == g)[0]
    Dg = D[np.ix_(idx, idx)]
    ng = len(idx)
    SS_within += np.sum(Dg**2) / ng

# 3. 组间平方和 (Between SS)
SS_between = SS_total - SS_within

# 4. R²
R2 = SS_between / SS_total

print(f"PERMANOVA R² = {R2:.4f}")



path_unstrat_table_T = path_unstrat_table.T # SampleID X Pathway
path_unstrat_table_T.index.name = 'SampleID'
path_unstrat_table_T = path_unstrat_table_T.round().astype(int) # 取整数
path_unstrat_table_T.head()


path_unstrat_table_T_filtered_columns = path_unstrat_table_T.columns[path_unstrat_table_T.sum(axis=0) > 10]
path_unstrat_table_T_filtered = path_unstrat_table_T.loc[:, path_unstrat_table_T_filtered_columns]
path_unstrat_table_T_filtered.shape


from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
# from pydeseq2 import inference
from pydeseq2.default_inference import DefaultInference
from IPython.display import clear_output


dds = DeseqDataSet(
    counts= path_unstrat_table_T_filtered,
    metadata= metadata_table.loc[path_unstrat_table_T_filtered.index],
    design='~ PartName',                                      # singale factor
    # design='~ PartName + AreaName',                           # multiple factors
    # design='~ PartName + AreaName + PartName * AreaName',       # interaction between factors
    refit_cooks= True,                                          # 过滤异常值
)

# 标准化 + 离散度估计 + 模型拟合
dds.deseq2()

clear_output()  # 清除冗长的输出信息


dds 
# dds.var['dispersions']
# dds.varm['LFC']


ds = DeseqStats(
    dds,
    contrast= ['PartName', 'Rhizosphere', 'Bulk'], # singale factor: Comparing Rhizosphere to Bulk, Bulk is the reference level.
    # contrast= ['AreaName', 'A', 'B'], # multiple factor: Comparing Rhizosphere to Bulk, Bulk is the reference level.
    # contrast= ['AreaName', 'A', 'C'], # multiple factor: Comparing Rhizosphere to Bulk, Bulk is the reference level.
    # contrast= ['AreaName', 'B', 'C'], # multiple factor: Comparing Rhizosphere to Bulk, Bulk is the reference level.
    alpha= 0.05,
    inference=DefaultInference(n_cpus=32),
)
ds.summary()

clear_output()                  # 清除冗长的输出信息
print(ds.results_df.shape)
ds.results_df.head(3)


plt.figure(figsize=(3*1.2, 3))
ds.plot_MA(s= 20, alpha=0.7) # s for size.


# 提取结果（对比组：condition_B vs condition_A）

sig_diff_func = ds.results_df.copy()

def func(row, threshold: float):
    '''padj < 0.05 and |Log2(FC)| > 1.0'''

    if row['padj'] < 0.05:
        if row['log2FoldChange'] > threshold:
            # return 'Enriched'
            return 'Rhizosphere'
        elif row['log2FoldChange'] < -threshold:
            # return 'Depleted'
            return 'Bulk'
        else:
            return 'Not Significant'
    else:
        return 'Not Significant'
    
sig_diff_func['Significance'] = sig_diff_func.apply(func, axis=1, threshold= 0.0)                                       # threshold = 0
sig_diff_func['-log10(padj)'] = sig_diff_func['padj'].map(lambda x: -np.log10(1e-200) if x == 0 else -np.log10(x))      # avoid log10(0), set to a very small value.

print('Total functions count: ', sig_diff_func.shape)
print('Significant different functions count: ', sig_diff_func.query('Significance != "Not Significant"').shape)
sig_diff_func.head(3)


# Add annotations.
sig_diff_func_annotated = (
    sig_diff_func
    .merge(
        right=metacyc_path_info,
        left_index=True,
        right_on='ID',
        how='left'
    )
    .set_index('ID')
)
print(f'Annotated significant different functions shape: {sig_diff_func_annotated.shape}')
sig_diff_func_annotated.head(3)


sig_diff_func_annotated.sort_values(by='log2FoldChange', ascending=False, inplace=True) # sort by log2FC
sig_diff_func_annotated.head()


def show_text(ax):
    ## show top 5 enriched and top 5 depleted functions.
    rhi_enrichments = sig_diff_func_annotated.index[:5].tolist()
    bulk_enrichments = sig_diff_func_annotated.index[-5:].tolist()
    names = rhi_enrichments + bulk_enrichments

    ## show all significant different functions.
    # names = sig_diff_func_annotated.query('Significance != "Not Significant"').index.tolist()
    
    texts = []
    for name in names:
        x = sig_diff_func_annotated.loc[name, 'log2FoldChange']
        y = sig_diff_func_annotated.loc[name, '-log10(padj)']
        color = sig_diff_func_annotated.loc[name, 'Significance']
        text = name
        texts.append((x, y, text, color))
    for x, y, text, color in texts:
        if color == 'Rhizosphere':
            color = '#429992'
        elif color == 'Bulk':
            color = '#DB7D8D'
        ax.text(x, y, text, color=color, fontsize=8)


# volcano plot for significant different functions.
sns.scatterplot(
    data=sig_diff_func_annotated,
    x='log2FoldChange',
    y='-log10(padj)',
    hue='Significance',
    # palette={'Enriched': 'red', 'Depleted': 'blue', 'Not Significant': 'gray'},
    palette={'Rhizosphere': '#429992', 'Bulk': '#DB7D8D', 'Not Significant': 'gray'},
    # palette='husl',
    # hue_order=['Rhizosphere', 'Bulk', 'Not Significant'],
    # s=10,
    alpha=0.9,
)
# plt.yscale('log')
plt.xlabel('Log2(Fold Change)', fontsize='large')
plt.ylabel('-Log10(adjusted p-value)', fontsize='large')
# plt.axvline(x=-1, color='gray', linestyle='--')
# plt.axvline(x=1, color='gray', linestyle='--')
# plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--') # significance threshold line
plt.title('DESeq2 Volcano Plot for KEGG Pathways', fontsize='large')

show_text(plt.gca())

plt.tight_layout()

plt.savefig(
    home_dir.joinpath('results/functional_analysis/deseq2_diff', 'deseq2_pathway_diff_Rhizosphere_vs_Bulk.svg'),
    format='svg',
)


sig_diff_func_annotated.head(3)


# sig_diff_func_annotated.query('log2FoldChange > 1 and padj < 0.05')#.shape
# sig_diff_func_annotated.query('log2FoldChange < -1 and padj < 0.05')#.shape


# barplot for top 50 significant different functions.

tem_df = sig_diff_func_annotated.copy()

tem_df['Top'] = tem_df['log2FoldChange'].abs()
tem_df = tem_df.sort_values(by='Top', ascending=False).head(50)
# tem_df

plt.figure(figsize=(6, 10))
sns.barplot(
    data= tem_df.reset_index().sort_values(by='log2FoldChange', ascending=False),
    x = 'log2FoldChange',
    y = 'Metacyc_pathway',
    hue='Significance',
    # hue_order=['Depleted', 'Enriched'],
    palette='husl',
)
plt.savefig(
    home_dir.joinpath('results/functional_analysis/deseq2_diff', 'deseq2_pathway_diff_barplot_Rhizosphere_vs_Bulk.svg'),
    format='svg',
    bbox_inches='tight',
)


logger.info('Saved DESeq2 results.')
(
    # sig_diff_func_annotated.query('Significance != "Not Significant"')
    sig_diff_func_annotated
    .to_csv(home_dir.joinpath('results/functional_analysis/deseq2_diff', 'deseq2_pathway_diff_Rhizosphere_vs_Bulk.csv'), index=True)
    # .shape
)


# taxa_diff_func: str = 'OTU'
taxa_diff_func: str = 'Genus'


# 1️⃣ 差异 OTU / ASV（ANCOM-BC）
diff_taxa = pd.read_csv(
    # home_dir.joinpath('results/different_taxa/ancombc', f'{taxa_diff_func}_ancombc_diff_taxa.csv'),
    home_dir.joinpath('results/different_taxa/lefse_and_ancombc_overlap', f'{taxa_diff_func}_lefse_ancombc_overlap_diff_taxa.csv'),
    index_col=0,
)
diff_taxa.set_index('Taxa', inplace=True)

# 2️⃣ 差异 MetaCyc pathway（DESeq2）
diff_func = pd.read_csv(
    home_dir.joinpath('results/functional_analysis/deseq2_diff', 'deseq2_pathway_diff_Rhizosphere_vs_Bulk.csv'),
    index_col=0,
)

## get top 5 enriched and top 5 depleted functions.
diff_func = diff_func.sort_values(by='log2FoldChange', ascending=False)
top5_diff_func_name = diff_func.index[:5].tolist() + diff_func.index[-5:].tolist()
diff_func = diff_func.loc[top5_diff_func_name]


diff_taxa.shape, diff_func.shape


diff_taxa.head(3)


diff_func.head(10)


taxa_otu_table: pd.DataFrame = (
    otu_table.merge(
        right=taxonomy_table[[taxa_diff_func]],
        left_index=True,
        right_index=True,
        how='left'
    )
    .groupby(by=[taxa_diff_func]).sum() # sum aggregation for each taxa
)
print('Taxa OTU table shape after taxa aggregation: ', taxa_otu_table.shape)
taxa_otu_table.head(3)


# 3️⃣ 原始丰度矩阵（用于相关性）
diff_taxa_otu_table = taxa_otu_table.loc[diff_taxa.index]
diff_func_pathway_table = path_unstrat_table.loc[diff_func.index]

diff_taxa_otu_table.shape, diff_func_pathway_table.shape


diff_taxa_otu_table.head(3)



diff_func_pathway_table.head(3)


from scipy.stats import spearmanr


correlation_results = []
for otu_id in diff_taxa_otu_table.index:
    for pathway in diff_func_pathway_table.index:
        # Spearman correlation
        corr_coef, p_value = spearmanr(
            diff_taxa_otu_table.loc[otu_id],
            diff_func_pathway_table.loc[pathway]
        )
        correlation_results.append({
            taxa_diff_func: otu_id,
            'Pathway': pathway,
            'Spearman_rho': corr_coef,
            'p_value': p_value
        })

# Convert results to DataFrame
correlation_df = pd.DataFrame(correlation_results)
correlation_df.head()


from statsmodels.stats.multitest import multipletests


# Multiple testing correction
correlation_df['p_adjusted'] = multipletests(correlation_df['p_value'], method='fdr_bh')[1]


correlation_df.head()


sig_correlation_df = (
    # correlation_df.query('p_adjusted < 0.05 and abs(Spearman_rho) > 0.8')
    correlation_df.query('p_adjusted < 0.05 and abs(Spearman_rho) > 0.5')
)
print(f'taxa number: {sig_correlation_df[taxa_diff_func].nunique()}')
print(f'function number: {sig_correlation_df["Pathway"].nunique()}')
print(f'significant correlations shape: {sig_correlation_df.shape}')
sig_correlation_df.head()


sig_correlation_df.query('Spearman_rho < 0') # There is no negative correlation pairs.


rhi_color = '#429992'
bulk_color = '#DB7D8D'


metacyc_path_info


heat_df = (
    sig_correlation_df
    .pivot(index= taxa_diff_func, columns='Pathway', values='Spearman_rho')
    .fillna(0)
)

(
    sig_correlation_df
    .merge(
        right=metacyc_path_info,
        left_on='Pathway',
        right_on='ID',
        how='left'
    )
)
sns.clustermap(
    data=heat_df,
    cmap='coolwarm',
    # cmap='vlag',
    # figsize=(8, 12),
    center=0,
    # center=0.5,
    row_cluster=True,
    row_colors= [
        rhi_color if diff_taxa.loc[taxa_name, 'Class'] == 'Rhizosphere' else bulk_color
        for taxa_name in heat_df.index
    ],
    col_cluster=True,
    col_colors= [
        rhi_color if diff_func.loc[path_name, 'Significance'] == 'Rhizosphere' else bulk_color
        for path_name in heat_df.columns
    ],
)
plt.title("Correlation between differential taxa and pathways")
plt.savefig(
    home_dir.joinpath('results/functional_analysis/taxa_function_correlation', f'correlation_heatmap_{taxa_diff_func}_pathways.svg'), 
    format='svg',
    bbox_inches='tight',
)


import networkx as nx 
from microbiome.network import NetworkVisualizer


# G = nx.from_pandas_edgelist(
#     sig_correlation_df,
#     source=taxa_diff_func,
#     target='Pathway',
# )
G = nx.Graph()
for _, row in sig_correlation_df.iterrows():
    node1 = row[taxa_diff_func]
    node2 = row['Pathway']
    G.add_node(node1, type='taxa')
    G.add_node(node2, type='pathway')
    G.add_edge(
        node1,
        node2,
        weight= row['Spearman_rho'],
        sign= 'positive' if row['Spearman_rho'] > 0 else 'negative',
    )


nv = NetworkVisualizer()  

plt.figure(figsize=(5, 5))
# nv.show_community(
nv.show(
    G, 
    pos_style='spring',
    node_color= ['lightgreen' if node[1]['type'] == 'taxa' else 'lightblue' for node in G.nodes(data=True)],
    # node_label_fontsize='5',
    edge_width_factor=0.5,
)
plt.title(f'Correlation Network between differential {taxa_diff_func} and pathways')


pos_init = nx.bipartite_layout(
    G,
    nodes=[n for n, d in G.nodes(data=True) if d['type'] == 'taxa'],
    scale=5,
    # center=(0, 0),
    # aspect_ratio=1.333,
    align='vertical',
    # align='horizontal',
)
# pos_init = nx.spring_layout(
#     G,
#     scale=5,
#     seed=123,
# )

modules = nx.algorithms.community.greedy_modularity_communities(G)
module_dict = {}
for i, module in enumerate(modules):
    for node in module:
        module_dict[node] = i

# plt.figure(figsize=(8, 8))
nx.draw(G, pos=pos_init,
        node_size=100,
        label=True,
        node_color= [plt.cm.tab10(module_dict[n]) for n in G.nodes()],
        with_labels=True,
        font_size=6,
        edge_color= ['red' if G[u][v]['sign'] == 'positive' else 'blue' for u, v in G.edges()],
        width=[abs(G[u][v]['weight']) * 0.2 for u, v in G.edges()]
)

print(f'node number: {G.number_of_nodes()}')
print(f'edge number: {G.number_of_edges()}')

# gephi can not make bipartite layout, so we save the layout here.
nx.write_gexf(
    G,
    home_dir.joinpath('results/functional_analysis/taxa_function_correlation', f'correlation_network_{taxa_diff_func}_pathways_bipartite_layout.gexf')
)
plt.savefig(
    home_dir.joinpath('results/functional_analysis/taxa_function_correlation', f'correlation_network_{taxa_diff_func}_pathways_bipartite_layout.svg'), 
    format='svg',
    bbox_inches='tight', # save the full figure
)


taxa: str = 'Genus'


sig_correlation_df.head(3)


tem_df = (
    sig_correlation_df
    # correlation_df
    .pivot(
        index= taxa,
        columns='Pathway',
        values='Spearman_rho'
    )
    .fillna(0)
)
tem_df.head(3)
# tem_df = 0
# tem_df


import pycirclize
print('Pycirclize version:', pycirclize.__version__)


# sig_correlation_df.head(3)
# correlation_df
circos = pycirclize.Circos.chord_diagram(
    tem_df,
    space=5,
    cmap='tab20',
    # label_kws=dict(r=94, size=10, color='black'),
    # r_lim=(93, 100)
)
circos.plotfig()
# plt.show()


taxa: str = 'Genus'

key_taxa: str = 'Flavobacterium'
group_name: str = "PartName"
key_pathway: str = 'superpathway of fucose and rhamnose degradation'


taxa_otu_table: pd.DataFrame = (
    otu_table.merge(
        right=taxonomy_table[[taxa]],
        left_index=True,
        right_index=True,
        how='left'
    ).groupby(by=[taxa]).sum() # sum aggregation for each taxa
)
taxa_otu_table = amplicon.OTUQC.normalize(taxa_otu_table, method='rel')
print(taxa_otu_table.shape)
taxa_otu_table.head(3)


key_taxa_otu_group_table: pd.DataFrame = (
    taxa_otu_table.T
    .merge(
        right=metadata_table[[group_name]],
        left_index=True,
        right_index=True,
        how='left'
    )
    [[key_taxa, group_name]] # filter columns: key_taxa and group_name
)
print(key_taxa_otu_group_table.shape)
key_taxa_otu_group_table.head(3)


metacyc_path_ID = metacyc_path_info.loc[
    metacyc_path_info['Metacyc_pathway'] == key_pathway,
    'ID'
].values[0]
print(f'Pathway ID for {key_pathway}: {metacyc_path_ID}')
key_taxa_otu_group_key_pathway_table = key_taxa_otu_group_table.merge(
    right=path_unstrat_table_T[[metacyc_path_ID]],
    left_index=True,
    right_index=True,
    how='left'
)
key_taxa_otu_group_key_pathway_table


plt.figure(figsize=(7, 3))

plt.subplot(121)
sns.violinplot(
    data=key_taxa_otu_group_key_pathway_table,
    x=group_name,
    y=key_taxa,
    inner=None, # quart
    cut=0,
    scale='width',
    color='#B5B6B6',
    linewidth=0,
    alpha=0.7,
)
sns.stripplot(
    data=key_taxa_otu_group_key_pathway_table,
    x=group_name,
    y=key_taxa,
    hue=group_name,
    palette='husl',
    legend=False,
    alpha=0.3,
    jitter=0.2, # scatter width 
    # size=3,
)
log2FC = sig_ancombc_results_df.query('FeatureID == @key_taxa')['Log2(FC)'].values[0]
p_adj = sig_ancombc_results_df.query('FeatureID == @key_taxa')['qvalue'].values[0]
plt.title(f'Abundance of \n{key_taxa} \nacross {group_name} \nlog2FC: {log2FC:.2f}, q-value: {p_adj:.2e}', fontsize='medium')

plt.subplot(122)
# sns.scatterplot(
sns.regplot(
# sns.lmplot(
    data=key_taxa_otu_group_key_pathway_table,
    # data=key_taxa_otu_group_key_pathway_table.query('PartName == "Riphizosphere"'),
    y=key_taxa,
    x=metacyc_path_ID,
    order=1,
)
# r_squared = correlation_df.query(f'{taxa} == @key_taxa and Pathway == @key_pathway')['Spearman_rho'].values[0]
# p_value = correlation_df.query(f'{taxa} == @key_taxa and Pathway == @key_pathway')['p_adjusted'].values[0]
# plt.title(f'Correlation between \n{key_taxa} \nand {key_pathway}\nR2:{r_squared:.2f}, p-value:{p_value:.2e}', fontsize='medium')