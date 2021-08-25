import pandas as pd
import numpy as np
import glob
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from matplotlib import rcParams
import seaborn as sns

def legend_without_duplicate_labels(ax, markerscale = 1):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), markerscale = markerscale)

def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

# figure size in inches
rcParams['figure.figsize'] = 10,6
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Arial'
rcParams['lines.linewidth'] = 2
sns.set_context("notebook", font_scale=1.7, rc={"lines.linewidth": 1.2, 
                                                "legend.fontsize": 16})

path = r'.\results\evaluate_scalability' 
all_files = glob.glob(os.path.join(path, '*.csv'))     

df_from_each_file = (pd.read_csv(f, sep=';') for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True)
df = df.iloc[:,:16]

df.columns = [re.sub(r'(@| )','_', re.sub(r':','', str(i).lower()).strip()) for i in df.columns]

df['Model'] = df['metrics'].apply(lambda x: str(x).split('-')[0])
df['Model2'] = df['Model'].replace({'stan':'STAN',
                                   'stamp':'STAMP', 
                                   'gru4rec': 'GRU4Rec+', 
                                   'slist': 'SLIST', 
                                   'slist2': 'SLIST (new items)',
                                   'slist_ext': 'SLIST Ext'
                                   })
df['Model'] = df['Model'].replace({'stan':'STAN',
                                   'stamp':'STAMP', 
                                   'gru4rec': 'GRU4Rec+', 
                                   'slist': 'SLIST', 
                                   'slist2': 'SLIST (new items)',
                                   'slist_ext': 'SLIST Ext (new items)'
                                   })

models = ['STAN', 'STAMP', 'GRU4Rec+', 'SLIST', 'SLIST (new items)', 'SLIST Ext (new items)']
order_df = pd.DataFrame({'Model': models,
                         'order': range(1,len(models)+1)})
df = pd.merge(df, order_df, how='left')

df = df.sort_values(by=['order','perc'], ascending=True, ignore_index=True)

## MRR10 graph
fig = plt.figure(figsize=(16, 5), dpi=80)
plt.subplot(1,2,1)
ax = sns.lineplot(data=df, x='perc', y='mrr_10', hue='Model')
ax.get_legend().remove()
plt.xlabel('% of Training Sessions')
plt.ylabel('MRR@10')
# plt.savefig('./plots/mrr10_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
# plt.clf()

## HR10 graph
plt.subplot(1,2,2)
ax = sns.lineplot(data=df, x='perc', y='hitrate_10', hue='Model')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('HR@10')
ax.legend(prop={'size': 13})
# ax.legend(loc='center left', bbox_to_anchor=(1.00, 0.5), ncol=1, frameon=False).get_frame().set_edgecolor('b')
plt.savefig('./plots/mrr10_hr10_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

## Cov10 graph
fig = plt.figure(figsize=(16, 5), dpi=80)
plt.subplot(1,2,1)
ax = sns.lineplot(data=df.loc[~df.Model.isin(['SLIST (new items)','SLIST Ext (new items)'])], x='perc', y='coverage_10', hue='Model')
ax.get_legend().remove()
plt.xlabel('% of Training Sessions')
plt.ylabel('Cov@10')
# plt.savefig('./plots/cov10_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
# plt.clf()

## Pop10 graph
plt.subplot(1,2,2)
ax = sns.lineplot(data=df.loc[~df.Model.isin(['SLIST (new items)','SLIST Ext (new items)'])], x='perc', y='popularity_10', hue='Model')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('Pop@10')
ax.legend(prop={'size': 13})
plt.savefig('./plots/cov10_pop10_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

df['training_time'] = df['training_time']/60 # convert to min

## training time graph
ax = sns.lineplot(data=df.loc[df.Model !='SLIST (new items)'], x='perc', y='training_time', hue='Model2')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('Training Time (min)')
plt.savefig('./plots/trainingtime_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# getting % increase for training time
base_train = df.loc[df.perc==10,['Model','training_time']].copy()
base_train.columns = ['Model','base_training_time']
df = pd.merge(df, base_train, how='inner')
df['train_perc'] = 100*(df['training_time']/df['base_training_time']-1)

ax = sns.lineplot(data=df.loc[~df.Model.isin(['STAN','SLIST (new items)','SLIST Ext (new items)'])], x='perc', y='train_perc', hue='Model2')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('% Increase in Training Time')
plt.savefig('./plots/trainingtimeperc_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

df['testing_time_seconds'] = df['testing_time_seconds']*1000

## test time graph
ax = sns.lineplot(data=df, x='perc', y='testing_time_seconds', hue='Model')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('Prediction (ms)')
plt.savefig('./plots/predictiontime_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

df['memory_usage'] = df['memory_usage']/(1e6)

## memory usage graph
ax = sns.lineplot(data=df.loc[df.Model !='SLIST (new items)'], x='perc', y='memory_usage', hue='Model2')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('Memory (MB)')
plt.savefig('./plots/memory_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# getting % increase for memory usage
base_mem = df.loc[df.perc==10,['Model','memory_usage']].copy()
base_mem.columns = ['Model','base_mem']
df = pd.merge(df, base_mem, how='inner')
df['mem_perc'] = 100*(df['memory_usage']/df['base_mem']-1)

ax = sns.lineplot(data=df.loc[~df.Model.isin(['STAN','SLIST (new items)'])], x='perc', y='mem_perc', hue='Model2')
ax.get_legend().set_title(None)
plt.xlabel('% of Training Sessions')
plt.ylabel('% Increase in Memory')
plt.savefig('./plots/memoryperc_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

comp_df = df.loc[df.perc==100,:].copy()
comp_df = comp_df.loc[comp_df.Model != 'SLIST (new items)']

# scatter plot mrr vs training time
ax = sns.scatterplot(data=comp_df, x='mrr_10', y='training_time', hue='Model2', s=300)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('Training Time (min)')
plt.legend(markerscale=2.5)
plt.savefig('./plots/mrr_v_train_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# scatter plot hr vs training time
ax = sns.scatterplot(data=comp_df, x='hitrate_10', y='training_time', hue='Model2', s=300)
ax.get_legend().set_title(None)
plt.xlabel('HR@10')
plt.ylabel('Training Time (min)')
plt.legend(markerscale=2.5)
plt.savefig('./plots/hr_v_train_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# scatter plot mrr vs hr
ax = sns.scatterplot(data=comp_df, x='mrr_10', y='hitrate_10', hue='Model2', s=300)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('HR@10')
plt.legend(markerscale=2.5)
plt.savefig('./plots/mrr_v_hr_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots()
sub_df = df.loc[df.Model != 'SLIST (new items)'].copy()
ax = sns.scatterplot(data=sub_df.loc[df.perc!=100], x='mrr_10', y='hitrate_10', hue='Model2', s=200, alpha=0.6)
ax = sns.scatterplot(data=sub_df.loc[df.perc==100], x='mrr_10', y='hitrate_10', hue='Model2', s=700, alpha=0.8, marker='*')
legend_without_duplicate_labels(ax, markerscale=2)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('HR@10')
# plt.legend(markerscale=1.2)
plt.savefig('./plots/mrr_v_hr_scalability2.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# scatter plot mrr vs cov
ax = sns.scatterplot(data=comp_df.loc[comp_df.Model != 'SLIST Ext (new items)'], x='mrr_10', y='coverage_10', hue='Model', s=300)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('Cov@10')
plt.legend(markerscale=2.5)
plt.savefig('./plots/mrr_v_cov_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots()
sub_df = df.loc[~df.Model.isin(['SLIST Ext (new items)','SLIST (new items)'])].copy()
ax = sns.scatterplot(data=sub_df.loc[df.perc!=100], x='mrr_10', y='coverage_10', hue='Model2', s=200, alpha=0.6)
ax = sns.scatterplot(data=sub_df.loc[df.perc==100], x='mrr_10', y='coverage_10', hue='Model2', s=700, alpha=0.8, marker='*')
legend_without_duplicate_labels(ax, markerscale=2)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('Cov@10')
# plt.legend(markerscale=1.2)
plt.savefig('./plots/mrr_v_cov_scalability2.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# scatter plot mrr vs pop
ax = sns.scatterplot(data=comp_df.loc[comp_df.Model != 'SLIST Ext (new items)'], x='mrr_10', y='popularity_10', hue='Model', s=300)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('Pop@10')
plt.legend(markerscale=2.5)
plt.savefig('./plots/mrr_v_pop_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots()
sub_df = df.loc[~df.Model.isin(['SLIST Ext (new items)','SLIST (new items)'])].copy()
ax = sns.scatterplot(data=sub_df.loc[df.perc!=100], x='mrr_10', y='popularity_10', hue='Model2', s=200, alpha=0.6)
ax = sns.scatterplot(data=sub_df.loc[df.perc==100], x='mrr_10', y='popularity_10', hue='Model2', s=700, alpha=0.8, marker='*')
legend_without_duplicate_labels(ax, markerscale=2)
ax.get_legend().set_title(None)
plt.xlabel('MRR@10')
plt.ylabel('Pop@10')
# plt.legend(markerscale=1.2)
plt.savefig('./plots/mrr_v_pop_scalability2.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# scatter plot cov vs pop
fig, ax = plt.subplots()
sub_df = df.loc[~df.Model.isin(['SLIST Ext (new items)','SLIST (new items)'])].copy()
ax = sns.scatterplot(data=sub_df.loc[df.perc!=100], x='coverage_10', y='popularity_10', hue='Model2', s=200, alpha=0.6)
ax = sns.scatterplot(data=sub_df.loc[df.perc==100], x='coverage_10', y='popularity_10', hue='Model2', s=700, alpha=0.8, marker='*')
legend_without_duplicate_labels(ax, markerscale=2)
ax.get_legend().set_title(None)
plt.xlabel('Cov@10')
plt.ylabel('Pop@10')
# plt.legend(markerscale=1.2)
plt.savefig('./plots/cov_v_pop_scalability2.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()


# barplot of number of training items in test set
items = df.loc[:,['perc', 'num_training_items']].copy().drop_duplicates(ignore_index=True)
items['training_items_perc'] = 100*items['num_training_items'] / items['num_training_items'].max()
sns.lineplot(data=items, x='perc',y='training_items_perc', color='xkcd:azure')
plt.ylim((0,107))
plt.xlabel('% of Training Sessions')
plt.ylabel('% of Training Items')
plt.savefig('./plots/train_sessions_v_train_items_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# plotting number of new items in test set
new_items = [1798,984,632,435,311,238,175,146,111,95]
new_items = pd.DataFrame({'new_items': new_items, 'perc': range(10,101,10)})
ax = sns.barplot(data=new_items, x='perc', y='new_items')
plt.xlabel('% of Training Sessions')
plt.ylabel('Number of New Items in Test Set')
show_values_on_bars(ax, "v", 0.3)
plt.savefig('./plots/train_sessions_v_new_items_scalability.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()


sub_df = df.loc[df.Model == 'SLIST'].copy()
sns.lineplot(data=df.loc[~df.Model.isin(['STAN'])], x='num_training_items', y='train_perc', hue='Model')
