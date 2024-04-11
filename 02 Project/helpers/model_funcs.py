'''Modeling Functions

This module contains functions related to creating and evaluating the models.

Copyright 2024, Brian Keith
All Rights
'''
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report

PALETTE = [
    '#66829a',
    '#003057',
    '#a9abac',
    '#54585A',
    '#BFB37C',
    '#857437',
]
PALETTE_CMAP = [
    '#66829a',
    # '#003057',
    '#a9abac',
    # '#54585A',
    '#BFB37C',
    # '#857437',
]
gt_cmap = LinearSegmentedColormap.from_list("custom_cmap", [hex2color(color) for color in PALETTE_CMAP])
PARTICIPANT_CLASS_MAP = {
    'A': 'AtRisk',
    'C': 'Healthy',
    'F': 'AtRisk',
}

def add_labels(epoch_df, participant_df, tmp_encoder):
    info_cols = ['participant_id','Group', 'Class']
    df = epoch_df.copy()
    dfp = participant_df.copy()
    
    df = df.merge(dfp[info_cols], on='participant_id')
    df = df[info_cols + [ col for col in df.columns if col not in info_cols]]
    # df = df[[col for col in df.columns if not col.endswith('psd')]]

    df.insert(
        df.columns.get_loc('Group') + 1,
        'Group_encoded',
        tmp_encoder.fit_transform(df['Group'])
        )
    
    return df

def create_train_test(df, mode = 'rbp', scaling = True, test_size = 0.2, random_state = 903027850):
    if mode == 'both':
        features = [col for col in df.columns if col.endswith('rbp') or col.endswith('psd')]
    elif mode == 'rbp':
        features = [col for col in df.columns if col.endswith('rbp')]
    elif mode == 'psd':
        features = [col for col in df.columns if col.endswith('psd')]
    else:
        raise ValueError('Invalid mode. Choose from: both, rbp, psd') 
    
    if scaling:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])

    xtrain, xtest, ytrain, ytest = train_test_split(
        scaled_features if scaling else df[features],
        df['Group_encoded'],
        test_size=test_size,
        random_state=random_state
    )
    #need to add get the ids of the ytrain test split so they cna be used later to get the aggregate classification
    IDtrain, IDtest = train_test_split(df['participant_id'].values, test_size=test_size, random_state=random_state)
    
    return xtrain, xtest, ytrain, ytest, IDtrain, IDtest

def score_model(test_ids, y_preds, participant_data, tmp_encoder, verbose = False):
    if len(set(y_preds)) == 3:
        pred_cat = 'Group'
        
    else:
        # raise ValueError('Invalid number of unique prediction values. Must be 3.')
        print('WARNING: Invalid number of unique prediction values. NONE WILL BE RETURNED FOR THE DICT.')
        return {
            'Predictions': None,
            'TruthValues': None,
            'PredictedValues': None,
            'ConfusionMatrix': None,
            'Accuracy_Group': None,
            'Accuracy_Class': None,
            'Scores': None
        }
        
    df_pred = pd.DataFrame({
        'participant_id': test_ids,
        'predictions': y_preds,
        })
    df_pred = df_pred.pivot_table(
        index='participant_id', 
        columns='predictions', 
        aggfunc='size', 
        fill_value=0,
        ).reset_index()
    df_pred.columns.name = None
    df_pred['predicted_group'] = df_pred[tmp_encoder.classes_].apply(lambda x: x.idxmax(), axis=1)
    df_pred['true_group'] = df_pred.merge(participant_data[['participant_id', pred_cat]], on='participant_id')[pred_cat]
    df_pred['predicted_class'] = df_pred['predicted_group'].map(PARTICIPANT_CLASS_MAP)
    df_pred['true_class'] = df_pred['true_group'].map(PARTICIPANT_CLASS_MAP)
    
    truths = df_pred['true_group'].values
    preds = df_pred['predicted_group'].values
    truths_class = df_pred['true_class'].values
    preds_class = df_pred['predicted_class'].values
    
    conf_matrix = pd.crosstab(truths, preds, rownames=['True'], colnames=['Predicted'], margins=False)
    scores_df = pd.DataFrame({
        pred_cat: tmp_encoder.classes_,
        'precision': precision_score(truths, preds, average=None, labels=tmp_encoder.classes_),
        'recall': recall_score(truths, preds, average=None, labels=tmp_encoder.classes_),
        'f1': f1_score(truths, preds, average=None, labels=tmp_encoder.classes_)
    })
    
    acc = accuracy_score(truths, preds)
    acc_class = accuracy_score(truths_class, preds_class)
    
    if verbose:
        display(df_pred)
        display(conf_matrix)
        print("Accuracy (Group):", acc)
        print("Accuracy (Class):", acc_class)
        display(scores_df)
    
    results_dict = {
        'Predictions': df_pred,
        'TruthValues': truths,
        'PredictedValues': preds,
        'ConfusionMatrix': conf_matrix,
        'Accuracy_Group': acc,
        'Accuracy_Class': acc_class,
        'Scores': scores_df
    }
    
    return results_dict

def generate_heatmap(truths, predicts, tmp_encoder, accs: list = [], title: str = 'Classifications of Participants'):
    '''Gets around issue with seaborn heatmap not displaying properly. https://github.com/microsoft/vscode-jupyter/issues/14363'''
    if len(set(predicts)) == 3:
        pred_cat = 'Group'
    else:
        raise ValueError('Invalid number of unique prediction values. Must be 3.')
    
    cm_vis = confusion_matrix(truths, predicts, labels=tmp_encoder.classes_)
    cm_vis_norm = cm_vis.astype('float') / cm_vis.sum(axis=1)[:, np.newaxis]
    
    def gen_plot(matrix, ax, pct = False):
        cax = ax.matshow(matrix, cmap=gt_cmap)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
        for (i, j), value in np.ndenumerate(matrix):
            if pct:
                ax.text(j, i, f'{value:.1%}', ha='center', va='center', color='black')
            else:
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', color='black')

        ax.set_xlabel('Predicted Labels', labelpad=5)
        ax.set_ylabel('True Labels', labelpad=5)
        
        ax.set_xticks(np.arange(len(tmp_encoder.classes_)))
        ax.set_yticks(np.arange(len(tmp_encoder.classes_)))
        ax.set_xticklabels(tmp_encoder.classes_)
        ax.set_yticklabels(tmp_encoder.classes_)
        
        ax.xaxis.set_ticks_position('bottom')

        ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-.5, len(tmp_encoder.classes_), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(tmp_encoder.classes_), 1), minor=True)

        ax.set_axisbelow(False)
        
        if pct:
            ax.set_title('% of Participants', pad=10, fontweight='bold')
        else:
            ax.set_title('# of Participants', pad=10,fontweight='bold')

    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{title}', fontsize=14, fontweight='bold', x = 0.57, y = 1.05)
    gen_plot(cm_vis, axs[0])
    gen_plot(cm_vis_norm, axs[1], pct = True)
    
    fig.patch.set_facecolor('#f2f2f2')
    plt.subplots_adjust(wspace=-0.15)
    
    if accs:
        accs_str = f'Accuracy (Group): {accs[0]:.1%}\nAccuracy (Class): {accs[1]:.1%}'
        fig.text(0.93, 0.98, accs_str, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.2'))
    
    # plt.tight_layout()
    #supress showing the plot and just return the figure
    plt.close()
    
    return fig
