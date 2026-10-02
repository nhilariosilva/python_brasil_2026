import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

def bar_plot(df, variable_name, ax = None):
    # Tabela de frequências absolutas
    tab = pd.crosstab(index = df[variable_name], columns = 'frequência')
    tab_normalize = pd.crosstab(index = df[variable_name], columns = 'frequência', normalize='columns')

    if(ax is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
    
    # Supondo que 'tab' seja uma tabela ou DataFrame que já foi definido
    tab_normalize.plot.bar(
        color='skyblue',
        legend=False,
        ax = ax
    )
    title = "Gráfico de barras - {}".format(variable_name)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(ax.get_xticks())
    ax.tick_params(axis='x', rotation=0, labelsize = 12)
    
    # Faz o gráfico de barras por proporções, mas exibe as contagem em cima de cada barra
    for container in ax.containers:
        for rect in container.get_children():
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, int(height * len(df[variable_name])), ha='center', va='bottom')

    ax.set_ylabel('Frequência', fontsize=12)
    ax.set_xlabel('Categorias', fontsize=12)

def sector_plot(df, variable_name, ax = None):
    # Configuração estética do Seaborn
    sns.set(style="whitegrid")
    tab = pd.crosstab(index = df[variable_name], columns = 'frequência')
    tab_series = tab['frequência']
    
    # Cria o gráfico de donut
    # plt.figure(figsize=(15, 7), dpi=80)
    if(ax is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12,6))
    
    # Cria o gráfico de donut
    tab_series.plot.pie(
        autopct = lambda pct: f'{pct:.0f}%',  # Adiciona os percentuais
        startangle = 90,                      # Inicia o gráfico a partir de 90 graus
        colors = sns.color_palette("pastel"), # Utiliza uma paleta de cores do Seaborn
        textprops = {'fontsize': 10},         # Ajusta o tamanho da fonte dos textos
        wedgeprops = {'width': 0.4},          # Ajusta a largura das fatias para criar o efeito donut
        pctdistance = 0.85,                   # Ajusta a distância dos percentuais
        ax = ax
    )
    
    # Adiciona um círculo branco no centro para criar o efeito donut
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    title = "Gráfico de setores - {}".format(variable_name)
    ax.set_title(title)

def histogram_plot(df, variable_name, ax1 = None, ax2 = None):
    if(ax1 is None and ax2 is None):
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 10]})
        ax1 = ax[0]
        ax2 = ax[1]
    # Boxplots
    sns.boxplot(x = df[variable_name], ax = ax1)
    # Histogramas
    sns.histplot(df[variable_name], kde = True, ax = ax2)

def pair_plot(df, variables, hue=None, regplot=True):
    if hue is not None:
        df = df.loc[:, variables + [hue]].copy()
        df[hue] = df[hue].astype(str)
        hue_order = np.sort(df[hue].unique())
    else:
        df = df.loc[:, variables].copy()

    def corrfunc(x, y, hue, **kws):
        if hue is not None:
            num_classes = len(hue_order) # Usa a ordem fixa
            color_palette = sns.color_palette().as_hex()
            
            corrs = []
            for class_value in hue_order:
                # Trata casos em que a classe pode estar ausente naquele subset
                mask = hue == class_value
                if mask.sum() > 1:
                    corr = pd.DataFrame({"x": x[mask], "y": y[mask]}).corr().iloc[0,1]
                else:
                    corr = np.nan
                corrs.append(corr)
                
            corr_xy = pd.DataFrame({"x": x, "y": y}).corr().iloc[0,1]
            ax = plt.gca()
            
            y_positions = np.linspace(0, 1, num_classes + 2)[1:-1]
            
            for j in range(len(corrs)):
                ax.annotate("{:.2f}".format(corrs[j]), xy=(0.3, y_positions[j]), xycoords=ax.transAxes,
                            ha='center', va='center', fontsize=18, color=color_palette[j])
            
            ax.annotate("({:.2f})".format(corr_xy), xy=(0.75, 0.5), xycoords=ax.transAxes,
                        ha='center', va='center', fontsize=18, color="black")
        else:
            ax = plt.gca()
            corr_xy = pd.DataFrame({"x": x, "y": y}).corr().iloc[0,1]
            ax.annotate("{:.2f}".format(corr_xy), xy=(0.5, 0.5), xycoords=ax.transAxes,
                        ha='center', va='center', fontsize=18, color="black")

    def regplot_hue(x, y, hue, **kwargs):
        ax = plt.gca()
        color_palette = sns.color_palette().as_hex()
        for i, class_value in enumerate(hue_order):
            xp = x[hue == class_value]
            yp = y[hue == class_value]
            if len(xp) > 1: # Prevenção de erro caso o grupo seja muito pequeno
                sns.regplot(x=xp, y=yp, scatter=False, 
                            line_kws={"color": color_palette[i]}, ax=ax)

    if hue is None:
        g = sns.PairGrid(df, diag_sharey=False)
        g.map_lower(sns.scatterplot, data=df, alpha=0.7)
        g.map_lower(sns.regplot, scatter=False, color="black")
    else:
        # hue_order garante que o Seaborn use a mesma ordem das suas funções
        g = sns.PairGrid(df, hue=hue, hue_order=hue_order, diag_sharey=False)
        g.map_lower(sns.scatterplot, alpha=0.7)
        g.map_lower(regplot_hue)

    g.map_diag(sns.histplot, kde=True)
    g.map_upper(corrfunc)

    if hue is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=sns.color_palette()[i], markersize=10, 
                               label=species)
                   for i, species in enumerate(hue_order)]
        
        plt.legend(handles=handles, title=hue, loc="upper left", bbox_to_anchor=(1, 1))

    plt.show()

def pair_plot_3d(df, x_var, y_var, z_var, hue=None):
    fig = px.scatter_3d(
        df, 
        x = x_var, 
        y = y_var, 
        z = z_var, 
        color = hue,
        opacity = 0.8, # Transparência ajuda a ver pontos que estão "atrás" dos outros
        # Escolhendo uma paleta de cores forte que fica boa em projetores
        color_discrete_sequence = px.colors.qualitative.Set1 
    )
    
    # Ajustes finos de layout para fins didáticos (fundo claro, grades nítidas)
    fig.update_layout(
        scene=dict(
            xaxis_title = x_var,
            yaxis_title = y_var,
            zaxis_title = z_var,
            # Fundo branco nas paredes do gráfico 3D fica mais limpo visualmente
            xaxis=dict(backgroundcolor = "white", gridcolor = "lightgray", showbackground = True),
            yaxis=dict(backgroundcolor = "white", gridcolor = "lightgray", showbackground = True),
            zaxis=dict(backgroundcolor = "white", gridcolor = "lightgray", showbackground = True),
        ),
        # Reduz as margens para o gráfico ocupar mais espaço na tela
        margin=dict(l = 0, r = 0, b = 0, t = 50), 
        title=f"Visão 3D: {x_var}, {y_var} e {z_var}",
        font=dict(size = 12) # Aumenta um pouco a fonte base
    )

    return fig