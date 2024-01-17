import contextily as cx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import MarkerCluster
from matplotlib.colors import ListedColormap, rgb2hex
from matplotlib_scalebar.scalebar import ScaleBar

from sampling_handler.misc import py_helpers


def plot_map_continous(df, column, title=None, markersize=0.1, figsize=(10,10), vmin=0, vmax=1., cbar_label=None, basemap=None):
    """Plots a continuous map of the given GeoDataFrame, using the specified column for color.

    Args:
        df (GeoDataFrame): The GeoDataFrame to plot.
        column (str): The name of the column to use for color.
        title (str, optional): The title of the plot. Defaults to None.
        markersize (float, optional): The size of the markers. Defaults to 0.1.
        figsize (tuple, optional): The size of the figure. Defaults to (10,10).
        vmin (float, optional): The minimum value for the color scale. Defaults to 0.
        vmax (float, optional): The maximum value for the color scale. Defaults to 1.
        cbar_label (str, optional): The label for the colorbar. Defaults to None.
        basemap (str, optional): The name of the basemap to use. Defaults to None.

    Returns:
        AxesSubplot: The plot axes.
    """
    #ax = sns.set_theme(style="whitegrid", palette="magma", rc={'figure.figsize': figsize})
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = df.plot(
        column,
        markersize=markersize,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cmap='magma'
    )

    # add colorbar
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    shrink_ratio = (xlims[1] - xlims[0] / ylims[1]-ylims[0])/100
    scatter = ax.collections[0]
    cbar = plt.colorbar(
        scatter,
        ax=ax,
        ticks=[val/100 for val in range(int(vmin*100), int(vmax*100), int((vmax*100 - vmin*100)/5))],
        orientation='horizontal',
        location='bottom',
        shrink=shrink_ratio*2
    )

    # Remove the border from the colorbar
    cbar.outline.set_visible(False)
    cbar.ax.set_xlabel(cbar_label if cbar_label else column)

    # add cross markers
    xticks, yticks = ax.get_xticks(), ax.get_yticks()
    [
        ax.scatter(
            lon, lat, marker='+', color='dimgrey', s=50, linewidth=0.5
        ) for lat in yticks for lon in xticks
    ]

    # rescale to original extent
    ax.axis(xmin=xlims[0],xmax=xlims[1])
    ax.axis(ymin=ylims[0],ymax=ylims[1])
    ax.grid(False)

    # set axis label
    ax.set_xlabel('Longitude', color='dimgrey')
    ax.set_ylabel('Latitude', color='dimgrey')

    # add scalebar
    ax.add_artist(
        ScaleBar(
            py_helpers.get_scalebar_distance(df),
            color='grey',
            location='lower right'
        )
    )

    # add northarrow
    ax.text(
        xlims[1]-(xlims[1]-xlims[0])/10,
        ylims[0]+(ylims[1]-ylims[0])/figsize[0]/1.25,
        u'\u2191\nN',
        horizontalalignment='center',
        verticalalignment='bottom',
        color='grey',
        size=figsize[0]*2,
    )

    sns.despine(offset=1, trim=True, ax=ax)
    if basemap:
        cx.add_basemap(ax, crs=df.crs.to_string(), source=basemap)

    rcParams = {
        'xtick.color': 'grey',   # color of the ticks
        'xtick.labelcolor': 'grey',
        'ytick.color': 'grey',   # color of the ticks
        'ytick.labelcolor': 'grey',
    }
    plt.rcParams.update(rcParams)
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')

    plt.title(title) if title else None
    fig.tight_layout()

    return ax


def plot_stratification(df, prob_column, stratum_column, bins=10, figsize=(15, 15), cmap='rocket'):
    """Plots a scatter plot and histogram of a probability column, stratified by a strata column.

    Args:
        df (pandas.DataFrame): The input dataframe.
        prob_column (str): The column name of the probability column to plot.
        stratum_column (str): The column name of the strata column to use for stratification.
        bins (int): The number of bins to use for the histogram.
        figsize (tuple): The size of the figure to be plotted.

    Returns:
        None
    """
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    # make all tick labels grey
    rcParams = {
        'text.color': 'grey',
        'xtick.color': 'grey',   # color of the ticks
        'xtick.labelcolor': 'grey',
        'ytick.color': 'grey',   # color of the ticks
        'ytick.labelcolor': 'grey',
        'axes.labelcolor': 'grey'
    }
    plt.rcParams.update(rcParams)

    ### PROBABILITY MAP###
    df.plot(prob_column, markersize=0.15, ax=ax[0, 0], cmap=cmap)
    # add colorbar
    xlims, ylims =  ax[0, 0].get_xlim(),  ax[0, 0].get_ylim()
    shrink_ratio = (xlims[1] - xlims[0] / ylims[1]-ylims[0])/100
    scatter =  ax[0, 0].collections[0]
    cbar = plt.colorbar(
        scatter,
        ax=ax[0, 0],
        shrink=shrink_ratio*2
    )

    # Remove the border from the colorbar
    cbar.outline.set_visible(False)
    #cbar.ax.set_xlabel(prob_column)

    # add cross markers
    xticks, yticks = ax[0, 0].get_xticks(), ax[0, 0].get_yticks()
    [
        ax[0, 0].scatter(
            lon, lat, marker='+', color='dimgrey', s=50, linewidth=0.5
        ) for lat in yticks for lon in xticks
    ]

    # rescale to original extent
    ax[0, 0].axis(xmin=xlims[0],xmax=xlims[1])
    ax[0, 0].axis(ymin=ylims[0],ymax=ylims[1])
    ax[0, 0].grid(False)

    # set axis label
    ax[0, 0].set_xlabel('Longitude', color='dimgrey')
    ax[0, 0].set_ylabel('Latitude', color='dimgrey')

    # add scalebar
    ax[0, 0].add_artist(
        ScaleBar(
            py_helpers.get_scalebar_distance(df),
            color='grey',
            location='lower right'
        )
    )

    # add northarrow
    ax[0, 0].text(
        xlims[1]-(xlims[1]-xlims[0])/10,
        ylims[0]+(ylims[1]-ylims[0])/10,
        u'\u2191\nN',
        horizontalalignment='center',
        verticalalignment='bottom',
        color='grey',
        size=15
    )

    sns.despine(offset=1, trim=True, ax= ax[0, 0])
    ax[0, 0].set_title('Probablity map', color='dimgrey')

    ### HISTOGRAM ###
    sns.histplot(
        df,
        x=prob_column,
        ax=ax[1, 0],
        bins=bins,
        #edgecolor='white',
        lw=0,
        alpha=0.8).set_xlim(0, 1)

    bounds = []
    for i in np.sort(df[stratum_column].unique()):
        bounds.append(df[prob_column][df[stratum_column] == i].max())

    for bound in np.sort(bounds)[:len(df[stratum_column].unique()) - 1]:
        ax[1, 0].axvline(bound, color='darkgrey', ymax=0.8, ls='--')

    cmap = sns.color_palette(cmap, as_cmap=True)
    colorlist = [cmap.colors[int(nr)] for nr in np.linspace(25, 230, len(ax[1, 0].patches))]
    for i in range(len(ax[1, 0].patches)):
        ax[1, 0].patches[i].set_facecolor(colorlist[i])

    # remove axes
    sns.stripplot(ax=ax[1, 0])
    sns.despine(ax=ax[1, 0], trim=True, offset=1)
    ax[1, 0].set_title('Probablity distribution', color='dimgrey')
    ax[1, 0].grid(False)
    ax[1, 0].set_ylabel('Nr. of Samples')
    ax[1, 0].set_xlabel('Probablity')

    ### STRATA PLOT###
    # get number of stata and create a discrete colormap
    nr_of_strata = df[stratum_column].max()
    colorlist = [cmap.colors[int(nr)] for nr in np.linspace(25, 230, nr_of_strata)]
    discrete_colors = ListedColormap(colorlist)

    # plot the data
    df.plot(stratum_column, markersize=0.15, ax=ax[0, 1], cmap=discrete_colors, classification_kwds=dict(bins=range(nr_of_strata)))

    # add colorbar
    xlims, ylims = ax[0, 1].get_xlim(), ax[0, 1].get_ylim()
    shrink_ratio = (xlims[1] - xlims[0] / ylims[1]-ylims[0])/100
    scatter = ax[0, 1].collections[0]
    cb = plt.colorbar(scatter, ax=ax[0, 1], ticks=range(1, nr_of_strata+1), shrink=shrink_ratio*2)
    # Remove the border from the colorbar
    cb.outline.set_visible(False)

    # add cross markers
    xticks, yticks = ax[0, 1].get_xticks(), ax[0, 1].get_yticks()
    [ax[0, 1].scatter(lon, lat, marker='+', color='dimgrey', s=50, linewidth=0.5) for lat in yticks for lon in xticks]

    # rescale to original extent
    ax[0, 1].axis(xmin=xlims[0],xmax=xlims[1])
    ax[0, 1].axis(ymin=ylims[0],ymax=ylims[1])
    ax[0, 1].grid(False)


    # set axis label
    ax[0, 1].set_xlabel('Longitude', color='dimgrey')
    ax[0, 1].set_ylabel('Latitude', color='dimgrey')
    ax[0, 1].set_title('Strata')

    # add scalebar
    ax[0, 1].add_artist(ScaleBar(
        py_helpers.get_scalebar_distance(df),
        color='grey',
        location='lower right'
    ))

    # add northarrow
    ax[0, 1].text(
        xlims[1]-(xlims[1]-xlims[0])/10,
        ylims[0]+(ylims[1]-ylims[0])/10,
        u'\u2191\nN',
        horizontalalignment='center',
        verticalalignment='bottom',
        color='grey',
        size=15
    )

    # remove axes
    sns.despine(ax=ax[0, 1], offset=1, trim=True)

    # strata counts
    # hist plot
    sns.countplot(x=df[stratum_column], ax=ax[1, 1], palette=discrete_colors.colors)
    sns.stripplot(ax=ax[1, 1])
    sns.despine(left=True, bottom=True, ax=ax[1, 1])
    ax[1, 1].set_title('Stratum counts')
    ax[1, 1].set_ylabel('Nr. of Samples', color='dimgrey')
    ax[1, 1].set_xlabel('Stratum', color='dimgrey')

    # move together
    fig.tight_layout()
    return ax

def interactive_map(df, popup='chg_prob', tooltips=None):

    centre = df.unary_union.convex_hull.centroid

    if tooltips:
        tooltips = folium.GeoJsonTooltip(fields=tooltips)

    # Use terrain map layer to see volcano terrain
    map_ = folium.Map(location=[centre.y, centre.x], tiles="Stamen Terrain", zoom_start=8, control_scale = True)
    #colormap = cm.LinearColormap(colors=['red','lightblue'], index=[90,100],vmin=90,vmax=100)
    cm = sns.color_palette("magma", as_cmap=True)
    def style_fn(feature):

        most_common = feature["properties"]["chg_prob"]
        ss = {
            "fillColor": rgb2hex(cm(most_common)),
            "fillOpacity": 0.1,
            "weight": 0.8,
            "color":  rgb2hex(cm(most_common)),
        }
        return ss
    marker_cluster = MarkerCluster(popups=df['chg_prob']).add_to(map_)

    for i, row in df.iterrows():

        tooltip = f"""<p><b>Sample ID</b>: {row['sampleid']}<br>
                      <b>Chg Prob</b>: {row['chg_prob']}<br>"""

        folium.Circle(
            location=[row.geometry.y, row.geometry.x],
            tooltip=tooltip,
            color=rgb2hex(cm(row.chg_prob)),
            fillColor=rgb2hex(cm(row.chg_prob)),
            fillOpacity=0.1,
            radius=70,
            weight=0.8
        ).add_to(marker_cluster)

    # Add custom basemaps to folium
    basemaps = {
        'Google Maps': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Maps',
            overlay = True,
            control = True
        ),
        'Google Satellite': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True
        ),
        'Google Terrain': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Terrain',
            overlay = True,
            control = True
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True
        ),
        'Esri Satellite': folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = True,
            control = True
        ),
        "Bing VirtualEarth": folium.TileLayer(
            tiles="http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
            attr="Microsoft",
            name="Bing VirtualEarth",
            overlay=True,
            control=True
        ),
    }

    # Add custom basemaps
    #basemaps['Google Maps'].add_to(map_)
    basemaps['Google Satellite'].add_to(map_)
    #basemaps['Bing VirtualEarth'].add_to(map_)
    basemaps['Esri Satellite'].add_to(map_)

    folium.LayerControl().add_to(map_)
    return map_
