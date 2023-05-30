# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:30:43 2023

@author: gavin
"""

import os
import numpy as np
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon, Patch
from datetime import timedelta

sys.path.append(r'C:\Users\gavin\Data - not synced\Code\python\lsss')

from korona_raw_read import korona_read_raw, filter_cat_by_regions
from pathlib import Path

import logging
logger = logging.getLogger('')
logger.setLevel(logging.ERROR)

projectDir = Path(r'C:\Users\gavin\OneDrive - Aqualyd Limited\Documents\Aqualyd\Projects\2023-01 CSIRO inversion')
dataDir = projectDir/'data'/'LSSS_DATA'/'KORONA'
resultsDir = projectDir/'results'

#%%
# Process the CSIRO example data

koronaFiles = dataDir.glob('*.raw')
# for testing, select some files
koronaFiles=list(koronaFiles)[0:1]

for f in koronaFiles:
    print('Processing file ' + f.name)

    dg = korona_read_raw(f)
    print(f'    File has {len(dg["RAW0"])} RAW datagrams')
        
    # Now process these datagrams to filter the plankton categories from Korona
    # by the Korona detected school regions.
    
    # Need the names for the numerical categories
    cat_names = dict([(v['number'], v['legend']) for k, v in dg['PIC0'][0]['categories'].items()])
    cat_names[0] = 'Uncat'
    cat_names[1] = 'Other'

    # do the actual filtering. Returns a DataFrame
    rc, rp = filter_cat_by_regions(dg['RBR0'], dg['RNF0'], dg['PID0'], dg['RAW0'], cat_names)
    rc.to_csv(resultsDir/(f.stem+'.csv'))
    np.save(resultsDir/(f.stem+'-polygons.npy'), rp)

#%%

# load in all the .csv results files into one DataFrame and create some plots

resultsFiles = resultsDir.glob('*-korona.csv')
rc_full = []
for r in resultsFiles:
    print('Loaded ' + r.name)
    rc_full.append(pd.read_csv(r, index_col='timestamp', parse_dates=True))
    
rc_full = pd.concat(rc_full)
rc_full['timeofday'] = [(t+timedelta(hours=10)).replace(year=2012,month=1,day=1) for t in rc_full.index]

# plots of region size distributions

for i, c in enumerate(rc_full.category.unique()):
    fig, axs = plt.subplots(1,4, layout='constrained')
    rc_full.query('category == @c').hist(column='height', bins=25, ax=axs[0])
    rc_full.query('category == @c').hist(column='width', bins=25, ax=axs[1])
    rc_full.query('category == @c').hist(column='area', bins=25, ax=axs[2])
    rc_full.query('category == @c').plot(x='width', y='height', ax=axs[3], kind='scatter')
    fig.suptitle(c)
    plt.close(fig)
    fig.savefig(resultsDir.joinpath(f'Figure_region_stats_{c}.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)

# exclude small regions. the above histograms were used to set the values here:
rc = rc_full.query('(category == "FBCyl"  and height > 2) or'
                   '(category == "FluidS" and height > 2) or'
                   '(category == "Gas") or'
                   '(category == "Hard"   and height > 2 and width > 5) or'
                   '(category == "Other"  and height > 1 and width > 4) or'
                   '(category == "Uncat"  and height > 2 and width > 50)')
#rc = rc_full

minSv = np.round(rc.mean_Sv).unique().min()
maxSv = np.round(rc.mean_Sv).unique().max()
cmap = mpl.cm.plasma

# group the plots by year/month
group_fmt = '%Y-%m'
groups = rc.index.strftime(group_fmt).unique()


# do a separate figure for each category
cats = rc.category.unique()
for c in cats:
    print(f'Doing category {c}')
    fig, axs = plt.subplots(5,5, sharey=True, sharex=True, 
                            layout='constrained', gridspec_kw={'wspace':0.00, 'hspace': 0.00})
    for ax, group in zip(axs.flat, groups):
        chunk = rc[(rc.index.strftime(group_fmt) == group) & (rc.category == c)]
        rcc = chunk
        sc = ax.scatter(rcc.timeofday, rcc.mid_depth, rcc.area, 
                        vmin=minSv, vmax=maxSv, cmap=cmap,
                        c=rcc.mean_Sv)
        ax.annotate(f'{group}', 
                     xy=(0.02, 0.99), xycoords='axes fraction', ha='left',
                     va='top', fontsize=6)
    
    # Adjust the labelling
    for ax in axs[-1,:]:
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    for ax in axs[:,1:].flat:
        ax.tick_params(axis='y', which='both', left=False, right=False)

    for ax in axs[:-1,:].flat:
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
            
    # colorbar for colour of circles
    norm = mpl.colors.Normalize(vmin=minSv, vmax=maxSv)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs[0,-1], orientation='vertical', label='$S_v$')
    cbar.ax.tick_params(labelsize='xx-small')
    # legend for size of circles
    #handles, labels = sc.legend_elements(prop="sizes")
    #fig.legend(handles, labels, loc='outside center right', title='Area')
    
    fig.supylabel('Range [m]', fontsize=10)
    fig.supxlabel('Time of day (UTC+10:00)', fontsize=10)
    fig.suptitle(c)
    plt.close(fig)
    fig.savefig(resultsDir/f'Figure_bubbles_{c}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)   

#%%
# and a category echogram of the entire dataset, split into months

#colours = {v:f'C{i+1}' for i, v in enumerate(cat_names.values())}
colours={'Hard':'C1','Gas':'C2', 'FluidS':'C3', 'FBCyl':'C4', 'Uncat':'C5', 'Other':'C6'}

rpFiles = resultsDir.glob('*-korona-polygons.npy')
for i, rpf in enumerate(rpFiles):
    print('Loading ' + rpf.name)
    rp = np.load(rpf, allow_pickle=True)
    rp = rp[()]

    fig, ax = plt.subplots(1, layout='constrained')
    for cat,regions in rp.items():
        cc = colours[cat]
        for r in regions:
            t, d = zip(*r)
            d = 91 - np.array(d) # convert to depth below surface
            t = mdates.date2num(np.array(t, dtype='datetime64[ns]'))
            #ax.plot(t, d, c=cc)
            ax.plot(t[0],d[0],c=cc) # avoids a bug where add_patch doesn't update the axis limits to fit the data
            
            polygon = Polygon((np.array((t,d))).T, closed=True, 
                              edgecolor=cc, facecolor=cc)
            ax.add_patch(polygon)

    patches = [Patch(color=c, label=l) for l, c in colours.items()]
    plt.legend(patches, colours.keys(), loc='center left', 
               bbox_to_anchor=(1.04,0.5), frameon=False)
    
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.invert_yaxis()
    ax.set_ylabel('Depth [m]')
    fig.savefig(resultsDir/f'Figure_categorised_echogram_{i+1:02d}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)   
    plt.close(fig)

