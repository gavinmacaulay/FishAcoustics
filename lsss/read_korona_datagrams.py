# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:30:43 2023

@author: gavin
"""

import os
from collections import defaultdict
from datetime import datetime as dt
import numpy as np
import pandas as pd
import sys
import xarray as xr
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon, Patch
from datetime import datetime, timedelta

from rasterio import features
import rasterio
import shapely

sys.path.append(r'C:\Users\gavin\Data - not synced\Code\python\lsss')

from ek_raw_io import RawSimradFile, SimradEOF
from pathlib import Path

projectDir = Path(r'C:\Users\gavin\OneDrive - Aqualyd Limited\Documents\Aqualyd\Projects\2023-01 CSIRO inversion')
dataDir = projectDir/'data'/'LSSS_DATA'/'KORONA'
resultsDir = projectDir/'results'


# Take the korona datagrams and filter the plankton categories by korona regions
def filter_cat_by_regions(rbr, rnf, pid, raw, cat_names):
    # flatten the rbr datagrams into a DataFrame
    region_borders = []
    for r in rbr:
        for b in r['border_infos'].values():
            region_borders.append((r['timestamp'], b['id'], b['start_depth'], b['end_depth']))
    
    region_borders = pd.DataFrame(region_borders, columns=['timestamp', 'id', 'start_depth', 'end_depth'])
    
    # make a 2d xarray of the plankton categories. This is for the entire file.
    raw_timestamps = [r['timestamp'] for r in raw]
    raw_timestamps = np.unique(raw_timestamps)
    raw_depths = np.arange(raw[0]['count'])
    sample_int = raw[0]['sample_interval'] * raw[0]['sound_velocity']/2 # [m]
    
    # 2D array to store plankton categories - covers entire raw file at sample resolution
    pc = xr.DataArray(0, coords=[raw_depths, raw_timestamps], dims=['depth', 'time'])
    
    # and now populate plankton_cats with the actual categories
    # this is inefficient and takes a long time to run
    print('    Creating category dataset')
    for p in pid: # for each category datagram
        time_i = np.argwhere(pc.time.data==p['timestamp'])[0][0]
        sample_starts = [s['start_sample_number'] for s in p['plankton_samples'].values()]
        sample_ends = np.array(sample_starts[1:]+[len(raw_depths)])
        # store categries for the current ping in a temporary vector
        ping_data = np.full(raw_depths.shape, 0)
        for s, s_start, s_end in zip(p['plankton_samples'].values(), sample_starts, sample_ends):
            for d in s['plankton_data'].values(): # for each data
                if d['plankton_number'] > 0:
                    ping_data[s_start:s_end] = d['plankton_number']
        # and then update the xarray once per ping. MUCH faster than operating 
        # directly on the xarray
        pc[:, time_i] = ping_data

    # now convert the depth axis from samples to metres
    pc['depth'] = raw_depths * sample_int
    
    # for each region in the file, find the plankton categories inside it
    rc = {'category': [], 'timestamp': [], 'mid_depth': [], 'area': [], 'height': [],
          'mean_Sv': [], 'width': []}
    
    region_polygon = {v:[] for v in cat_names.values()}
    
    print(f'    Filtering by regions ({len(rnf)})')
    for region in rnf:
        ids = region['border_ids']
        b = region_borders.query('id in @ids')
        
        # convert into a region mask (coordinates of timestamp and depth)
        num_depths = int(round(region['bb_height']/sample_int))
        region_mask_timestamps = b.timestamp.unique()
        region_mask_depths = np.linspace(region["bb_y"], region["bb_y"] + region["bb_height"], num_depths, endpoint=True)
        
        region_mask = np.full((len(region_mask_depths), len(region_mask_timestamps)), True)
        
        # set to false all places in the mask that are inside the region    
        for index, row in b.iterrows():
            depth_i = (region_mask_depths>=row.start_depth)&(region_mask_depths<=row.end_depth)
            time_i = region_mask_timestamps == row.timestamp
            region_mask[depth_i, time_i] = False
        
        # now extract the region_mask bounding box of the planton categories
        rr = pc.sel(time=region_mask_timestamps, depth=region_mask_depths, method='nearest')
        # and select out the pc where region_mask is False
        bins = np.bincount(ma.masked_array(rr, mask=region_mask).data.flatten())
        categories = np.arange(len(bins))
        
        i = np.argmax(bins)
        rc['category'].append(cat_names[categories[i]])
        rc['timestamp'].append(region['timestamp'])
        rc['mid_depth'].append(region['bb_y'] + region['bb_height']/2)
        rc['area'].append(region['area'])
        rc['height'].append(region['bb_height'])
        rc['mean_Sv'].append(region['log_mean_Sv'])
        rc['width'].append(region['bb_width'])
        
        # generate a polygon version of the region and tag with category
        time_axis = region_mask_timestamps
        time_axis = np.append(time_axis, time_axis[-1]+np.diff(time_axis[-2:]))
        
        depth_axis = region_mask_depths
        depth_axis = np.append(depth_axis, depth_axis[-1]+np.diff(depth_axis[-2:]))
        
        if (region_mask.shape[0] > 2) and (region_mask.shape[1] > 2):
            for shape, value in features.shapes(region_mask.astype(np.int16), mask=(region_mask<1), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
                outer = np.array(shape['coordinates'][0])
                outer[:,0] = time_axis[outer[:,0].astype('int')]
                outer[:,1] = depth_axis[outer[:,1].astype('int')]
                region_polygon[rc['category'][-1]].append(outer)
            
    region_classification = pd.DataFrame(data=rc)
    region_classification.set_index('timestamp', inplace=True)

    return region_classification, region_polygon

#%%
# Process the CSIRO example data

koronaFiles = dataDir.glob('*.raw')
# for testing, select some files
koronaFiles=list(koronaFiles)[0:]

import logging
logger = logging.getLogger('')
logger.setLevel(logging.ERROR)


for f in koronaFiles:
    print('Processing file ' + f.name)
    
    with RawSimradFile(str(f), "r", storage_options={}) as fid:
        pid = []
        rbr = []
        rnf = []
        raw = []

        print('    Reading file')

        config_datagram = fid.read(1)
        config_datagram["timestamp"] = np.datetime64(config_datagram["timestamp"].replace(tzinfo=None), "[ns]")
       
        while True:
            try:
                datagram = fid.read(1)
                if isinstance(datagram, bytes):
                    continue
            except SimradEOF:
                break
            datagram["timestamp"] = np.datetime64(datagram["timestamp"].replace(tzinfo=None), "[ns]")

            if datagram['type'] == 'PIC0':
                plankton_config = datagram
            if datagram['type'] == 'PID0':
                pid.append(datagram)
            if datagram['type'] == 'RBR0':
                rbr.append(datagram)
            if datagram['type'] == 'RNF0':
                rnf.append(datagram)
            if datagram['type'] == 'RAW0':
                raw.append(datagram)
        fid.close()
        print(f'    File has {len(raw)} RAW datagrams')
        
        # Now process these datagrams to filter the plankton categories from Korona
        # by the Korona detected school regions.
        
        # Need the names for the numerical categories
        cat_names = dict([(v['number'], v['legend']) for k, v in plankton_config['categories'].items()])
        cat_names[0] = 'Uncat'
        cat_names[1] = 'Other'

        # do the actual filtering. Returns a DataFrame
        rc, rp = filter_cat_by_regions(rbr, rnf, pid, raw, cat_names)
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

