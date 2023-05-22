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

import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\gavin\Data - not synced\Code\python\lsss')

from ek_raw_io import RawSimradFile, SimradEOF
from pathlib import Path

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
    
    plankton_cats = xr.DataArray(0, coords=[raw_depths, raw_timestamps], dims=['depth', 'time'])
    
    # and now populate plankton_cats with the actual categories
    # this is inefficient and takes a long time to run
    for p in pid: # for each category datagram
        sample_starts = [s['start_sample_number'] for s in p['plankton_samples'].values()]
        sample_ends = np.array(sample_starts[1:]+[len(raw_depths)])
        for s, s_start, s_end in zip(p['plankton_samples'].values(), sample_starts, sample_ends):
            for d in s['plankton_data'].values(): # for each data
                mask = dict(time=p['timestamp'], depth=raw_depths[(raw_depths>=s_start)&(raw_depths<s_end)])
                if d['plankton_number'] > 0:
                    plankton_cats.loc[mask] = d['plankton_number']
    
    # now convert the depth axis from samples to metres
    plankton_cats['depth'] = raw_depths * sample_int
    
    # for each region in the file, find the plankton categories inside it
    rc = {'category': [], 'timestamp': [], 'mid_depth': [], 'area': [], 'height': [],
          'mean_Sv': []}
    
    for region in rnf:
        ids = region['border_ids']
        b = region_borders.query('id in @ids')
        
        # convert into a region mask (coordinates of timestamp and depth)
        num_depths = int(round(region['bb_height']/sample_int))
        region_mask_timestamps = b.timestamp.unique()
        region_mask_depths = np.linspace(region["bb_y"], region["bb_y"] + region["bb_height"], num_depths, endpoint=True)
        
        region_mask = xr.DataArray(True, coords=[region_mask_depths, region_mask_timestamps], 
                                      dims=['depth', 'time'])
        
        # set to false all places in the mask that are inside the region    
        for index, row in b.iterrows():
            mask = dict(time=row.timestamp, 
                        depth=region_mask_depths[(region_mask_depths>=row.start_depth)&(region_mask_depths<=row.end_depth)])
            region_mask.loc[mask] = False
        
        # now extract the region_mask bounding box of the planton categories
        rr = plankton_cats.sel(time=region_mask.time, depth=region_mask.depth, method='nearest')
        # and select out the plankton_cats where region_mask is False
        bins = np.bincount(ma.masked_array(rr, mask=region_mask).data.flatten())
        categories = np.arange(len(bins))
        
        i = np.argmax(bins)
        rc['category'].append(cat_names[categories[i]])
        rc['timestamp'].append(region['timestamp'])
        rc['mid_depth'].append(region['bb_y'] + region['bb_height']/2)
        rc['area'].append(region['area'])
        rc['height'].append(region['bb_height'])
        rc['mean_Sv'].append(region['log_mean_Sv'])
            
    region_classification = pd.DataFrame(data=rc)
    region_classification.set_index('timestamp', inplace=True)

    return region_classification

#%%
# Process the CSIRO example data

projectDir = Path(r'C:\Users\gavin\OneDrive - Aqualyd Limited\Documents\Aqualyd\Projects\2023-01 CSIRO inversion')
dataDir = projectDir/'data'/'LSSS_DATA'/'KORONA'
resultsDir = projectDir/'results'

koronaFiles = dataDir.glob('*.raw')
# for testing, select just one file
#koronaFiles=list(koronaFiles)[0:1]

pid = []
rbr = []
rnf = []
raw = []

for f in koronaFiles:
    print('Processing file ' + f.name)
    with RawSimradFile(str(f), "r", storage_options={}) as fid:
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
        
        # Now process these datagrams to filter the plankton categories from Korona
        # by the Korona detected school regions.
        
        # Need the names for the numerical categories
        cat_names = dict([(v['number'], v['legend']) for k, v in plankton_config['categories'].items()])
        cat_names[0] = 'Uncat'
        cat_names[1] = 'Other'

        # do the actual filtering. Returns a DataFrame
        rc = filter_cat_by_regions(rbr, rnf, pid, raw, cat_names)
        rc.to_csv(resultsDir/(f.stem+'.csv'))

#%%
# now report on those region category stats
cats, counts = np.unique(rc.category, return_counts=True)
plt.bar(cats, counts)
plt.ylabel('Number of regions')

rc.hist(column='mid_depth', density=False, rwidth=0.8)
plt.xlabel('Region depth [m]')
plt.ylabel('Number of regions')


