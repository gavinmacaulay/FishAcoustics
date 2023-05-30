# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:35:54 2023

@author: gavin
"""

import os
import numpy as np
import pandas as pd
#import sys
import xarray as xr
import numpy.ma as ma
from datetime import datetime, timedelta

from rasterio import features
import rasterio

#sys.path.append(r'C:\Users\gavin\Data - not synced\Code\python\lsss')

from ek_raw_io import RawSimradFile, SimradEOF

def korona_read_raw(f):
    # read in Korona .raw file (Simrad .raw files with extra datagrams)
    
    korona_raw = {}
    
    with RawSimradFile(str(f), "r", storage_options={}) as fid:
        while True:
            try:
                datagram = fid.read(1)
                if isinstance(datagram, bytes): # skips unparsed datagrams
                    continue
            except SimradEOF:
                break
            
            datagram["timestamp"] = np.datetime64(datagram["timestamp"].replace(tzinfo=None), "[ns]")

            if datagram['type'] not in korona_raw:
                korona_raw[datagram['type']] = [datagram]
            else:
                korona_raw[datagram['type']].append(datagram)

        fid.close()

    return korona_raw
 
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

