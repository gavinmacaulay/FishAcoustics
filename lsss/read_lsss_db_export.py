# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:15:16 2021

@author: gavin
"""

import xml.etree.ElementTree as ET
import datetime as dt

import pandas as pd
import numpy as np

def read_lsss_db_export(lsss_format, filename, acocat, frequency):
    """
    acocat is always converted to a string
    format should be integer
    
    frequency should be integer in Hz

    Parameters
    ----------
    format : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if lsss_format == 20:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        channel_time = []
        channel_lat = []
        channel_lon = []
        channel_sA = []
        channel_sA_sum = []
        channel_acocat = []
        channel_dist = []
        channel_thickness = []
        channel_min_depth = []
        channel_max_depth = []
        
        acocat = str(acocat)
        frequency = str(frequency)
        
        acolist = root.find('acocat_list')
        
        categories = []
        for c in acolist:
            categories.append(c.attrib['acocat'])
        print(f'File contains these acoustic categories: {categories}')
        if acocat not in categories:
            print(f'Provided acocat ({acocat}) is not in this xml file!')
            
        for n in root.iter('distance'): # loop over each integration interval
            thickness = float(n.find('pel_ch_thickness').text)
            distance = float(n.find('integrator_dist').text) # [nmi]
            start_time = dt.datetime.strptime(n.attrib['start_time'], '%Y-%m-%d %H:%M:%S')
            #print(f'Reading data at ping time of {start_time}')
            start_lat = float(n.find('lat_start').text)
            start_lon = float(n.find('lon_start').text)
            
            ch = n.find(f"./frequency/[@freq='{frequency}']")
            if ch: 
                num_channels = int(ch.find('num_pel_ch').text)
                sA_values = np.zeros((num_channels,))
                channel_time.append(start_time)
                channel_lat.append(start_lat)
                channel_lon.append(start_lon)
                channel_dist.append(distance)
                channel_thickness.append(thickness)
                channel_acocat.append(acocat)
                channel_min_depth.append(float(ch.find('upper_interpret_depth').text))
                channel_max_depth.append(float(ch.find('lower_interpret_depth').text))
                channel_type = ch.find('ch_type')
                if (channel_type.attrib['type'] == 'P'): # pelagic data
                    for sA in channel_type: # loop over all acoustic categories
                        if sA.attrib['acocat']  == acocat: # if we find the specified acoustic category
                            for child in sA: # loop over all sa values
                                channel_id = int(child.attrib['ch'])
                                sA_values[channel_id-1] = float(child.text)
                channel_sA.append(sA_values)
                channel_sA_sum.append(sA_values.sum())
 
    
        # combine into a DataFrame 
        r = pd.DataFrame(zip(channel_time, channel_lat, channel_lon, channel_sA_sum, 
                             channel_sA, channel_acocat, channel_dist, channel_thickness,
                             channel_min_depth, channel_max_depth), 
                         columns=['time', 'lat', 'lon', 'sA_sum', 'sA', 'acocat', 
                                  'distance', 'thickness', 'channel_min_depth',
                                  'channel_max_depth'])
    return r
