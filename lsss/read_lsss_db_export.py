# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:15:16 2021

@author: gavin
"""

import xml.etree.ElementTree as ET
import datetime as dt

import pandas as pd
import numpy as np

def read_lsss_db_export(format, filename, acocat):
    """
    acocat is always converted to a string
    format should be integer

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
    
    if format == 20:
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
        
        acocat = str(acocat)
        
        acolist = root[6]
        
        categories = []
        for c in acolist:
            categories.append(c.attrib['acocat'])
        print(f'File contains these acoustic categories: {categories}')
        if acocat not in categories:
            print(f'Provided acocat ({acocat}) is not in this xml file!')
            
        for n in root.iter('distance'): # loop over each integration interval
            thickness = float(n[2].text)
            distance = float(n[1].text) # [nmi]
            start_time = dt.datetime.strptime(n.attrib['start_time'], '%Y-%m-%d %H:%M:%S')
            #print(f'Reading data at ping time of {start_time}')
            start_lat = float(n[5].text)
            start_lon = float(n[7].text)
            ch = n[9]
            if ch.attrib['freq'] == '38000':                
                channel_start_depth = float(ch[5].text)
                num_channels = int(ch[1].text)
                sA_values = np.zeros((num_channels,))
                channel_time.append(start_time)
                channel_lat.append(start_lat)
                channel_lon.append(start_lon)
                channel_dist.append(distance)
                channel_thickness.append(thickness)
                channel_acocat.append(acocat)
                if (ch[11].attrib['type'] == 'P'): # pelagic data
                    for sA in ch[11]: # loop over all acoustic categories
                        if sA.attrib['acocat']  == acocat: # if we find the specified acoustic category
                            for child in sA: # loop over all sa values
                                channel_id = int(child.attrib['ch'])
                                sA_values[channel_id-1] = float(child.text)
                channel_sA.append(sA_values)
                channel_sA_sum.append(sA_values.sum())
 
    
        # combine into a DataFrame 
        r = pd.DataFrame(zip(channel_time, channel_lat, channel_lon, channel_sA_sum, channel_sA, channel_acocat, channel_dist, channel_thickness), 
                         columns=['time', 'lat', 'lon', 'sA_sum', 'sA', 'acocat', 'distance', 'thickness'])
    return r
