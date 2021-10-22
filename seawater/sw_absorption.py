# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:31:04 2021

@author: gavin
"""
import numpy as np

def sw_absorption(f, S, T, D, method, pH):
    """ 
    function alpha = sw_absorption(f, S, T, D, method, pH)
    
    Returns the absorption coefficient [dB/km] for
    the given acoustic frequency (f [kHz]), salinity
    (S, [ppt]), temperature (T, [degC]), depth
    (D, [m]), and optionally pH.
    
    Note that the salinity units are ppt, not the more
    modern psu. See the comment on page 2 of Doonan et al.
    for a discussion of this. For most purposes, using psu
    values in place of ppt will make no difference to the
    resulting alpha value.
    
    Will work correctly with vector inputs for S, T,
    and D.
    
    By default function implements the formula given in
    Doonan, I.J. Coombs, R.F. McClatchie, S. (2003). 
    The absorption of sound in seawater in relation to 
    estimation of deep-water fish biomass. ICES 
    Journal of Marine Science 60: 1-9.
     
    Note that the paper has two errors in the formulae given
    in the conclusions.
    
    Argument 'method', allows the user to specify the absorption formula to use.
    'doonan'  - Doonan et al (2003)
    'fandg'   - Francois & Garrison (1982)
    'fands'   - Fisher & Simmonds(1977)
    'aandm'   - Ainslie & McColm (1998)
    
    % Written by Gavin Macaulay, August 2003
    % $Id$
    """
    
    # if nargin < 6
    #     pH = 8.1
    #     if mean(S) < 10:
    #         pH = 7
    #     end
    #     print(f'Assuming pH of {pH}')
    # end
    
    # if isempty(T) || isempty(D) || isempty(S)
    #     disp('The length of the T, D, and S parameters must be greater than 0')
    #     alpha = []
    #     return
    # end
    
    error = 0
    # if ((~isempty(find(f < 10, 1))) || (~isempty(find(f > 120, 1)))) ...
    #     && (nargin < 5 || strcmp(method,'doonan'))
    #     disp('The formula is only valid for frequencies between 10 and 120 kHz.')
    #     disp('Try the Francois & Garrison formula')
    #     error = 1
    # end
    
    # if (max(T) > 20) && (nargin < 5 || strcmp(method,'doonan'))
    #     disp('The formula is only valid for temperatures less than 20 deg C')
    #     disp('Try the Francois & Garrison formula')
    #     error = 1
    # end
    
    if error == 1:
        return
       
    if method == 'doonan':
        # Doonan et al, 2003
        c = 1412. + 3.21*T + 1.19*S + 0.0167*D
        A2 = 22.19 * S * (1.0 + 0.017 * T)
        f2 = 1.8 * np.power(10.,(7.0 - 1518./(T+273.)))
        P2 = np.exp(-1.76e-4*D)
        A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T*T - 1.5e-8*T*T*T
        P3 = 1.0 - 3.83e-5*D + 4.9e-10*D*D
        alpha = A2*P2*f2*f*f/(f2*f2 + f*f) / c + A3*P3*f*f
    elif method == 'fandg':
        # Francois & Garrison, 1982
        c = 1412. + 3.21*T + 1.19*S + 0.0167*D
        A1 = 8.86 * np.power(10.,0.78*pH - 5.) / c
        A2 = 21.44 * S * (1 + 0.025 * T) / c
        A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T*T - 1.5e-8*T*T*T
        # These need to be an array, so ensure that...
        T = np.array(T)
        A3 = np.array(A3)
        
        T_mask = T > 20.0
        T_t = T[T_mask]
        A3[T_mask] = 3.964e-4 - 1.146e-5*T_t + 1.45e-7*T_t*T_t - 6.5e-10*T_t*T_t*T_t
        f1 = 2.8 * np.sqrt(S/35.) * np.power(10., 4. - 1245./(T+273))
        f2 = (8.17 * np.power(10., 8. - 1990/(T+273.))) / (1.+0.0018*(S-35.))
        P2 = 1.0 - 1.37e-4*D + 6.2e-9*D*D
        P3 = 1.0 - 3.83e-5*D + 4.9e-10*D*D
        alpha = f*f * (A1*f1/(f1*f1 + f*f) + A2*P2*f2/(f2*f2 + f*f) + A3*P3)
    elif method == 'fands':
        # Fisher & Simmonds, 1977
        f = f *1e3
        P_atm = D/0.9809 # equation uses pressure in atmospheres. Approx conversion to pressure.
        T_C = T
        f1 = 1320.*(T_C + 273.15)*np.exp(-1700./(T_C + 273.15))
        f2 = (1.55e+7)*(T_C + 273.15)*np.exp(-3052./(T_C + 273.15))
        A = (8.95e-8)*(1. + .023*T_C - (5.1e-4)*T_C*T_C)
        B = (4.88e-7)*(1. + .013*T_C) * (1. - (.9e-3)*P_atm)
        C = (4.76e-13)*(1. - .04*T_C + (5.9e-4)*T_C*T_C)*(1. - (3.8e-4)*P_atm)
        a = (A*f1*(f*f))/(f1*f1 + f*f) + (B*f2*(f*f))/(f2*f2 + f*f) +  C*(f*f)
        alpha = a*1.0e3
    elif method == 'aandm':
        # Ainslie & McColm, 1998
        z = D/1.0e3 
        f1 = 0.78*np.sqrt(S/35.) * np.exp(T/26.)
        f2 = 42.*np.exp(T/17.)
        alpha = (0.106 * (f1*f*f/(f*f+f1*f1))*np.exp((pH-8.)/0.56) +
            0.52*(1.+T/43.)*(S/35.)*(f2*f*f)/(f*f+f2*f2)*np.exp(z/6.) +
            0.00049*f*f*np.exp(-(T/27+z/17.)))
    
    return alpha
