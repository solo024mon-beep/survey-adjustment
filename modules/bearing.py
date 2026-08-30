import math

def dd2dms(dd:int):
    """
    Parameters:
    ----------
    dd: decimal degrees(float or int)

    Returns:
    -------
    prints {d}° {m}’ {s}″

    Returns: d,m,s
    """
    d = int(dd)
    m = int((dd-d)*60)
    s = (((dd-d)*60) - m)* 60
    return d,m,s

def dms_to_decimal(value:str):
    """

    Parameters:
    ----------
    Value:
        'd m s'

    Returns:
    -------
    decimal degrees(float or int)
    """
    c = list(map(float,value.split(' ')))
    degrees,mnt,sec = c[0], c[1],c[2]
    decimal_degrees = degrees + (mnt / 60) + (sec / 3600)
    return decimal_degrees

def dif_DMS(obs_bear, cal_bear):
    """finding difference in string bearings"""
    dif = dms_to_decimal(obs_bear) - dms_to_decimal(cal_bear)
    return dd2dms(dif)

