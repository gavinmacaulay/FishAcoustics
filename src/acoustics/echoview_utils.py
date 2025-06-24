"""Code useful for working with Echoview output files."""

import pandas as pd


def read_evl(evlfile: str, tz: str = None) -> pd.DataFrame:
    """Read in an Echoview .evl file.

    Parameters
    ----------
    evlfile :
        The Echoview .evl file path.
    tz :
        The Echoview .evl file doesn't include a timezone, so use this if you want
        times with timezones. A value of None returns naive timestamps.

        Apply the given time zone to the timestamp derived from the times in the .evl file.
        Accepts [tzinfo identifiers](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
        Use 'Etc/GMT+x' for specifying a timezone in hours relative to UTC. Note that the sign of
        x is reversed to what you would normally expect (e.g, NZ time is 'Etc/GMT-12')

    Returns
    -------
    :
        Pandas DataFrame with columns corresponding to the columns in the .evl file and
        an index of timestamp (combining the date and time columns in the .evl file).
    """
    evl = pd.read_csv(evlfile, delimiter=' ', skiprows=2, names=['date', 'time', 'depth', 'status'],
                      dtype={'date': str, 'time': str})

    evl.index = pd.to_datetime(evl['date']+'T'+evl['time']+'00', format='%Y%m%dT%H%M%S%f')
    evl.index = evl.index.tz_localize(tz, ambiguous='NaT')

    return evl
