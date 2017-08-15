import numpy as np
import pandas as pd


def get_seeds_data(data_dir):
    """Function to collect the SEEDS catalogue data and return it as two
    pandas dataframes: seeds_a_all and seeds_b_all which contain events from
    the seeds catalogue observed in COR2 images from STEREO-A and STEREO-B
    respectively. The columns of the pandas dataframe are:
    time : the time the CME appears in the COR2 field of view
    frames : the number of COR2 images in which the CME is present
    pa : the central position angle of the CME (degrees)
    width : the angular width of the CME (degrees)
    speed : the speed of the CME (km/s)
    acc : the acceleration of the CME (km/s/s)
    tm: the same as time, just renamed for compatibility
    """
    seeds_root = data_dir
    seeds_a_fn = seeds_root + r'/SEEDS_STA.csv'
    seeds_b_fn = seeds_root + r'/SEEDS_STB.csv'
    seeds_a_all = pd.read_csv(seeds_a_fn)
    seeds_b_all = pd.read_csv(seeds_b_fn)
    # Use some converter functions to get the time formats correct on import
    seeds_a_all.time = pd.to_datetime(seeds_a_all.time, dayfirst=True)
    seeds_b_all.time = pd.to_datetime(seeds_b_all.time, dayfirst=True)
    seeds_a_all['tm'] = pd.Series(seeds_a_all.time, index=seeds_a_all.index)
    seeds_b_all['tm'] = pd.Series(seeds_b_all.time, index=seeds_b_all.index)
    return seeds_a_all, seeds_b_all

    
def get_cactus_data(data_dir):
    """Function to collect the CACTus catalogue data and return it as two
    pandas dataframes: cactus_a_all and cactus_b_all which contain events from
    the seeds catalogue observed in COR2 images from STEREO-A and STEREO-B
    respectively. The columns of the pandas dataframe are:
    flow : CACTus flow number?
    time : the time the CME appears in the COR2 field of view
    liftoff_time : time CME 
    frames : the number of COR2 images in which the CME is present
    cc_pa : the central position angle of the CME (degrees)
    width : the angular width of the CME (degrees)
    speed : the speed of the CME (km/s)
    speed_variation : variation in CME speed (km/s)
    min_speed : the minimum speed of the CME in the COR2 field of view (km/s)
    max_speed : the maximum speed of the CME in the COR2 field of view (km/s)
    pa : the same as cc_pa, just renamed for compatibilty
    tm: the same as time, just renamed for compatibility
    """
    cactus_root = data_dir
    cactus_a_fn = cactus_root + r'/CACTus_STA_f.csv'
    cactus_b_fn = cactus_root + r'/CACTus_STB_f.csv'
    cactus_a_all = pd.read_csv(cactus_a_fn)
    cactus_b_all = pd.read_csv(cactus_b_fn)
    # Use some converter functions to get the time formats correct on import
    cactus_a_all.time = pd.to_datetime(cactus_a_all.time, dayfirst=True)
    cactus_b_all.time = pd.to_datetime(cactus_b_all.time, dayfirst=True)
    cactus_a_all['pa'] = pd.Series(cactus_a_all.cc_pa, index=cactus_a_all.index)
    cactus_b_all['pa'] = pd.Series(cactus_b_all.cc_pa, index=cactus_b_all.index)
    cactus_a_all['tm'] = pd.Series(cactus_a_all.time, index=cactus_a_all.index)
    cactus_b_all['tm'] = pd.Series(cactus_b_all.time, index=cactus_b_all.index)
    return cactus_a_all, cactus_b_all
    

def compare_cme_data(SSW_index, SSW_df, comp_df):
    """Takes data of a CME with SSW_index in the SSW data frame SSW_df, and
    returns the index of the nearest CME in the comparison data frame comp_df.
    """
    # Extract comparison data from dataframe
    time = SSW_df.tm[SSW_index]
    pa = SSW_df.pa[SSW_index]

    # Initialise differences as difference between CME 0 and the SS CME
    comp_index = 0
    time_diff = (time - comp_df.time[0])/np.timedelta64(1, 's')

    for i in range(1, len(comp_df)):
        # Work out differences for CME i
        new_time_diff = ((time - comp_df.time[i])/np.timedelta64(1, 's'))
        # If difference for CME i smaller replace current best differences
        if abs(new_time_diff) < abs(time_diff):
            if abs(comp_df.pa[i] - pa) < 45:
                time_diff = new_time_diff
                comp_index = i
                
    return comp_index


def match_to(cor2a, cor2b, a_all, b_all):
    """Matches SSW COR2 CMEs with CMEs in a catalogue, returning the indexes of
    the matches.
    """
    # Loop over SSW CMEs for STEREO-A
    index_list_a = range(len(cor2a))
    for i in range(len(cor2a)):
        index_list_a[i] = compare_cme_data(i, cor2a, a_all)
    # Loop over SSW CMEs for STEREO-B
    index_list_b = range(len(cor2b))
    for i in range(len(cor2b)):
        index_list_b[i] = compare_cme_data(i, cor2b, b_all)
    return index_list_a, index_list_b
    
    
def get_cme_data(index_list, df):
    """Extracts CME data from pandas dataframe df for set indexes.
    """
    # Initialse output
    output = range(0, len(index_list))
    # Set outut to be correct values from dataframe
    for i in output:
        output[i] = df[index_list[i]]
    return output


def extract_match_data(index_list_a, index_list_b, a_all, b_all):
    """Extracts CME data from pandas dataframe df for set indexes.
    """    
    # Extract data for matched events
    a_pa = get_cme_data(index_list_a, a_all.pa)
    b_pa = get_cme_data(index_list_b, b_all.pa)
    a_wid = get_cme_data(index_list_a, a_all.width)
    b_wid = get_cme_data(index_list_b, b_all.width)
    a_sp = get_cme_data(index_list_a, a_all.speed)
    b_sp = get_cme_data(index_list_b, b_all.speed)
    a_tb = get_cme_data(index_list_a, a_all.tm)
    b_tb = get_cme_data(index_list_b, b_all.tm)
    # Create a Pandas dataframe
    dfa = pd.DataFrame({'pa': a_pa, 'wid': a_wid, 'speed': a_sp, 'tb': a_tb})
    dfb = pd.DataFrame({'pa': b_pa, 'wid': b_wid, 'speed': b_sp, 'tb': b_tb})
    return dfa, dfb    

    
def check_match_times(SSW_times, comp_times, value=False):
    """Checks whether the matched events are reasonable matches to the SSW
    events by checking that they are within 12 hours of each other. If value is
    set to True the function will print the index of any events in the SSW
    data frame which do not match.
    """
    j = 0
    # Loop through time differences to check
    for i in range(0, len(SSW_times)):
        # Allow 12 hr window
        if abs((comp_times[i]-SSW_times[i])/np.timedelta64(1, 's')) > 43200:
            j = j+1
            if value == True:
                print i
                print SSW_times[i]
    return j

    
def check_match_pas(SSW_pas, comp_pas, value=False):
    """Checks whether the matched events are reasonable matches to the SSW
    events by checking that they are within 45 degrees of each other. Position
    angles must be calculated anticlockwise from solar north. If value is
    set to True the function will print the index of any events in the SSW
    data frame which do not match.
    """
    j = 0
    # Loop through time differences to check
    for i in range(0, len(SSW_pas)):
        # Allow 12 hr window
        if abs(comp_pas[i]-SSW_pas[i]) > 45:
            j = j+1
            if value == True:
                print i
    return j

def get_matched_lists(c2a, c2b, data_dir):
    """Function to take the list of SSW CMEs and return pandas dataframes of
    the closest events in the SEEDS and CACtus catalogues. The output
    dataframes have the following columns:
    pa : the position angle of the CME (degrees)
    speed : the speed of the CME (km/s)
    tb : the appearance time of the CME in the COR2 field of view
    wid : the angular extent of the CME (degrees)
    """
    seeds_a_all, seeds_b_all = get_seeds_data(data_dir)
    index_list_a, index_list_b = match_to(c2a, c2b, seeds_a_all, seeds_b_all)
    seeds_a, seeds_b = extract_match_data(index_list_a, index_list_b, seeds_a_all, seeds_b_all)
    cactus_a_all, cactus_b_all = get_cactus_data(data_dir)
    index_list_a, index_list_b = match_to(c2a, c2b, cactus_a_all, cactus_b_all)
    cactus_a, cactus_b = extract_match_data(index_list_a, index_list_b, cactus_a_all, cactus_b_all)
    return seeds_a, seeds_b, cactus_a, cactus_b
    