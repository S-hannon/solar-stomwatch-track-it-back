import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import scipy.ndimage as ndi
import sunpy.sun.constants as spconsts
from skimage import measure
from astropy.time import Time
plt.switch_backend('Agg')

# Constants needed
rsun = spconsts.radius.to('km')


def load_cor1_data(data_file):
    """ This is a function to import the Solar Stormwatch Identifications from
    made using the cor1 Coronagraphs on the STEREO-A and STEREO-B spacecraft.
    The input to the function is the path of the cor1 data file. The raw data
    are imported into a pandas dataframe with the following
    fields:
    craft : 'A' for STEREO-A, 'B' for STEREo-B
    assets : solar stormwatch asset name,
    zid   : zooniverse id,
    tb    : Time of entry into cor1 (in Julian Days (JD))
    tm    : Time of reaching mid point of cor1 (in JD),
    pa   : CME location (counterclockwise angle from image north, degrees),
    edg   : CME edge (clockwise angle of lower edge from image north, degs),
    wid   : CME width (in degrees),
    dt    : timing difference between entry and midpoint
    """
    # Get some lists for binning the cor1 data into.
    craft = []
    assets = []
    zids = []
    cor1_bgn =[]
    cor1_mid =[]
    cor1_loc = []
    cor1_wid = []
    cor1_edg = []
    time_err = 0
    area_err = 0
    # Have to go over the file line-by-line as it has non-standard use of two
    # delimiters in the text file.
    with open(data_file) as f:
        lines = f.readlines();
        # Remove unhelpful whitespace and end of line terminators.
        for l in lines:
            elements = l[:-1].replace(' ','').split(';')
            # Use asset tag to work out if STA or STB
            a_test = elements[0].split('_')
            if a_test[1] == 'cor1a.flv':
                craft.append('A')
            elif a_test[1] == 'cor1b.flv':
                craft.append('B')
            # Append asset name, Zooniverse ID
            assets.append(elements[0])
            zids.append(int(elements[1]))
            # Append appearance time.
            if elements[3] == '\\N':
                cor1_bgn.append('NaN')
                time_err += 1
            else:
                cor1_bgn.append(float(elements[3]))
            # Append time at middle of field-of-view
            if elements[4] == '\\N':
                cor1_mid.append('NaN')
                time_err += 1
            else:
                cor1_mid.append(float(elements[4]))
            # Append direction/location and width. Remember the location is
            # given as clockwise angle from solar north. This needs to be
            # converted to position angle (counter-clockwise angle from solar-N)
            if elements[5] == '\\N':
                area_err += 1
                cor1_wid.append('NaN')
                cor1_edg.append('NaN')
                cor1_loc.append('NaN')
            else:
                x = [float(i) for i in elements[5].split(',')]
                cor1_wid.append(x[0])
                cor1_edg.append(x[1])
                cor1_loc.append(x[1] + (x[0]/2))
    # Convert the Lists of cor1 data into a pandas dataframe
    C = pd.DataFrame({'assets' : assets, 'zid' : zids, 'tb' : cor1_bgn,
                      'tm' : cor1_mid, 'pa' : cor1_loc, 'wid' : cor1_wid,
                      'edg' : cor1_edg, 'craft' : craft})
    # A few of the datatypes need setting to floats.
    C['tb'] = C['tb'].astype('float')
    C['tm'] = C['tm'].astype('float')
    C['wid'] = C['wid'].astype('float')
    # Position angle is a special case, needs converting from clockwise angle
    # to position angle (counterclockwise angle)
    C['pa'] = 360.0 - C['pa'].astype('float')
    # Calculate time taken between first appearence (tb) and mid-point (tm).
    # Also convert to seconds from days.
    C['dt'] = pd.Series((C.tm-C.tb)*24*60*60, index=C.index)
    return C


def load_cor2_data(data_file):
    """ This is a function to import the Solar Stormwatch Identifications from
    made using the COR2 Coronagraphs on the STEREO-A and STEREO-B spacecraft.
    The input to the function is the path of the COR2 data file. The raw data
    are imported into a pandas dataframe with the following
    fields:
    craft : 'A' for STEREO-A, 'B' for STEREo-B
    assets : solar stormwatch asset name,
    zid   : zooniverse id,
    tb    : Time of entry into COR2 (in Julian Days (JD))
    tm    : Time of reaching mid point of COR2 (in JD),
    pa   : CME location (counterclockwise angle from image north, degrees),
    edg   : CME edge (clockwise angle of lower edge from image north, degs),
    wid   : CME width (in degrees),
    dt    : timing difference between entry and midpoint
    """
    # Get some lists for binning the COR2 data into.
    craft = []
    assets = []
    zids = []
    cor2_bgn =[]
    cor2_mid =[]
    cor2_loc = []
    cor2_wid = []
    cor2_edg = []
    time_err = 0
    area_err = 0
    # Have to go over the file line-by-line as it has non-standard use of two
    # delimiters in the text file.
    with open(data_file) as f:
        lines = f.readlines();
        # Remove unhelpful whitespace and end of line terminators.
        for l in lines:
            elements = l[:-1].replace(' ','').split(';')
            # Use asset tag to work out if STA or STB
            a_test = elements[0].split('_')
            if a_test[1] == 'cor2a.flv':
                craft.append('A')
            elif a_test[1] == 'cor2b.flv':
                craft.append('B')
            # Append asset name and Zooniverse ID
            assets.append(elements[0])
            zids.append(int(elements[1]))
            # Append appearance time.
            if elements[3] == '\\N':
                cor2_bgn.append('NaN')
                time_err += 1
            else:
                cor2_bgn.append(float(elements[3]))
            # Append time at middle of field-of-view
            if elements[4] == '\\N':
                cor2_mid.append('NaN')
                time_err += 1
            else:
                cor2_mid.append(float(elements[4]))
            # Append direction/location and width. Remember the location is
            # given as clockwise angle from solar north. This needs to be
            # converted to position angle (counter-clockwise angle from solar-N)
            if elements[5] == '\\N':
                area_err += 1
                cor2_wid.append('NaN')
                cor2_edg.append('NaN')
                cor2_loc.append('NaN')
            else:
                x = [float(i) for i in elements[5].split(',')]
                cor2_wid.append(x[0])
                cor2_edg.append(x[1])
                cor2_loc.append(x[1] + (x[0]/2))
    # Convert the lists of COR2 data into a pandas dataframe
    C = pd.DataFrame({'assets' : assets, 'zid' : zids, 'tb' : cor2_bgn,
                      'tm' : cor2_mid, 'pa' : cor2_loc, 'wid' : cor2_wid,
                      'edg': cor2_edg, 'craft' : craft})
    # A few of the datatypes need setting to floats.
    C['tb'] = C['tb'].astype('float')
    C['tm'] = C['tm'].astype('float')
    C['wid'] = C['wid'].astype('float')
    # Position angle is a special case, needs converting from clockwise angle
    # to position angle (counterclockwise angle)
    C['pa'] = 360.0 - C['pa'].astype('float')
    # Calculate time taken between first appearence (tb) and mid-point (tm).
    # Also convert to seconds from days.
    C['dt'] = pd.Series((C.tm-C.tb)*24*60*60, index=C.index)
    return C
    
    
def cor1_submission_quality_control(C, data_dir):
    """ This function takes the dataframe of all the cor1 submissions and
    applies some basic quality control rules to remove submissions that are
    clearly spurious. For example, by definition a CME must reach the middle
    of the cor1 field-of-view after it entered, therefore tm > tb. This
    function works on the dataframe output from the load_cor1_data function
    """
    # Use logical indexing to identify submissions with bad values, i.e. dt<0,
    # pa == 360 or wid == 0
    bad_wid  = C['wid'] == 0
    bad_edg = C['edg'] == 0.0
    bad_time = C['dt'] <= 0
    # Define new variable speed and add to data frame
    # Use average mid point distances calculated from the SSW masks applied to
    # some example images.
    # Do STA and STB seperately
    # Preallocate
    C['speed'] = pd.Series(np.zeros(C['tm'].shape), index=C.index)
    # Calc A speeds
    ida = C['craft'] == 'A'
    C['speed'][ida] = (1.22*rsun.value)/C.dt[ida]
    # Calc B speeds
    idb = C['craft'] == 'B'
    C['speed'][idb] = (1.32*rsun.value)/C.dt[idb]
    # identify submissions with bad speeds
    bad_speed_lo = C['speed'] < 10
    bad_speed_hi = C['speed'] > 3000
    # Find any values in C1 that meet any of these conditions.
    all_bad  = np.logical_or(bad_wid, bad_edg)
    all_bad = np.logical_or(all_bad, bad_time)
    all_bad = np.logical_or(all_bad, bad_speed_lo)
    all_bad = np.logical_or(all_bad, bad_speed_hi)
    # Drop these bad values.
    C.drop(C.index[all_bad], axis=0, inplace=True)
    # Re-index the array.
    C.set_index(np.arange(len(C)), inplace=True)
    # Print to screen the numbers of events removed:
    print 'Initial number of submissions: {}'.format(len(all_bad))
    print 'Number of submissions with pa = 360 : {}'.format(sum(bad_edg))
    print 'Number of submissions with wid = 0 : {}'.format(sum(bad_wid))
    print 'Number of submissions with dt < 0 : {}'.format(sum(bad_time))
    print 'Number of submissions with speed < 50 : {}'.format(sum(bad_speed_lo))
    print 'Number of submissions with speed > 3000 : {}'.format(sum(bad_speed_hi))
    print 'Total number of submissions removed : {}'.format(sum(all_bad))
    print 'New number of submissions: {}'.format(len(C))
    qc_summary = os.path.join(data_dir, "cor1_qc_summary.dat")
    with open(qc_summary, 'w') as f:
        f.write('Initial number of submissions: {}\n'.format(len(all_bad)))
        f.write('Number of submissions with pa = 360 : {}\n'.format(sum(bad_edg)))
        f.write('Number of submissions with wid = 0 : {}\n'.format(sum(bad_wid)))
        f.write('Number of submissions with dt < 0 : {}\n'.format(sum(bad_time)))
        f.write('Number of submissions with speed < 50 : {}\n'.format(sum(bad_speed_lo)))
        f.write('Number of submissions with speed > 3000 : {}\n'.format(sum(bad_speed_hi)))
        f.write('Total number of submissions removed : {}\n'.format(sum(all_bad)))
        f.write('New number of submissions: {}\n'.format(len(C)))
    return C


def cor2_submission_quality_control(C, data_dir):
    """ This function takes the dataframe of all the COR2 submissions and
    applies some basic quality control rules to remove submissions that are
    clearly spurious. For example, by definition a CME must reach the middle
    of the COR2 field-of-view after it entered, therefore tm > tb. This
    function works on the dataframe output from the load_COR2_data function
    """
    # Use logical indexing to identify submissions with bad values, i.e. dt<0,
    # pa == 360 or wid == 0
    bad_wid  = C['wid'] == 0
    bad_edg = C['edg'] == 0.0
    bad_time = C['dt'] <= 0
    # Define new variable speed and add to data frame
    # Use average mid point distances calculated from the SSW masks applied to
    # some example images.
    # Do STA and STB seperately
    # Preallocate
    C['speed'] = pd.Series(np.zeros(C['tm'].shape), index=C.index)
    # Calc A speeds
    ida = C['craft'] == 'A'
    C['speed'][ida] = (6.31*rsun.value)/C.dt[ida]
    # Calc B speeds
    idb = C['craft'] == 'B'
    C['speed'][idb] = (7.0*rsun.value)/C.dt[idb]
    # identify submissions with bad speeds
    bad_speed_lo = C['speed'] < 50
    bad_speed_hi = C['speed'] > 3000
    # Find any values in c2 that meet any of these conditions.
    all_bad  = np.logical_or(bad_wid, bad_edg)
    all_bad = np.logical_or(all_bad, bad_time)
    all_bad = np.logical_or(all_bad, bad_speed_lo)
    all_bad = np.logical_or(all_bad, bad_speed_hi)
    # Drop these bad values.
    C.drop(C.index[all_bad], axis=0, inplace=True)
    # Re-index the array.
    C.set_index(np.arange(len(C)), inplace=True)
    # Print to screen the numbers of events removed:
    print 'Initial number of submissions: {}'.format(len(all_bad))
    print 'Number of submissions with pa = 360 : {}'.format(sum(bad_edg))
    print 'Number of submissions with wid = 0 : {}'.format(sum(bad_wid))
    print 'Number of submissions with dt < 0 : {}'.format(sum(bad_time))
    print 'Number of submissions with speed < 50 : {}'.format(sum(bad_speed_lo))
    print 'Number of submissions with speed > 3000 : {}'.format(sum(bad_speed_hi))
    print 'Total number of submissions removed : {}'.format(sum(all_bad))
    print 'New number of submissions: {}'.format(len(C))
    qc_summary = os.path.join(data_dir, "cor2_qc_summary.dat")
    with open(qc_summary, 'w') as f:
        f.write('Initial number of submissions: {}\n'.format(len(all_bad)))
        f.write('Number of submissions with pa = 360 : {}\n'.format(sum(bad_edg)))
        f.write('Number of submissions with wid = 0 : {}\n'.format(sum(bad_wid)))
        f.write('Number of submissions with dt < 0 : {}\n'.format(sum(bad_time)))
        f.write('Number of submissions with speed < 50 : {}\n'.format(sum(bad_speed_lo)))
        f.write('Number of submissions with speed > 3000 : {}\n'.format(sum(bad_speed_hi)))
        f.write('Total number of submissions removed : {}\n'.format(sum(all_bad)))
        f.write('New number of submissions: {}\n'.format(len(C)))
    return C
    

def select_cor1_subset(C, time_start=None, time_stop=None, craft=None):
    """ This function can be used to select a subset of the cor1 subsmissions,
    and can be used to select a subset by either time, craft or both.
    Inputs:
    C = Dataframe of cor1 submissions.
    time_start = Time to start selecting submissions in Julian Date
    time_stop = Time to stop selecting events Julian Date
    craft = Either 'A' or 'B' to identify STEREO-A or STEREO-B
    Examples -
    - Select all STEREO-A data
    C_subset = select_cor1_subset(C,craft='A')
    - Select all data after 1st-Jan-2010 (Julian date = 2455197.5)
    C_subset = select_cor1_subset(C,t_start=2455197.5)
    - Select all STEREO-B data between 1-March-2009 and 30-March-2009
    C_subset = select_cor1_subset(C,t_start=2454891.5,t_stop=2454920.5,craft='B')
    """
    # Make a copy of C, from which the subset will be created.
    Cs = C.copy()
    # First do the craft subset selection.
    if craft == 'A':
        Cs = Cs[Cs['craft'] == 'A'].copy()
    elif craft == 'B':
        Cs = Cs[Cs['craft'] == 'B'].copy()
    # Now do time selections.
    # First case is all times after time_start, no upper limit.
    if time_start != None and time_stop == None:
        Cs = Cs[Cs['tm'] >= time_start].copy()
    elif time_start == None and time_stop != None:
        Cs = Cs[Cs['tm'] <= time_stop].copy()
    elif time_start != None and time_stop != None:
        Cs = Cs[np.logical_and(Cs['tm'] >= time_start,
                               Cs['tm'] <= time_stop)].copy()
    # Re-index the arrays.
    Cs.set_index(np.arange(len(Cs)), inplace=True)
    return Cs


def select_cor2_subset(C, time_start=None, time_stop=None, craft=None):
    """ This function can be used to select a subset of the COR2 subsmissions,
    and can be used to select a subset by either time, craft or both.
    Inputs:
    C = Dataframe of COR2 submissions.
    time_start = Time to start selecting submissions in Julian Date
    time_stop = Time to stop selecting events Julian Date
    craft = Either 'A' or 'B' to identify STEREO-A or STEREO-B
    Examples -
    - Select all STEREO-A data
    C_subset = select_COR2_subset(C,craft='A')
    - Select all data after 1st-Jan-2010 (Julian date = 2455197.5)
    C_subset = select_COR2_subset(C,t_start=2455197.5)
    - Select all STEREO-B data between 1-March-2009 and 30-March-2009
    C_subset = select_COR2_subset(C,t_start=2454891.5,t_stop=2454920.5,craft='B')
    """
    # Make a copy of C, from which the subset will be created.
    Cs = C.copy()
    # First do the craft subset selection.
    if craft == 'A':
        Cs = Cs[Cs['craft'] == 'A'].copy()
    elif craft == 'B':
        Cs = Cs[Cs['craft'] == 'B'].copy()
    # Now do time selections.
    # First case is all times after time_start, no upper limit.
    if time_start != None and time_stop == None:
        Cs = Cs[Cs['tm'] >= time_start].copy()
    elif time_start == None and time_stop != None:
        Cs = Cs[Cs['tm'] <= time_stop].copy()
    elif time_start != None and time_stop != None:
        Cs = Cs[np.logical_and(Cs['tm'] >= time_start,
                               Cs['tm'] <= time_stop)].copy()
    # Re-index the arrays.
    Cs.set_index(np.arange(len(Cs)), inplace=True)
    return Cs
    
    
def make_time_pa_grid(t_start, t_stop, dpa=1.0, dt=1.0):
    """ This function makes a time and position angle grid for clustering the
    Solarstormwatch subsmissions.
    Inputs:
    t_start = Start date of time grid (in Julian Day)
    t_stop = Stop date of time grid (in Julian Day)
    dpa = Angular resolution of the grids position angle axis (in degrees).
          Defaults to 1deg.
    dt = Time resolution of the grids time axis (in hours) Defaults to 1hr
    Returns:
    pa_grid = 1-d array of positiong angle grid values
    time_grid = 1-d array of time grid values
    P = 2-d array of position angles created from meshgrid of pa and t.
    T = 2-d array of time created from meshgrid of pa and t.
    """
    # Position angle grid:
    pa_grid = np.arange(0, 360+dpa, dpa)
    # Time grid:
    dt = dt/24 # Convert time resolution to fraction of a day for Julian dates
    time_grid = np.arange(t_start, t_stop+dt, dt)
    # Make a meshgrid of these values, for doing the running counts in.
    T, P = np.meshgrid(time_grid, pa_grid)
    return pa_grid, time_grid, P, T


def count_submissions(C, pa_grid, time_grid, dpa=20.0, dt=2.0):
    """ This function counts the number of cor1 Solar Stormwatch submissions
    in a 2-d grid of mid-point time (tm) and position angle (pa). It does this
    by performing a running count on a 2-d grid of position angles and times.
    Inputs:
    C = Dataframe of cor1 submissions you want to look for clusters in.
    dpa = Position angle window width for the running count. Default = 20 deg
    dt = Time window width for the running count. Default = 2 hr
    Returns:
    COUNT = A 2-d array giving the running count of the cor1 submissions,
            Array coordinates correspond to the meshgrid of pa_grid and
            time_grid. COUNT[i,j] contains the number of cor1 submissions
            within the pa_grid[i]+/-dpa and time_grid[j]+/-dt window
    """
    # Preallocate array for the running count data
    COUNT = np.zeros((len(pa_grid), len(time_grid)), dtype=float)
    # Convert time resolution to daily unit
    dt = dt/24.0
    # Loop over pa grid and time grid to count events in rolling window.
    for i in range(len(pa_grid)):
        # Identify all entries within window around pa_grid[j]
        id_pa = np.logical_and(C['pa'] >= pa_grid[i]-dpa,
                               C['pa'] < pa_grid[i]+dpa)
        for j in range(len(time_grid)):
            # Identify all entries within window around t_grid[i], using the subset
            # of entries from id_pa.
            id_joint = np.logical_and(C['tm'][id_pa] >= time_grid[j]-dt,
                                      C['tm'][id_pa] < time_grid[j]+dt)
            # Update X with the number of true elements in id_joint
            COUNT[i,j] = sum(id_joint)
    return COUNT


def segment_clusters(C, pa_grid, time_grid, dpa, dt, COUNT, count_thresh=20):
    """ This function applies thresholding and segmentation techniques to the
    running count data provided by cluster submissions. The thresholding turns
    the COUNT data into a binary array, with 1's above the threshold, and 0's
    below. Areas of COUNT above a suitable threshold are probably regions
    representing a CME, as many people have seen a similar feature at a similar
    time and location. The region above the treshold is then dilated so that it
    includes the locations of all points contributing to the maximum count.
    The dilation depends on the time and pa window widths used in
    count_submissions
    Input:
    C = Dataframe of cor1 submissions you want to look for clusters in.
    pa_grid = 1-d array of positiong angle grid values
    time_grid = 1-d array of time grid values
    dpa = Position angle window width used in count_submissions
    dt = Time window width used in count_submissions
    COUNT = Running count data returned by cluster_submissions
    count_thresh = Threshold to apply to COUNT to identify regions of clustered
             submissions. Defaults to 20. This isn't optimised.
    Returns:
    C = cor1 dataframe, updated with a new column of cluster labels.
    COUNT_SEGMENT = A binary version of COUNT, made by thresholding the COUNT
                    input
    CME_SEGMENTS = The segmented version of COUNT_SEGMENT, where each
                   individual cluster of submissions is labelled with an integer.
                   0 indicates the background. Labels >0 indicate clusters of
                   submissions.
    """
    # Make a copy of count for segmenting.
    COUNT_SEG = COUNT.copy()
    # Set values below thresh to 0, above thresh to 1.
    COUNT_SEG[COUNT_SEG < count_thresh] = 0
    COUNT_SEG[COUNT_SEG > 0] = 1
    # Dilate this region so that the thresholded area selects all points that
    # contribute to the count in this region. So Dilate each point in the
    # thresholded region by the window width
    # Need to calculate size of structuring element. This determined by grid
    # size and clustering window width. i.e how many grid cells == window width
    # Get grid resolution
    grid_dp = pa_grid[1] - pa_grid[0]
    grid_dt = (time_grid[1] - time_grid[0])*24.0 #in hours, like window res.
    # Work out
    s_dp = np.fix(dpa/grid_dp)
    s_dt = np.fix(dt/grid_dt)
    struct = np.ones((2*(s_dp)+1,2*(s_dt)+1))
    struct == True
    # Segment and remove very small features
    CME_SEG = measure.label(COUNT_SEG)
    lab,cnt = np.unique(CME_SEG, return_counts=True)
    for l in lab[np.where(cnt < 5)]:
        CME_SEG[CME_SEG == l] = 0
    CME_SEG[CME_SEG != 0] = 1
    # Port this back to being the segmented count array
    COUNT_SEG = CME_SEG
    del CME_SEG
    # Segment out the thresholded regions using the measure.label() routine
    # See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.measurements.label.html.
    # This would work on whole array, but do it over each event like this to
    # flag cells which could belong to two or more cmes and blank these out
    labs = np.unique(measure.label(COUNT_SEG))
    CME_SEGMENTS = np.zeros(COUNT_SEG.shape)
    # Loop over non zero values
    for l in labs[1:]:
        CME_SEG_part = measure.label(COUNT_SEG)
        # Set non labels==l to 0
        CME_SEG_part[CME_SEG_part != l] = 0
        # Binary dilate
        CME_SEG_part = ndi.binary_dilation(CME_SEG_part,
                                           structure=struct).astype('int32')
        # Set zero values of CME_SEG_OUT to
        ida = np.logical_and(CME_SEGMENTS == 0, CME_SEG_part != 0)
        CME_SEGMENTS[ida] = l
    # Now use segmented regions in CME_SEGMENTS to assign the relevant label to
    # each element of C
    labels = []
    for i in C.index:
        id_t = np.argmin( np.abs(time_grid - C['tm'][i]))
        id_p = np.argmin( np.abs(pa_grid - C['pa'][i]))
        labels.append(CME_SEGMENTS[id_p, id_t])
    # Add new column to C with these CME labels
    C['label'] = pd.Series(labels, index=C.index)
    return C,COUNT_SEG,CME_SEGMENTS


def build_CME_dataframe(C):
    """This function builds a dataframe of CMEs defined using the dataframe
    of cor1 submissions that were labelled by segment_clustered_submissions.
    All elements in the cor1 dataframe belonging to a particular cluster are
    averaged together, and the standard deviation calculated as a measure of
    the uncertainty in the average. Returns a dataframe with the following
    fields:
    tm      = Mean time-of-middle in Julian Date format
    tm_utc  = Mean time-of-middle in UTC date string format
    tm_err  = Standard deviation of time-of-middle estimates (in days)
    tb      = Mean time-of-appearance in Julian Date format
    tb_utc  = Mean time-of-appearence in UTC date string format
    tb_err  = Standard deviation of time-of-appearance estimates (in days)
    pa      = Mean position angle (in degrees)
    pa_err  = Standard deviation of position angle estimates (in degrees)
    wid     = Mean CME width (in degrees)
    wid_err = Standard deviation of CME width estimates (in degrees)
    N       = Number of submissions in the cluster.
    """
    # Get list of unique labels, excluding zero as this is the background label,
    # and doesnt relate to clusters of subsmissions.
    unique_labels = np.unique(C['label'][C['label'] != 0])
    # Loop over the unique CME labels to identify clusters of submissions and average
    # the properties to define events. Bin this data into a pandas dataframe.
    # Setup some lists to store average event properties, error estimates and the
    # number of samples in each cluster.
    tm_av = []
    tm_err = []
    tb_av = []
    tb_err = []
    pa_av = []
    pa_err = []
    wid_av = []
    wid_err = []
    n_samp = []
    # Break clause for if there are no CMEs in this block.
    if len(unique_labels) == 0:
        # Return empty dataframe.
        C_cmes = pd.DataFrame({'tb' : tb_av, 'tb_err' : tb_err,
                               'tm' : tm_av, 'tm_err' : tm_err,
                               'pa' : pa_av, 'pa_err' : pa_err,
                               'wid' : wid_av, 'wid_err' : wid_err,
                               'N' : n_samp}, index=unique_labels)
        # Add on the UTC columns
        C_cmes['tm_utc'] = []
        C_cmes['tb_utc'] = []
        return C_cmes
    for l in unique_labels:
        tb_av.append(np.mean(C['tb'][C['label'] == l]))
        tb_err.append(np.std(C['tb'][C['label'] == l]))
        tm_av.append(np.mean(C['tm'][C['label'] == l]))
        tm_err.append(np.std(C['tm'][C['label'] == l]))
        pa_av.append(np.mean(C['pa'][C['label'] == l]))
        pa_err.append(np.std(C['pa'][C['label'] == l]))
        wid_av.append(np.mean(C['wid'][C['label'] == l]))
        wid_err.append(np.std(C['wid'][C['label'] == l]))
        n_samp.append(sum(C['label'] == l))
    # Create a pandas dataframe of the CME average properties.
    C_cmes = pd.DataFrame({'tb' : tb_av, 'tb_err' : tb_err,
                           'tm' : tm_av, 'tm_err' : tm_err,
                           'pa' : pa_av, 'pa_err' : pa_err,
                           'wid' : wid_av, 'wid_err' : wid_err,
                           'N' : n_samp}, index=unique_labels)
    # Add on columns of UTC times.
    C_cmes['tm_utc'] = Time(C_cmes['tm'], format='jd').isot
    C_cmes['tb_utc'] = Time(C_cmes['tb'], format='jd').isot
    C_cmes.sort_values('tm', axis=0, ascending=True, inplace=True)
    C_cmes.set_index(np.arange(len(C_cmes)), inplace=True)
    return C_cmes


def cor1_block_summary_plot(C, T, P, COUNT, CME_SEG, craft, fig_dir, count_thresh=None, tag=None):
    """Function to produce a summary plot that shows the running counts and
    the identified clusters of points. Figure is saved as a png in the figure
    directory set at top of script.
    Input:
    C = The dataframe of cor1 submissions that has been used to identify
        clusters of submissions
    T = Meshgrid of times returned from make_time_pa_grid
    P = Meshgrid of position angles returned from make_time_pa_grid
    COUNT = Running windowed count of submissions in the position time grid.
    COUNT_SEG = Thresholded and segmented count returned from segment_clusters.
    craft = 'A' or 'B' to show which STEREO spacecraft
    fig_dir = location of folder to store plots
    count_thresh = The threshold of counts used to identify clusters. If this
                   argument is set, it will draw a contour at this level on the
                   count data.
    tag = A string tag to append to the filename that gives info on the windows
          and thresholds used in the clustering.
    """
    # FIGURE CODE - Plot to show how the counting and thresholding works.
    # Plot summary of the count data and the thresholded count data.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
    c1 = ax1.contourf(T, P, COUNT, np.arange(0, 205, 5), cmap=mplcm.viridis,
                      zorder=1)
    # Add on a contour of the threshold used to define events
    if count_thresh != None:
        ax1.contour(T, P, COUNT, levels=[count_thresh], color='k', zorder=2)
    # Add a colorbar to the plot
    cbax = fig.add_axes([0.935, 0.125, 0.015, 0.8])
    plt.colorbar(c1, cax=cbax, orientation='vertical')
    ftsz = 12
    cbax.set_ylabel('Number of Observations', fontsize=ftsz)
    plt.setp(cbax.get_yticklabels(), fontsize=ftsz)
    ax1.set_xlabel('Julian date')
    ax1.set_ylabel('Position angle')
    ax2.plot(C['tm'], C['pa'], 'r.', zorder=1)
    ax2.contour(T, P, CME_SEG, levels=np.unique(CME_SEG), colors='k', zorder=2)
    ax2.set_xlabel('Julian date')
    ax2.set_ylabel('Position angle')
    ax2.set_xlabel('Julian date')
    ax2.set_ylabel('Position angle')
    # Update this axis's limits
    tl  = [T[0,0], T[0,-1]]
    ax1.set_xlim(tl)
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_ylim([0, 360])
    ax2.set_ylim(ax1.get_ylim())
    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.98,
                        hspace=0.125)
    cbax.set_position([0.9325, 0.565, 0.015, 0.415])
    # Make a folder for these figures.
    new_fig_dir = os.path.join(fig_dir, tag)
    if not os.path.exists(new_fig_dir):
        os.mkdir(new_fig_dir)
    ttag = Time(tl[0], format='jd').isot[:10]
    if tag == None:
        name = os.path.join(new_fig_dir, 'cor1' + craft + '_cmes_' + ttag + '.png')
    else:
        name = os.path.join(new_fig_dir, 'cor1' + craft + '_cmes_' + ttag + '_'+ tag + '.png')
    plt.savefig(name)
    plt.close('all')

    
def cor2_block_summary_plot(C, T, P, COUNT, CME_SEG, craft, fig_dir, count_thresh=None, tag=None):
    """Function to produce a summary plot that shows the running counts and
    the identified clusters of points. Figure is saved as a png in the figure
    directory set at top of script.
    Input:
    C = The dataframe of COR2 submissions that has been used to identify
        clusters of submissions
    T = Meshgrid of times returned from make_time_pa_grid
    P = Meshgrid of position angles returned from make_time_pa_grid
    COUNT = Running windowed count of submissions in the position time grid.
    COUNT_SEG = Thresholded and segmented count returned from segment_clusters.
    craft = 'A' or 'B' to show which STEREO spacecraft
    fig_dir = location of folder to store plots
    count_thresh = The threshold of counts used to identify clusters. If this
                   argument is set, it will draw a contour at this level on the
                   count data.
    tag = A string tag to append to the filename that gives info on the windows
          and thresholds used in the clustering.
    """
    # FIGURE CODE - Plot to show how the counting and thresholding works.
    # Plot summary of the count data and the thresholded count data.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    c1 = ax1.contourf(T, P, COUNT, np.arange(0, 205, 5), cmap=mplcm.viridis,
                      zorder=1)
    # Add on a contour of the threshold used to define events
    if count_thresh != None:
        ax1.contour(T, P, COUNT, levels=[count_thresh], color='k', zorder=2)
    # Add a colorbar to the plot
    cbax = fig.add_axes([0.935, 0.125, 0.015, 0.8])
    plt.colorbar(c1, cax=cbax, orientation='vertical')
    ftsz = 12
    cbax.set_ylabel('Number of Observations', fontsize=ftsz)
    plt.setp(cbax.get_yticklabels(), fontsize=ftsz)
    ax1.set_xlabel('Julian date')
    ax1.set_ylabel('Position angle')
    ax2.plot(C['tm'], C['pa'], 'r.', zorder=1)
    ax2.contour(T, P, CME_SEG, levels=np.unique(CME_SEG), colors='k', zorder=2)
    ax2.set_xlabel('Julian date')
    ax2.set_ylabel('Position angle')
    ax2.set_xlabel('Julian date')
    ax2.set_ylabel('Position angle')
    # Update this axis's limits
    tl  = [T[0,0], T[0,-1]]
    ax1.set_xlim(tl)
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_ylim([0, 360])
    ax2.set_ylim(ax1.get_ylim())
    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.98,
                        hspace=0.125)
    cbax.set_position([0.9325, 0.565, 0.015, 0.415])
    # Make a folder for these figures.
    new_fig_dir = os.path.join(fig_dir, tag)
    if not os.path.exists(new_fig_dir):
        os.mkdir(new_fig_dir)
    ttag = Time(tl[0], format='jd').isot[:10]
    if tag == None:
        name = os.path.join(new_fig_dir, 'cor2' + craft + '_cmes_' + ttag + '.png')
    else:
        name = os.path.join(new_fig_dir, 'cor2' + craft + '_cmes_' + ttag + '_'+ tag + '.png')
    plt.savefig(name)
    plt.close('all')

    
def run_cor1(pa_grid_width, t_grid_width, count_dp, count_dt, count_thresh, data_dir, fig_dir):
    """
    :input: p_grid_width # Position angle grid width (degrees)
    :input: t_grid_width # temporal grid width (hours)
    :input: count_dp # position angle window width for counting submissions (degs)
    :input: count_dt # time window width for counting submissions (hours)
    :input: count_thresh # Threshold to apply to count to identify clusters of subs
    """
    # Load in the cor1 data
    cor1_data = os.path.join(data_dir,'cor1_timing_and_wid.txt')
    C1 = load_cor1_data(cor1_data)
    # Do some basic quality control on the submissions
    C1 = cor1_submission_quality_control(C1, data_dir)
    for cr in range(2):
        if cr == 0:
            craft = 'A'
        else:
            craft = 'B'
        # Build empty dataframe for storing events in.
        ALL_CMES = pd.DataFrame({'tb' : [], 'tb_utc' : [], 'tb_err' : [],
                                 'tm' : [], 'tm_utc' : [], 'tm_err' : [],
                                 'pa' : [], 'pa_err' : [], 'wid' : [],
                                 'wid_err' : [], 'N' : []})
        # Make a string tag to append to filenames making clear selected variables
        tag = 'dt{0:02d}_dp{1:02d}_t{2:02d}'.format(int(count_dt), int(count_dp),
                                            int(count_thresh))
        # Define the time span and window width to analyse.
        #The cor1 data studied by Stormwatch span the period 2007-02-28 to 2007-02-16.
        window_width = 27 # Window block size (days).
        # Define start and end dates to loop through in steps of window_width
        start_date = Time('2007-02-28 12:00:00', format='iso')
        end_date = Time('2010-02-12 12:00:00', format='iso')
        # Get array of times of beginning of each window (in julian days)
        times = np.arange(start_date.jd, end_date.jd + window_width, window_width)
        for t in times:
            # Pull out subset of the cor1A identifications.
            t_lo = t - 5
            t_hi = t + window_width + 5
            C1s= select_cor1_subset(C1, time_start=t_lo, time_stop=t_hi,
                                    craft=craft)
            # DATA PROCESSING CODE
            # Build a position-angle and time grid, for clustering the events.
            t_start = t
            t_stop = t + window_width
            p_grid, t_grid, P, T = make_time_pa_grid(t_start, t_stop,
                                                  dpa=pa_grid_width,
                                                  dt=t_grid_width)
            # Count the number of submissions, so that we can search for clusters.
            COUNT = count_submissions(C1s, p_grid, t_grid, dpa=count_dp,
                                      dt=count_dt)
            # Segment the COUNT data into clusters relating to likely single CMEs
            C1s, COUNT_SEG, CME_SEGS = segment_clusters(C1s, p_grid, t_grid,
                                                        count_dp, count_dt,
                                                        COUNT,
                                                        count_thresh=count_thresh)
            # Make Summary plots
            cor1_block_summary_plot(C1s, T, P, COUNT, CME_SEGS, craft, fig_dir,
                                    count_thresh=count_thresh, tag=tag)
            # DATA PROCESSING CODE
            # Build dataframe of CMEs using the labels in C1a
            cmes = build_CME_dataframe(C1s)
            # Append to the whole list of CMEs.
            ALL_CMES = ALL_CMES.append(cmes, ignore_index=True)
        # Save the dataframe to a CSV file in the data directory.
        name = data_dir + r'\COR1'+ craft +'_CMES_' + tag +'.csv'
        ALL_CMES.to_csv(name, sep=',')

        
def run_cor2(pa_grid_width, t_grid_width, count_dp, count_dt, count_thresh, data_dir, fig_dir):
    """
    :input: p_grid_width # Position angle grid width (degrees)
    :input: t_grid_width # temporal grid width (hours)
    :input: count_dp # position angle window width for counting submissions (degs)
    :input: count_dt # time window width for counting submissions (hours)
    :input: count_thresh # Threshold to apply to count to identify clusters of subs
    """
    # Load in the COR2 data
    cor2_data = os.path.join(data_dir,'cor2_timing_and_wid.txt')
    C2 = load_cor2_data(cor2_data)
    # Do some basic quality control on the submissions
    C2 = cor2_submission_quality_control(C2, data_dir)
    for cr in range(2):
        if cr == 0:
            craft = 'A'
        else:
            craft = 'B'
        # Build empty dataframe for storing events in.
        ALL_CMES = pd.DataFrame({'tb' : [], 'tb_utc' : [], 'tb_err' : [],
                                 'tm' : [], 'tm_utc' : [], 'tm_err' : [],
                                 'pa' : [], 'pa_err' : [], 'wid' : [],
                                 'wid_err' : [], 'N' : []})
        # Make a string tag to append to filenames making clear selected variables
        tag = 'dt{0:02d}_dp{1:02d}_t{2:02d}'.format(int(count_dt), int(count_dp), int(count_thresh))
        # Define the time span and window width to analyse.
        #The COR2 data studied by Stormwatch span the period 2007-02-28 to 2007-02-16.
        window_width = 27 # Window block size (days).
        # Define start and end dates to loop through in steps of window_width
        start_date = Time('2007-02-28 12:00:00', format='iso')
        end_date = Time('2010-02-12 12:00:00', format='iso')
        # Get array of times of beginning of each window (in julian days)
        times = np.arange(start_date.jd, end_date.jd + window_width, window_width)

        for t in times:
            # Pull out subset of the COR2A identifications.
            t_lo = t - 5
            t_hi = t + window_width + 5
            C2s= select_cor2_subset(C2, time_start=t_lo, time_stop=t_hi, craft=craft)
            # DATA PROCESSING CODE
            # Build a position-angle and time grid, for clustering the events.
            t_start = t
            t_stop = t + window_width
            p_grid, t_grid, P, T = make_time_pa_grid(t_start, t_stop,
                                                  dpa=pa_grid_width, dt=t_grid_width)
            # Count the number of submissions, so that we can search for clusters.
            COUNT = count_submissions(C2s, p_grid, t_grid, dpa=count_dp, dt=count_dt)
            # Segment the COUNT data into clusters relating to likely single CMEs
            C2s,COUNT_SEG,CME_SEGS = segment_clusters(C2s, p_grid, t_grid,
                                          count_dp ,count_dt, COUNT,
                                          count_thresh=count_thresh)
            # Make Summary plots
            cor2_block_summary_plot(C2s, T, P, COUNT, CME_SEGS, craft, fig_dir,
                                    count_thresh=count_thresh, tag=tag)

            # DATA PROCESSING CODE
            # Build dataframe of CMEs using the labels in C2a
            cmes = build_CME_dataframe(C2s)
            # Append to the whole list of CMEs.
            ALL_CMES = ALL_CMES.append(cmes, ignore_index=True)

        # Save the dataframe to a CSV file in the data directory.
        name = data_dir + r'\COR2'+ craft +'_CMES_' + tag +'.csv'
        ALL_CMES.to_csv(name,sep=',')
