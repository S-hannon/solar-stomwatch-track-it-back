import glob
import numpy as np
import os
import pandas as pd
import sunpy.map
import sunpy.map as smap
import matplotlib.pyplot as plt
import sunpy.sun.constants as spconsts
import fnmatch
import sunpy.coordinates as spc
import astropy.units as u
from astropy.io import fits
from sunpy.net import vso
from sunpy.net.helioviewer import HelioviewerClient
from astropy.time import Time
plt.switch_backend('Agg')

# Constants needed
rsun = spconsts.radius.to('km')


def HPC_to_HPR(lon, lat):
    """
    Function to calculate helioprojective-radial coordinates (elongation and
    position angles) from helioprojective-cartesian coordinates (longitudes and
    latitudes).
    """
    # Put it in rads for np
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    # Elongation calc:
    # Get numerator and denomenator for atan2 calculation
    btm = np.cos(lat)*np.cos(lon)
    top = np.sqrt((np.cos(lat)**2)*(np.sin(lon)**2) + (np.sin(lat)**2))
    el = np.arctan2(top, btm)
    # Position angle calc:
    top = -np.cos(lat)*np.sin(lon)
    btm = np.sin(lat)
    pa = np.arctan2(top, btm)
    pa[lon >= 0] = 2*np.pi + pa[lon >= 0]
    # Put it back into degs
    # Take 90 degrees off both and times it by -1, because...
    el = np.rad2deg(el)*u.deg
    pa = np.rad2deg(pa)*u.deg
    return el, pa


def calc_radial_distance(filename, data_dir):
    """
    A function to calculate the radial distance in the plane of the sky
    of the smallest non masked elongation and the median non masked elongation
    along the row of pixels passing horizontally through the Sun center.
   """
    smap = sunpy.map.Map(filename)
    # Load in the COR2 MASK
    mask_root = data_dir
    print smap.detector
    if smap.detector == 'COR1':
        mask_fn = mask_root + r'\cor1_mask.fts'
        dims = [int(x.value) for x in smap.dimensions]
        mask = sunpy.image.rescale.resample(fits.getdata(mask_fn), dims)
    elif smap.detector == 'COR2' and smap.observatory == 'STEREO A':
        mask_fn = mask_root + r'\cor2A_mask.fts'
        mask = fits.getdata(mask_fn)
    elif smap.detector == 'COR2' and smap.observatory == 'STEREO B':
        mask_fn = mask_root + r'\cor2B_mask.fts'
        mask = fits.getdata(mask_fn)

    smap.mask = np.logical_not(mask)
    # Get all pixel positions
    x = np.arange(1, smap.dimensions.x.value+1, 1)
    y = np.arange(1, smap.dimensions.y.value+1, 1)
    # Mesh these and reorder to get all possible coordinate pairs
    xx, yy = np.meshgrid(x, y)
    lon, lat = smap.pixel_to_data(xx*u.pix, yy*u.pix)
    # Get elon, pa
    el, pa = HPC_to_HPR(lon.to('deg').value, lat.to('deg').value)
    # Get elongations along central row
    el_row = el[np.int(smap.reference_pixel[0].value), :]
    # Get mask along central row
    msk_row = smap.mask[np.int(smap.reference_pixel[0].value), :]
    # Pull out the not masked elongations
    el_row = el_row[np.logical_not(msk_row)]
    # Calculate radial distance of the min elongation
    rb = np.tan(el_row.min().to('rad').value)*smap.dsun.value
    # Get mid point
    el_mid = np.median(el_row)
    # Convert to POS distance
    rm = np.tan(el_mid.to('rad').value)*smap.dsun.value
    # Return the distances
    return rb, rm


def match_events(c1, c2):
    """ Loop through C2 events and find close C1 events. Get indexes of c1 and
    c2 dataframes, these will be modified to create a list of events to remove.
    """
    id_c1_bad = [c for c in c1.index]
    id_c2_bad = [c for c in c2.index]
    # Loop over c2 and match c1 events to it.
    for i in range(len(c2.index)):
        # Find difference in midpoint times and position angles
        dt = c2['tm'][i]-c1['tm']
        dp = c2['pa'][i]-c1['pa']
        # C1 must be before C2 (only dt>0 valid)
        # And dt < 12 hours for association.
        id_good = np.logical_and(dt > 0, dt <= 12.0/24.0)
        # Only keep good entries
        dp = dp[id_good]
        dt = dt[id_good]
        # PA's must be within 45 degrees of each other
        id_good = np.abs(dp) <= 45
        dt = dt[id_good]
        dp = dp[id_good]
        # If any C1 events meet this criteria find the closest
        if len(dt) > 0:
            # Which time matches closest?
            ind_t = np.argmin(dt)
            for p in id_c1_bad:
                if p == ind_t:
                    id_c1_bad.remove(ind_t)
                    id_c2_bad.remove(i)

    # TIDY TUP THE C1 and C2 dataframes
    # Drop unmatched rows
    c1.drop(id_c1_bad, axis=0, inplace=True)
    # Get rid of julian dates
    c1.drop('tb', axis=1, inplace=True)
    c1.drop('tm', axis=1, inplace=True)
    # Put dates into datetime and timedelta formats
    c1['tm'] = pd.to_datetime(c1['tm_utc'])
    c1['tm_err'] = pd.to_timedelta(c1['tm_err'], 'D')
    c1['tb'] = pd.to_datetime(c1['tb_utc'])
    c1['tb_err'] = pd.to_timedelta(c1['tb_err'], 'D')
    c1.drop('tb_utc', axis=1, inplace=True)
    c1.drop('tm_utc', axis=1, inplace=True)
    # Add in distance measures. These will be filled in the loop below
    c1['rb'] = pd.Series(np.zeros(len(c1)), index=c1.index)
    c1['rm'] = pd.Series(np.zeros(len(c1)), index=c1.index)
    # Rename and reorder
    new_nms = ['n', 'pa', 'pa_e', 'tb_e', 'tm_e', 'wid', 'wid_e', 'tm', 'tb',
               'rb', 'rm']
    keys = c1.keys()
    names = {keys[i]: new_nms[i] for i in range(len(keys))}
    c1.rename(columns=names, inplace=True)
    names = ['tb', 'tb_e', 'tm', 'tm_e', 'pa', 'pa_e', 'wid', 'wid_e', 'n',
             'rb', 'rm']
    c1 = c1[names]
    c1.set_index(np.arange(len(c1)), inplace=True)

    # Drop unmatched rows
    c2.drop(id_c2_bad, axis=0, inplace=True)
    # Tidy up the arrays before merging
    c2.drop('tb', axis=1, inplace=True)
    c2.drop('tm', axis=1, inplace=True)
    # Put dates into datetime and timedelta formats
    c2['tm'] = pd.to_datetime(c2['tm_utc'])
    c2['tm_err'] = pd.to_timedelta(c2['tm_err'], 'D')
    c2['tb'] = pd.to_datetime(c2['tb_utc'])
    c2['tb_err'] = pd.to_timedelta(c2['tb_err'], 'D')
    c2.drop('tb_utc', axis=1, inplace=True)
    c2.drop('tm_utc', axis=1, inplace=True)
    # Add in distance measures. These will be filled in the loop below
    c2['rb'] = pd.Series(np.zeros(len(c2)), index=c2.index)
    c2['rm'] = pd.Series(np.zeros(len(c2)), index=c2.index)
    # Rename and reorder
    new_nms = ['n', 'pa', 'pa_e', 'tb_e', 'tm_e', 'wid', 'wid_e', 'tm', 'tb',
               'rb', 'rm']
    keys = c2.keys()
    names = {keys[i]: new_nms[i] for i in range(len(keys))}
    c2.rename(columns=names, inplace=True)
    names = ['tb', 'tb_e', 'tm', 'tm_e', 'rb', 'rm', 'pa', 'pa_e', 'wid',
             'wid_e', 'n']
    c2 = c2[names]
    c2.set_index(np.arange(len(c2)), inplace=True)
    return c1, c2


def download_images(c1, c2, craft, c1_hvr, c2_hvr, cme_root):
    """Download the COR1 and COR2 fits files for the frame nearest tm, to get
    coords and do distance calc. Download several pngs around the event.
    """
    # Open up a VSO client and a Helioviewer Client
    vsoclient = vso.VSOClient()
    hv = HelioviewerClient()
    # Loop over the events, make event directory, download closest fits file
    # and several pngs.
    for i in range(len(c1)):
        print 'length of c1'
        print len(c1)
        print i
        cme_dir = cme_root + r'\cme_' + '{0:02d}'.format(i+1)
        if not os.path.exists(cme_dir):
            os.makedirs(cme_dir)
        #######################################################################
        # Search for COR1 data in +/- 30min window around event tm1
        dt = pd.Timedelta(30, 'm')
        tl = [(c1['tm'][i]-dt).isoformat(), (c1['tm'][i]+dt).isoformat()]
        # Query VSO to find what files available
        qr = vsoclient.query(vso.attrs.Time(tl[0], tl[1]),
                             vso.attrs.Instrument('cor1'),
                             vso.attrs.Source(craft))
        # Take out values that are not 0deg polarised
        q = 0
        while q <= len(qr)-1:
            tpe = qr[q]['info'].split(';')[3].strip()
            if tpe == '0deg.':
                q = q+1
            else:
                qr.pop(q)
        # Find closest mathced in time
        # Get datetimes of the query resutls
        times = np.array([pd.to_datetime(x['time']['start']) for x in qr])
        # Remove querys before the closest match
        q = 0
        while q < np.argmin(np.abs(times - c1['tm'][i])):
            qr.pop(0)
            q = q+1
        # Remove querys after the closest match
        while len(qr) > 1:
            qr.pop(1)
        # Download best matched file
        vsoclient.get(qr, path=cme_dir+r"\{file}").wait()
        for t in times:
            hv.download_png(t.isoformat(), 7.5, c1_hvr,
                            x0=0, y0=0, width=1024, height=1024,
                            directory=cme_dir, watermark=True)
        ###################################################################
        # Search for COR2 data around tm2
        dt = pd.Timedelta(90, 'm')
        tl = [(c2['tm'][i]-dt).isoformat(), (c2['tm'][i]+dt).isoformat()]
        qr = vsoclient.query(vso.attrs.Time(tl[0], tl[1]),
                             vso.attrs.Instrument('cor2'),
                             vso.attrs.Source(craft))
        # Take out values not 2048x2048 double exposures
        q = 0
        while q <= len(qr)-1:
            tpe = qr[q]['info'].split(';')[2].strip()
            sze = qr[q]['info'].split(';')[-1].strip()
            if tpe == 'DOUBLE' and sze == '2048x2048':
                q = q+1
            else:
                qr.pop(q)
        # Find closest mathced in time
        # Get datetimes of the query resutls
        times = np.array([pd.to_datetime(x['time']['start']) for x in qr])
        # Remove querys before the closest match
        q = 0
        while q < np.argmin(np.abs(times - c2['tm'][i])):
            qr.pop(0)
            q = q+1
        # Remove querys after the closest match
        while len(qr) > 1:
            qr.pop(1)
        # Download best matched file
        vsoclient.get(qr, path=cme_dir+r"\{file}").wait()
        # Use the query times to download helioviewer PNGS
        for t in times:
            hv.download_png(t.isoformat(), 14.7, c2_hvr,
                            x0=0, y0=0, width=2048, height=2048,
                            directory=cme_dir, watermark=True)
        ###################################################################
        # Now use the fits files to work out the mid point distance and
        # update the arrays
        c1_file = glob.glob(cme_dir+r'\*c1*.fts')
        rb, rm = calc_radial_distance(c1_file, cme_root)
        c1['rb'][i] = rb
        c1['rm'][i] = rm
        # Repeat for COR2
        c2_file = glob.glob(cme_dir+r'\*c2*.fts')
        rb, rm = calc_radial_distance(c2_file, cme_root)
        c2['rb'][i] = rb
        c2['rm'][i] = rm
    return c1, c2


def run_matching(data_dir):
    """Start the data processing
    """
    # Loop over the craft
    craft_names = ['STEREO_A', 'STEREO_B']
    for craft in craft_names:
        print craft
        if craft == 'STEREO_A':
            cme_root = data_dir
            # Load in COR1 and COR2 CMES
            c1_fn = data_dir + r'/COR1A_CMES_dt01_dp15_t07.csv'
            c1 = pd.read_csv(c1_fn, index_col=0)
            c2_fn = data_dir + r'/COR2A_CMES_dt01_dp15_t12.csv'
            c2 = pd.read_csv(c2_fn, index_col=0)
            c1_hvr = "[28,1,100]"
            c2_hvr = "[29,1,100]"
            # Use C1 and C2 events lists to get matched event list
            c1, c2 = match_events(c1, c2)
            # Get images and use to find the distances for each event
            c1, c2 = download_images(c1, c2, craft, c1_hvr, c2_hvr, cme_root)
            c1_fn = cme_root + r'/COR1A_CMES_dt01_dp15_t07_matched.csv'
            c2_fn = cme_root + r'/COR2A_CMES_dt01_dp15_t12_matched.csv'
            c1.to_csv(c1_fn, sep=',')
            c2.to_csv(c2_fn, sep=',')

        elif craft == 'STEREO_B':
            cme_root = data_dir
            # Load in COR1 and COR2 CMES
            c1_fn = data_dir + r'/COR1B_CMES_dt01_dp15_t07.csv'
            c1 = pd.read_csv(c1_fn, index_col=0)
            c2_fn = data_dir + r'/COR2B_CMES_dt01_dp15_t12.csv'
            c2 = pd.read_csv(c2_fn, index_col=0)
            c1_hvr = "[30,1,100]"
            c2_hvr = "[31,1,100]"
            # Use C1 and C2 events lists to get matched event list
            c1, c2 = match_events(c1, c2)
            # Get images and use to find the distances for each event
            c1, c2 = download_images(c1, c2, craft, c1_hvr, c2_hvr, cme_root)
            c1_fn = cme_root + r'/COR1B_CMES_dt01_dp15_t07_matched.csv'
            c2_fn = cme_root + r'/COR2B_CMES_dt01_dp15_t12_matched.csv'
            c1.to_csv(c1_fn, sep=',')
            c2.to_csv(c2_fn, sep=',')

            
def find_files(root, name):
    """
    Function to search recursively through the spice directory to list all
    files mathcing name
    """
    outfiles = []
    for dirpath, dirnames, files in os.walk(root):
            # Skip dos directories for now
            if dirpath.endswith('dos') == False:
                for f in fnmatch.filter(files, name):
                    outfiles.append(os.path.join(dirpath, f))
    return outfiles


def load_euvi_data(data_file):
    """ This is a function to import the Solar Stormwatch Identifications from
    made using the cor1 Coronagraphs on the STEREO-A and STEREO-B spacecraft.
    The input to the function is the path of the cor1 data file. The raw data
    are imported into a pandas dataframe with the following
    fields:
    craft: 'A' for STEREO-A, 'B' for STEREO-B
    assets: solar stormwatch asset name,
    zid: zooniverse id,
    wav: chosen wavelength, either 17.1, 19.5, 28.4 or 30.4 nm,
    time: Time of entry into EUVI (in Julian Days (JD)),
    lft: Left side of soure location box
    btm: Bottom side of source location box
    wid: Width of source location box
    hei: Height of source location box
    """
    # Get some lists for binning the cor1 data into.
    craft = []
    assets = []
    zids = []
    euvi_wav = []
    euvi_time =[]
    euvi_lft = []
    euvi_btm = []
    euvi_wid = []
    euvi_hei = []
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
            if a_test[2] == 'a':
                craft.append('A')
            elif a_test[2] == 'b':
                craft.append('B')
            # Append chosen wavelength
            wav = a_test[3].split('.')
            euvi_wav.append(wav[0])

            # Append asset name, Zooniverse ID
            assets.append(elements[0])
            zids.append(int(elements[1]))

            # Append appearance time
            if elements[4] == '\\N':
                euvi_time.append('NaN')
                time_err += 1
            else:
                euvi_time.append(float(elements[4]))

            # Append direction/location and width. Remember the location is
            # given as clockwise angle from solar north. This needs to be
            # converted to position angle (counter-clockwise angle from
            # solar-N)
            if elements[5] == '\\N':
                area_err += 1
                euvi_lft.append('NaN')
                euvi_btm.append('NaN')
                euvi_wid.append('NaN')
                euvi_hei.append('NaN')
            else:
                x = [float(i) for i in elements[5].split(',')]
                euvi_lft.append(x[0])
                euvi_btm.append(x[1])
                euvi_wid.append(x[2])
                euvi_hei.append(x[3])

    # Convert the Lists of cor1 data into a pandas dataframe
    C = pd.DataFrame({'assets' : assets, 'zid' : zids, 'wav' : euvi_wav,
                      'tm' : euvi_time, 'l' : euvi_lft, 'b' : euvi_btm,
                      'w' : euvi_wid, 'h' : euvi_hei, 'craft' : craft})

    # A few of the datatypes need setting to floats.
    C['tm'] = C['tm'].astype('float')
    C['l'] = C['l'].astype('float')
    C['b'] = C['b'].astype('float')
    C['w'] = C['w'].astype('float')
    C['h'] = C['h'].astype('float')

    # Calculate the mid point of the box identified by users.
    C['xc'] = C['l'] + C['w']/2
    C['yc'] = C['b'] + C['h']/2
    return C


def select_euvi_subset(C, time_start=None, time_stop=None, craft=None):
    """ This function can be used to select a subset of the euvi subsmissions,
    and can be used to select a subset by either time, craft or both.
    Inputs:
    C = Dataframe of euvi submissions.
    time_start = Time to start selecting submissions in Julian Date
    time_stop = Time to stop selecting events Julian Date
    craft = Either 'A' or 'B' to identify STEREO-A or STEREO-B
    Examples -
    - Select all STEREO-A data
    C_subset = select_euvi_subset(C, craft='A')
    - Select all data after 1st-Jan-2010 (Julian date = 2455197.5)
    C_subset = select_euvi_subset(C, t_start=2455197.5)
    - Select all STEREO-B data between 1-March-2009 and 30-March-2009
    C_subset = select_euvi_subset(C, t_start=2454891.5, t_stop=2454920.5,
    craft='B')
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
    Cs.set_index(np.arange(len(Cs)),inplace=True)
    return Cs


def download_euvi_files(eucme, craft_vso, cme_root):
    """
    Downloads euvi images of events.
    """
    # Loop over the events, make event directory, download closest fits file
    # and several pngs.
    files = []
    for i in range(len(eucme)):
        print i
        # Now load in the most appropriate EUVI image for each event.
        vsoclient = vso.VSOClient()
        # Create folder for images to be stored in
        cme_dir = cme_root + r'\cme_' + '{0:02d}'.format(i+1)
        print cme_dir
        if not os.path.exists(cme_dir):
            os.makedirs(cme_dir)
        # If there is no EUVI data?
        if pd.isnull(eucme['tm'][i]):
            files.append(None)
            # Skip rest of go
            continue
        # Search for COR1 data in +/- 30min window around event tm1
        dt = pd.Timedelta(30,'m')
        print dt
        tl = [(eucme['tm'][i]-dt).isoformat(), (eucme['tm'][i]+dt).isoformat()]
        print tl
        # Query VSO to find what files available
        qr = vsoclient.query(vso.attrs.Time(tl[0], tl[1]),
                             vso.attrs.Instrument('euvi'),
                             vso.attrs.Source(craft_vso))

#        #Take out values that are not from the 171Angstrom Channel
#        q=0
#        while q <= len(qr)-1:
#            wve = qr[q]['wave']['wavemin']
#            if wve=="171":
#                q = q+1
#            else:
#                qr.pop(q)
        print len(qr)
        if len(qr) > 1:
            # Find closest mathced in time
            # Get datetimes of the query resutls
            times = np.array([pd.to_datetime(x['time']['start']) for x in qr])
            # Remove querys before the closest match
            q = 0
            while q < np.argmin(np.abs(times - eucme['tm'][i])):
                qr.pop(0)
                q = q+1
            # Remove querys after the closest match
            while len(qr) > 1:
                qr.pop(1)
                # Download best matched file

            fn = vsoclient.get(qr, path=cme_dir+r"\{file}").wait()
            files.append(fn[0])
        else:
            files.append(None)
    print(files)


def HPC_LONLAT(smap):
    """
    Function to calculate the HPC longitude and latidue for all pixels in the
    input sunpy map smap
    """
    # Get all pixel positions
    x = np.arange(1, smap.dimensions.x.value+1, 1)
    y = np.arange(1, smap.dimensions.y.value+1, 1)
    # Mesh these and reorder to get all possible coordinate pairs
    xx, yy = np.meshgrid(x, y)
    hpc_lon, hpc_lat = smap.pixel_to_data(xx*u.pix, yy*u.pix)
    return hpc_lon, hpc_lat


def find_carr_coords(cme_root, craft, c1cme, eucme):
    """ Function to calculate carrington longitude and latitude of events in
    eucme and add these to the dataframe. These are used to create plots of
    each event.
    """
    # Find the euvi fits files
    if craft == 'A':
        fn_tag = '*eua.fts'
    elif craft == 'B':
        fn_tag = '*eub.fts'

    # Calculate Carrington Longitudes and Latitudes of source locations
    eucme['carrington_longitude'] = pd.Series(np.zeros(len(eucme))*np.NaN,
                                              index=eucme.index)
    eucme['carrington_latitude'] = pd.Series(np.zeros(len(eucme))*np.NaN,
                                             index=eucme.index)
    for i in range(len(eucme)):
        print i
        cme_root_new = r'/home/sq112614/SS/Images/EUVI/ST' + craft
        cme_root2 = cme_root_new + '/cme_' + '{0:02d}'.format(i+1)
        euvi_files = find_files(cme_root2, fn_tag)
        print len(euvi_files)
        if len(euvi_files) > 0:
            print '**********'
            print cme_root2
            print euvi_files
            # Files exist, load in the subpy map to work out carrington coordinates
            euvi = smap.Map(euvi_files[0])
            # Convert the 256x256 coordinates into correctly scaled coordinates
            xn = eucme['l'][i] + eucme['w'][i]/2
            yn = eucme['b'][i] + eucme['h'][i]/2
            xn = (xn/256)*euvi.dimensions[0]
            yn = euvi.dimensions[1] - (yn/256)*euvi.dimensions[1]

            # Get HPC and HPR coords
            lon, lat = HPC_LONLAT(euvi)
            el, pa = HPC_to_HPR(lon.to('deg').value, lat.to('deg').value)
            # Get Carrington coords:
            cf = euvi.coordinate_frame
            hpc = spc.Helioprojective(Tx=lon, Ty=lat, D0=cf.D0, B0=cf.B0,
                                      L0=cf.L0,
                                      representation=cf.representation)
            hcc = spc.Heliocentric(D0=cf.D0, B0=cf.B0, L0=cf.L0)
            hcc = spc.hpc_to_hcc(hpc, hcc)
            hgs = spc.HeliographicStonyhurst(dateobs=cf.dateobs)
            hgc = spc.HeliographicCarrington(dateobs=cf.dateobs)
            hgs = spc.hcc_to_hgs(hcc, hgs)
            hgc = spc.hgs_to_hgc(hgs, hgc)

            # Try looking up carrington coordinates
            idy = np.int(np.round(yn.value))
            idx = np.int(np.round(xn.value))
            hgc_lon = hgc.lon[idy, idx]
            hgc_lat = hgc.lat[idy, idx]
            # If returned NaNs extrapolate to limb
            if np.logical_or(np.isnan(hgc_lon), np.isnan(hgc_lat)):
                print('Limb Extrapolation')
                # Get elongation of Limb
                el_limb = np.arctan(0.99*euvi.rsun_meters/euvi.dsun).to('deg')
                # Get position angle of point
                ppa = pa[idy, idx]
                # Find pixel best matching this elongation and pa
                pa_diff = np.abs(pa-ppa)
                el_diff = np.abs(el-el_limb)
                diff = np.sqrt(pa_diff**2 + el_diff**2)
                idy, idx = np.unravel_index(np.argmin(diff), diff.shape)
                hgc_lon = hgc.lon[idy, idx]
                hgc_lat = hgc.lat[idy, idx]

            # Make sure hgc lon runs between 0 and 360, rather than -180, 180
            if hgc_lon.value < 0:
                hgc_lon = 360*u.deg + hgc_lon

            euvi.peek()
            plt.contour(pa.value, levels=[c1cme['pa'][i]], colors=['r'])
            plt.plot(xn.value, yn.value, 'yo', markersize=12)
            plt.plot(idx, idy, 'y*', markersize=12)
            plt.contour(hgc.lon.value, colors=['white'])
            plt.contour(hgc.lat.value, colors=['m'])
            plt.draw()
            name = cme_root2 + r'\euvi_eruption_id.png'
            plt.savefig(name)
            plt.close('all')
            # Mark out the coords for chucking in the array
            eucme.loc[eucme.index == i, 'carrington_longitude'] = hgc_lon.value
            eucme.loc[eucme.index == i, 'carrington_latitude'] = hgc_lat.value
    return eucme


def run_euvi(data_dir):
    """ Run the euvi code
    """
    # Load in the EUVI submissions
    euvi_data = data_dir + r'/euvi_timing_and_loc.txt'
    cme_root = data_dir
    C = load_euvi_data(euvi_data)

    # Load in the matched C2-C1 CMEs
    #for cr in [1]:
    for cr in range(2):
        if cr == 0:
            craft = 'A'
            craft_vso = 'STEREO_A'
            # Get directories and filenames of matched CME lists
            c1_fn = cme_root + r'/COR1A_CMES_dt01_dp15_t07_matched.csv'
            c2_fn = cme_root + r'/COR2A_CMES_dt01_dp15_t12_matched.csv'
        else:
            craft = 'B'
            craft_vso = 'STEREO_B'
            c1_fn = cme_root + r'/COR1B_CMES_dt01_dp15_t07_matched.csv'
            c2_fn = cme_root + r'/COR2B_CMES_dt01_dp15_t12_matched.csv'
        # Use some converter functions to get the time formats correct on import
        d = {'tb' : pd.to_datetime, 'tm' : pd.to_datetime,
             'tm_e' : pd.to_timedelta, 'tb_e' : pd.to_timedelta}
        # Import the data
        c1cme = pd.read_csv(c1_fn, index_col=0, converters=d)
        c2cme = pd.read_csv(c2_fn, index_col=0, converters=d)

        # Build empty dataframe for storing events in.
        eucme = pd.DataFrame({'tm' : [], 'tm_err' : [], 'l' : [], 'l_err' : [],
                              'b' : [], 'b_err' : [], 'w' : [], 'w_err' : [],
                              'h' : [], 'h_err' : [], 'N' : [], 'wav_284' : [],
                              'wav_304' : [], 'wav_195' : [], 'wav_171' : []})

        # Define the time span and window width to analyse.
        # The cor1 data studied by Stormwatch span the period 2007-02-28 to
        # 2007-02-16.
        window_width = 0.5 # Window block size (days).

        # Loop over the CMEs and pull out all euvi identifications within 12 hour
        # window before C1 start time
        tm = []
        tm_e = []
        l = []
        l_e = []
        b = []
        b_e = []
        w = []
        w_e = []
        h = []
        h_e = []
        N = []
        wav_284 = []
        wav_304 = []
        wav_195 = []
        wav_171 = []
        for i in range(len(c1cme)):
            # Get start time of C1 cme in JD
            ts = Time(c1cme['tb'][i], format='datetime')
            # Look for all euvi points in this window
            Cs = select_euvi_subset(C, time_start=ts.jd - window_width,
                                    time_stop=ts.jd, craft=craft)
            # If there are any matchable observations
            if len(Cs) > 0:
                Csav = Cs.mean()
                Cser = Cs.std()
                # Append the mean values and standard deviations as errors
                tm.append(Time(Csav['tm'], format='jd').datetime)
                tm_e.append(Cser['tm'])
                l.append(Csav['l'])
                l_e.append(Cser['l'])
                b.append(Csav['b'])
                b_e.append(Cser['b'])
                w.append(Csav['w'])
                w_e.append(Cser['w'])
                h.append(Csav['h'])
                h_e.append(Cser['h'])
                N.append(len(Cs))
                wav_284.append(len(Cs[Cs.wav == '284']))
                wav_304.append(len(Cs[Cs.wav == '304']))
                wav_195.append(len(Cs[Cs.wav == '195']))
                wav_171.append(len(Cs[Cs.wav == '171']))
            else:
                tm.append(np.NaN)
                tm_e.append(np.NaN)
                l.append(np.NaN)
                l_e.append(np.NaN)
                b.append(np.NaN)
                b_e.append(np.NaN)
                w.append(np.NaN)
                w_e.append(np.NaN)
                h.append(np.NaN)
                h_e.append(np.NaN)
                N.append(0)
                wav_284.append(0)
                wav_304.append(0)
                wav_195.append(0)
                wav_171.append(0)

        # Convert the lists of EUVI data into a pandas dataframe
        eucme = pd.DataFrame({'tm' : tm, 'tm_e' : tm_e, 'l' : l, 'l_e' : l_e,
                              'b' : b, 'b_e' : b_e, 'w' : w, 'w_e' : w_e,
                              'h' : h, 'h_e' : h_e, 'N' : N,
                              'wav_284' : wav_284, 'wav_304' : wav_304,
                              'wav_195' : wav_195, 'wav_171' : wav_171})
        eucme['tm_e'] = pd.to_timedelta(eucme['tm_e'], 'D')
        # Download the relevant euvi data
        #download_euvi_files(eucme, craft_vso, cme_root)
        # Get Carrington coordinates and images
        eucme = find_carr_coords(cme_root, craft, c1cme, eucme)

        # Save the dataframe to csv
        if craft == 'A':
            # Get updated filenames
            fn = cme_root + r'\EUVIA_CMES_matched.csv'
        elif craft == 'B':
            # Get updated filenames
            fn = cme_root + r'\EUVIB_CMES_matched.csv'
        eucme.to_csv(fn, sep=',')