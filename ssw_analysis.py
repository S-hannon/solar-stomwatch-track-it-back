from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.font_manager as font_manager
import scipy.stats as sps


def get_ssw_data(cme_root):
    """ Function to gather all the data from the separate csv files which make
    up the SSW event list. Returns 6 pandas dataframes:
    c1a : data for each event observed by STEREO-A in the COR1 field of view
    c1b : data for each event observed by STEREO-B in the COR1 field of view
    c2a : data for each event observed by STEREO-A in the COR2 field of view
    c2b : data for each event observed by STEREO-B in the COR2 field of view
    euvia : data for each event observed by STEREO-A in the EUVI field of view
    euvib : data for each event observed by STEREO-B in the EUVI field of view
    
    These dataframes have the following columns:
        
    c1a, c1b, c2a and c2b dataframes:
    tb : The time the CME entered the field of view (UTC)
    tb_e : The uncertainty on the time the CME entered the field of view (days)
    tm : The time the CME reached half way through the field of view (UTC)
    tm_e :  The uncertainty on the time the CME reached half way through the
        field of view (days)
    pa : The position angle of the CME (degrees), measured anti-clockwise from
        Solar North
    pa_e : The uncertainty of the position angle (degrees)
    wid : The apparent angular width of the CME (degrees)
    wid_e: The uncertainty on the angular width (degrees)
    n : The number of classifications which were clustered together to give the
        CME properties
    rb : The radial distance in the plane of the sky of the smallest non masked
        elongation along the row of pixels passing horizontally through the Sun
        center.
    rm : The radial distance in the plane of the sky of the median non masked
        elongation along the row of pixels passing horizontally through the Sun
        center.
    dr : The distance between the centre and and halfway through the field of
        view at the time of the CME
    dt : The time between the CME entering and reaching halfway through the
        field of view (s)
    dt_e : The uncertainty of the time difference dt (s)
    speed : The average speed of the CME between the time the CME entered the
        field of view and the time it reached half way through the field of
        view (km/s)
    speed_e : The uncertainty of the speeds (km/s)
    lat : The latitude of the CME (degrees)
        
    euvia and euvib dataframes:
    N: The number of classifications which were clustered together to give the
        CME properties
    tm : The time the CME appeared in the EUVI field of view (UTC)
    tm_e : The uncertainty on the time the CME appeared in the EUVI field of
        view (UTC)
    b, b_e, h, h_e, l, l_e, w, w_e : pixel coordinates describing the location
        of the box participants drew around the CME source region
    wav_171 : The number of classifications included for which the participant
        chose the EUVI image with a wavelength of 17.1nm
    wav_195 : The number of classifications included for which the participant
        chose the EUVI image with a wavelength of 19.5nm
    wav_284 : The number of classifications included for which the participant
        chose the EUVI image with a wavelength of 28.4nm
    wav_304 : The number of classifications included for which the participant
        chose the EUVI image with a wavelength of 30.4nm
    carrington_longitude : The Carrington longitude of the CME source region
    carrington_latitude : The Carrington latitude of the CME source region    
    """
    # Data files names
    c1_fn_a = cme_root + r'/COR1A_CMES_dt01_dp15_t07_matched.csv'
    c2_fn_a = cme_root + r'/COR2A_CMES_dt01_dp15_t12_matched.csv'
    c1_fn_b = cme_root + r'/COR1B_CMES_dt01_dp15_t07_matched.csv'
    c2_fn_b = cme_root + r'/COR2B_CMES_dt01_dp15_t12_matched.csv'
    euvi_fn_a = cme_root + r'/EUVIA_CMES_matched.csv'
    euvi_fn_b = cme_root + r'/EUVIB_CMES_matched.csv'
    # Use some converter functions to get the time formats correct on import
    d = {'tb': pd.to_datetime, 'tm': pd.to_datetime,
         'tm_e': pd.to_timedelta, 'tb_e': pd.to_timedelta}
    euvid = {'tm': pd.to_datetime, 'tm_e': pd.to_timedelta}
    # Read the csv files
    c1a = pd.read_csv(c1_fn_a, index_col=0, converters=d)
    c2a = pd.read_csv(c2_fn_a, index_col=0, converters=d)
    c1b = pd.read_csv(c1_fn_b, index_col=0, converters=d)
    c2b = pd.read_csv(c2_fn_b, index_col=0, converters=d)
    euvia = pd.read_csv(euvi_fn_a, index_col=0, converters=euvid)
    euvib = pd.read_csv(euvi_fn_b, index_col=0, converters=euvid)
    return c1a, c1b, c2a, c2b, euvia, euvib


def timing_uncertainties_plot(c2a, c2b, fig_dir):
    """Function to plot the timing uncertainties of the COR2 appearance vs. 
    midpoint times.
    """
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

    # Midpoint and appearance time errors comparison barplot COR2
    plt.figure()

    # Plot COR2 appearance and midpoint error bars
    x = (np.append(c2a.tm_e, c2b.tm_e)/np.timedelta64(1, 's'))/3600
    y = (np.append(c2a.tb_e, c2b.tb_e)/np.timedelta64(1, 's'))/3600
    n_groups = len(x)
    index = np.arange(n_groups)
    bar_width = 0.3
    plt.bar(index, x, bar_width, color='b', label='Mid-point')
    plt.bar(index + bar_width, y, bar_width, color='g', label='Appearance')
    
    # Format
    big_font = {'size':'16'}
    small_font = {'size':'12'}
    font_prop = font_manager.FontProperties(size=12)
    plt.xlabel('Event Number', **big_font)
    plt.ylabel('Timing Uncertainty (hours)', **big_font)
    plt.xticks(index, range(0, len(x)), **small_font)
    plt.yticks(**small_font)
    plt.legend(loc=2, prop=font_prop)
    plt.xlim(0, n_groups + bar_width)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(fig_dir + r'/Timing_Uncertainties.png')


def latitude_plot(c1a, c1b, c2a, c2b, fig_dir):
    """Function to plot SSW latitudes; shows a histogram of COR1 and COR2
    latitudes, a scatter plot showing COR1 and COR2 latitudes, and a scatter
    plot ot the COR2-COR1 latitude differences.
    """
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    # Set subplots
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    small_font = {'size':'12'}

    # First subplot: histogram of COR1 and COR2 latitudes
    ax1.hist(np.append(c1a.lat, c1b.lat), bins=np.arange(-65, 30+1, 10),
             alpha=0.5, label='COR1')
    ax1.hist(np.append(c2a.lat, c2b.lat), bins=np.arange(-65, 30+1, 10),
             alpha=0.5, label='COR2')
    ax1.set_xlabel('Latitude (degrees)', **small_font)
    ax1.set_ylabel('Frequency', **small_font)
    ax1.legend(loc=0)
    
    # Second subplot: COR1 vs. COR2 latitudes
    ax2.scatter(c1a.lat, c2a.lat, color='r', label='STA')
    ax2.errorbar(c1a.lat, c2a.lat, xerr=c1a.pa_e, yerr=c2a.pa_e, color='r',
                 ls='none', label=None)
    ax2.scatter(c1b.lat, c2b.lat, color='b', label='STB')
    ax2.errorbar(c1b.lat, c2b.lat, xerr=c1b.pa_e, yerr=c2b.pa_e, color='b',
                 ls='none', label=None)
    ax2.set_xlabel('COR1 Latitude (degrees)', **small_font)
    ax2.set_ylabel('COR2 Latitude (degrees)', **small_font)
    ax2.legend(scatterpoints=1, loc=0)
    ax2.axhline(0, color='k')
    ax2.axvline(0, color='k')
    ax2.set_xlim((-90, 90))
    ax2.set_ylim((-90, 90))
    ax2.plot([-90,90],[-90,90],'k--')

    # Final subplot: COR2 to COR1 latitude differences
    lat_d = np.append(c2a.lat, c2b.lat) - np.append(c1a.lat, c1b.lat)
    lat_d_e = np.sqrt((np.append(c2a.pa_e, c2b.pa_e)**2) +
                      (np.append(c1a.pa_e, c1b.pa_e)**2))
    x = range(0, len(lat_d))
    ax3.scatter(x, lat_d)
    ax3.errorbar(x, lat_d, yerr=lat_d_e, color='k', ls='none')
    ax3.axhline(0, color='k')
    ax3.set_xlabel('Event Number', **small_font)
    ax3.set_ylabel('COR2-COR1 Latitude Difference (degrees)', **small_font)

    # Save plot
    plt.tight_layout()
    plt.savefig(fig_dir + r'/Latitude Plots')

    
def width_plot(c1a, c1b, c2a, c2b, fig_dir):
    """Function to plot SSW widths; shows a histogram of COR1 and COR2 widths,
    a scatter plot showing COR1 and COR2 widths, and a scatter plot ot the
    COR2-COR1 width differences.
    """
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    # Set subplots
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    small_font = {'size':'12'}

    # First subplot: histograms of COR1 and COR2 widths
    ax1.hist(np.append(c1a.wid, c1b.wid), bins=np.arange(0, 100+1, 10),
             alpha=0.5, label='COR1')
    ax1.hist(np.append(c2a.wid, c2b.wid), bins=np.arange(0, 100+1, 10),
             alpha=0.5, label='COR2')
    ax1.set_xlabel('Width (degrees)', **small_font)
    ax1.set_ylabel('Frequency', **small_font)
    ax1.legend(loc=0)

    # Second subplot: COR1 vs. COR2 widths
    ax2.scatter(c1a.wid, c2a.wid, color='r', label='STA')
    ax2.errorbar(c1a.wid, c2a.wid, xerr=c1a.pa_e, yerr=c2a.pa_e, color='r',
                 ls='none', label=None)
    ax2.scatter(c1b.wid, c2b.wid, color='b', label='STB')
    ax2.errorbar(c1b.wid, c2b.wid, xerr=c1b.wid_e, yerr=c2b.wid_e, color='b',
                 ls='none', label=None)
    ax2.set_xlabel('COR1 Width (degrees)', **small_font)
    ax2.set_ylabel('COR2 Width (degrees)', **small_font)
    ax2.legend(scatterpoints=1, loc=0)
    ax2.axhline(0, color='k')
    ax2.axvline(0, color='k')
    ax2.set_xlim((0, 100))
    ax2.set_ylim((0, 100))
    ax2.plot((0, 100), (0, 100), ls="--", c=".3")

    # Final subplot: COR2-COR1 width differences
    wid_d = np.append(c2a.wid, c2b.wid) - np.append(c1a.wid, c1b.wid)
    wid_d_e = np.sqrt((np.append(c2a.wid_e, c2b.wid_e)**2) +
                      (np.append(c1a.wid_e, c1b.wid_e)**2))
    x = range(0, len(wid_d))
    ax3.scatter(x, wid_d)
    ax3.errorbar(x, wid_d, yerr=wid_d_e, color='k', ls='none')
    ax3.set_xlabel('Event Number', **small_font)
    ax3.set_ylabel('COR2-COR1 Width Difference (degrees)', **small_font)

    # Save plot
    plt.tight_layout()
    plt.savefig(fig_dir + r'/Width Plots.png')


def speed_plot(c1a, c1b, c2a, c2b, fig_dir):
    """Function to plot SSW speeds; shows a histogram of COR1 and COR2 speeds,
    a scatter plot showing COR1 and COR2 speeds, and a scatter plot ot the
    COR2-COR1 speed differences.
    """
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    # Set subplots
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    small_font = {'size':'12'}

    # First subplot: histogram of COR2 Speeds
    ax1.hist(np.append(c1a.speed, c1b.speed), bins=np.arange(0, 1000+1, 50),
             alpha=0.5, label='COR1')
    ax1.hist(np.append(c2a.speed, c2b.speed), bins=np.arange(0, 1000+1, 50),
             alpha=0.5, label='COR2')
    ax1.set_xlabel('Speed (kms$^{-1}$)', **small_font)
    ax1.set_ylabel('Frequency', **small_font)
    ax1.legend(loc=0)

    # Second subplot: COR1 vs. COR2 Speeds
    ax2.scatter(c1a.speed, c2a.speed, color='r', label='STA')
    ax2.errorbar(c1a.speed, c2a.speed, xerr=c1a.speed_e, yerr=c2a.speed_e,
                 color='r', ls='none', label=None)
    ax2.scatter(c1b.speed, c2b.speed, color='b', label='STB')
    ax2.errorbar(c1b.speed, c2b.speed, xerr=c1b.speed_e, yerr=c2b.speed_e,
                 color='b', ls='none', label=None)
    ax2.set_xlabel('COR1 Speed (kms$^{-1}$)', **small_font)
    ax2.set_ylabel('COR2 Speed (kms$^{-1}$)', **small_font)
    ax2.legend(scatterpoints=1, loc=0)
    ax2.set_xlim((-250, 650))
    ax2.set_ylim((-250, 1400))
    ax2.plot((-250, 1300), (-250, 1300), ls="--", c=".3")

    # Final subplot: COR2 to COR1 speed differences
    speed_d = np.append(c2a.speed, c2b.speed) - np.append(c1a.speed, c1b.speed)
    speed_d_e = np.sqrt((np.append(c2a.speed_e, c2b.speed_e)**2) +
                        (np.append(c1a.speed_e, c1b.speed_e)**2))
    x = range(0, len(speed_d))
    ax3.scatter(x, speed_d)
    ax3.errorbar(x, speed_d, yerr=speed_d_e, color='k', ls='none')
    ax3.axhline(0, color='k')
    ax3.set_xlabel('Event Number', **small_font)
    ax3.set_ylabel('COR2-COR1 Speed Difference (kms$^{-1}$)', **small_font)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(fig_dir + r'/Speed Plots.png')


def time_dist_plot(x_df, x_limit, title, fig_dir):
    """ Creates a time-distance plot.
    x = vector of the 4 points to be plotted in order: COR1 appearance time,
    COR1 midpoint time, COR2 appearance time, COR2 midpoint time
    x_errors = error bars
    title = title for the plot
    x_limit = set length of x axis
    """
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    # Initiate figure
    plt.figure()
    labels = ['EUVI Appearance', 'COR1 Appearance', 'COR1 Midpoint',
              'COR2 Appearance', 'COR2 Midpoint']
    # y estimates:
    y = [1, 1.5, 2.65, 2.5, 8.75]
    # Set colours to be yellow for EUVI, blue for COR1 and red for COR2
    colours = ['y', 'b', 'b', 'r', 'r']
    # Set markers to be triangle for EUVI appearance time, circle for COR1/COR2
    # start times and square for COR1/COR2 mid-point times
    markers = ['^', 'o', 's', 'o', 's']

    # Plot
    for i in range(len(x_df)):
        if len(x_df) == 5:
            j = i
        else:
            j = i+1
        plt.scatter(x_df.x[i], y[i], color=colours[j], marker=markers[j],
                    label=labels[j])
        plt.errorbar(x_df.x[i], y[i], xerr=x_df.x_err[i], color=colours[j],
                     ls='none', label=None)
    
    # Format
    plt.legend(loc=0, scatterpoints=1)
    plt.xlim(x_limit)
    plt.xticks(rotation='vertical')
    big_font = {'size':'16'}
    plt.xlabel('Time', **big_font)
    plt.ylabel('Distance (R$_\odot$)', **big_font)
    plt.title(title)
    ax = plt.gca()
    date_formatter = mdate.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    
    # Save
    name = fig_dir + r'/SS_' + title + '.png'
    plt.savefig(name)
    return


def get_time_dist_data(euvi, cor1, cor2, craft, i):
    """ Extracts the relevant data for time distance plots.
    """
    # Get points for CME i
    x = []
    x_err = []
    # Check for EUVI data
    if pd.isnull(euvi.tm[i]) == False:
        if pd.isnull(euvi.tm_e[i]) == False:
            x.append(euvi.tm[i])
            x_err.append(euvi.tm_e[i])
    # Append COR1 and COR2 data
    x.append(cor1.tb[i])
    x.append(cor1.tm[i])
    x.append(cor2.tb[i])
    x.append(cor2.tm[i])
    x_err.append(cor1.tb_e[i])
    x_err.append(cor1.tm_e[i])
    x_err.append(cor2.tb_e[i])
    x_err.append(cor2.tm_e[i])
    x_df = pd.DataFrame({'x' : x, 'x_err' : x_err})
    # Set Title
    title = '%s CME %s' % (craft, i)
    # x axis limits
    x_check = x_df
    x_check['start'] = pd.Series(x_df.x - x_df.x_err, index=x_df.index)
    x_check['end'] = pd.Series(x_df.x + x_df.x_err, index=x_df.index)
    starts = x_check.sort_values(by='start')
    start = x_check.start[starts.index[0]]
    ends = x_check.sort_values(by='end')
    end = x_check.end[ends.index[len(x_check)-1]]
    x_limit = (start - np.timedelta64(30,'m'), end + np.timedelta64(30,'m'))
    return x_df, x_limit, title

    
def time_dist_all_plots(euvia, euvib, c1a, c1b, c2a, c2b, fig_dir):
    """Creates and saves time distance plots for all events.
    """
    for i in range(len(c2a)):
        x_df, x_limit, title = get_time_dist_data(euvia, c1a, c2a, 'STEREO-A', i)
        time_dist_plot(x_df, x_limit, title, fig_dir)
    for i in range(len(c2b)):
        x_df, x_limit, title = get_time_dist_data(euvib, c1b, c2b, 'STEREO-B', i)
        time_dist_plot(x_df, x_limit, title, fig_dir)


def seeds_cactus_comparison_plot(c2a, c2b, seeds_a, seeds_b, cactus_a, cactus_b, fig_dir):
    """Function to plot the differences in position angles and angular widths
    of SSW events between the SSW, SEEDS and CACTus catalogues.
    """
    # Find differences
    seeds_pa_d = np.append(c2a.pa, c2b.pa) - np.append(seeds_a.pa, seeds_b.pa)
    cactus_pa_d = np.append(c2a.pa, c2b.pa) - np.append(cactus_a.pa, 
                                                        cactus_b.pa)
    seeds_wid_d = np.append(c2a.wid, c2b.wid) - np.append(seeds_a.wid,
                                                          seeds_b.wid)
    cactus_wid_d = np.append(c2a.wid, c2b.wid) - np.append(cactus_a.wid,
                                                           cactus_b.wid)
    
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 7
    fig_size[1] = 7
    plt.rcParams["figure.figsize"] = fig_size

    # Set subplots and shared axis
    f, axarr = plt.subplots(2, sharex=True)
    small_font = {'size':'12'}
    plt.xlabel('Event Number', **small_font)
    x = np.arange(0, len(seeds_pa_d))

    # First subplot: comparing position angles
    # SEEDS points
    axarr[0].scatter(x, seeds_pa_d, color='g', label='SSW-SEEDS')
    axarr[0].errorbar(x, seeds_pa_d, yerr=np.append(c2a.pa_e, c2b.pa_e),
                      color='g', ls='none')
    # CACTus points
    axarr[0].scatter(x, cactus_pa_d, color='y', label='SSW-CACTus')
    axarr[0].errorbar(x, cactus_pa_d, yerr=np.append(c2a.pa_e, c2b.pa_e),
                      color='y', ls='none')
    # Format
    axarr[0].axhline(0, color='k')
    axarr[0].legend(loc=0, scatterpoints=1)
    axarr[0].set_ylabel('Position Angle Difference (degrees)', **small_font)
    axarr[0].set_ylim((-60, 60))

    # Second subplot: comparing widths
    # SEEDS points
    axarr[1].scatter(x, seeds_wid_d, color='g', label='SSW-SEEDS')
    axarr[1].errorbar(x, seeds_wid_d, yerr=np.append(c2a.wid_e, c2b.wid_e),
                      color='g', ls='none')
    # CACTus points
    axarr[1].scatter(x, cactus_wid_d, color='y', label='SSW-CACTus')
    axarr[1].errorbar(x, cactus_wid_d, yerr=np.append(c2a.wid_e, c2b.wid_e),
                      color='y', ls='none')
    # Format
    axarr[1].axhline(0, color='k')
    axarr[1].legend(loc=0, scatterpoints=1)
    axarr[1].set_ylabel('Apparent Width Difference (degrees)', **small_font)
    axarr[1].set_ylim((-120, 120))
    
    # Save the plot into figure directory
    plt.tight_layout()
    plt.savefig(fig_dir + r'/SEEDS_CACTus_Comparisons.png')


def get_deflections_data(c1a, c1b, c2a, c2b, defs_root):
    """ This is a function to import relevant data from the Solar Stormwatch
    event list and model data of the location of the heliospheric current sheet
    returning the data in 4 pandas dataframes: da, db, lda and ldb. da and db
    contains data on all events observed by STEREO-A with corresponding EUVI
    data, and db data for events observed by STEREO-B. lda and ldb are subsets
    of da and db for events which have been identified as 'limb' events -
    events with source regions at the edge of the solar disk, as seen in the
    coronagraph images. defs_root is the location of the deflections data.

    The output pandas dataframes contain the following fields:
    carrington_rotation: The Carrington rotation during which the CME occured
    carrington_longitude: The Carrington longitude of the CME source region
    carrington_latitude: The Carrington latitude of the CME source region
    r_150: The latitude of the heliospheric current sheet (HCS) at the source
        longitude of the CME, at a distance of 1.50 solar radii
    r_250: The latitude of the HCS at the source longitude of the CME, at a
        distance of 2.50 solar radii
    r_275: The latitude of the HCS at the source longitude of the CME, at a
        distance of 2.75 solar radii
    r_875: The latitude of the HCS at the source longitude of the CME, at a
        distance of 8.75 solar radii
    r_3000: The latitude of the HCS at the source longitude of the CME, at a
        distance of 30.00 solar radii
    pa_change: The change in the central position angle of the CME between the
        COR1 and COR2 fields of view in degrees.
    pa_change_e: The error on the change of the central position angle in
        degrees.
    lat_dist_hcs: The latitudinal distance between the heliospheric current
        sheet and the CME source region
    hcs_e_p: An estimate of the errors of lat_dist_hcs - the difference between
        the latitudinal distance between the heleiospheric current sheet and
        CME source region at the CME source longitude, and a longitude 10
        degrees higher.
    hcs_e_n: An estimate of the errors of lat_dist_hcs - the difference between
        the latitudinal distance between the heleiospheric current sheet and
        CME source region at the CME source longitude, and a longitude 10
        degrees lower.
    """
    # Read in all the data sets
    da = pd.read_csv(defs_root + r'/STA_CMEs_HCS.csv')
    db = pd.read_csv(defs_root + r'/STB_CMEs_HCS.csv')
    da_plus10 = pd.read_csv(defs_root + r'/STA_CMEs_HCS_plus10.csv')
    da_minus10 = pd.read_csv(defs_root + r'/STA_CMEs_HCS_minus10.csv')
    db_plus10 = pd.read_csv(defs_root + r'/STB_CMEs_HCS_plus10.csv')
    db_minus10 = pd.read_csv(defs_root + r'/STB_CMEs_HCS_minus10.csv')

    # Calculate plane-of-sky deflections and errors
    da['pa_change'] = pd.Series((c2a.lat - c1a.lat), index=da.index)
    db['pa_change'] = pd.Series((c2b.lat - c1b.lat), index=db.index)
    da['pa_change_e'] = pd.Series(np.sqrt(((c2a.pa_e)**2) + ((c1a.pa_e)**2)), index=da.index)
    db['pa_change_e'] = pd.Series(np.sqrt(((c2b.pa_e)**2) + ((c1b.pa_e)**2)), index=db.index)

    # Calculate latitudinal distances to HCS & errors
    da['lat_dist_hcs'] = pd.Series((da.carrington_latitude - da.r_875), index=da.index)
    db['lat_dist_hcs'] = pd.Series((db.carrington_latitude - db.r_875), index=db.index)
    da['hcs_e_p'] = pd.Series((da_plus10.r_875 - da.r_875), index=da.index)
    db['hcs_e_p'] = pd.Series((db_plus10.r_875 - db.r_875), index=db.index)
    da['hcs_e_n'] = pd.Series((da_minus10.r_875 - da.r_875), index=da.index)
    db['hcs_e_n'] = pd.Series((db_minus10.r_875 - db.r_875), index=db.index)

    # Drop rows with no EUVI data
    da = da.drop([0, 1, 3, 4])
    db = db.drop([2, 3, 10])

    # Pick out limb events
    lda = da.drop([5, 9, 14, 15, 16, 17, 18, 19])
    ldb = db.drop([0, 4, 8, 12, 13, 14, 15, 18, 19, 20])

    return da, db, lda, ldb


def deflections_plot(da, db, lda, ldb, fig_dir):
    """A function to plot plane-of-sky latitude deflection against latitudinal
    distance from heliospheric current sheet. Saves a plot into the figure
    directory with two subplots: all events, and limb events only.
    """
    
    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size

    # Font properties
    big_font = {'size':'16'}

    # Specify subplots
    f, axarr = plt.subplots(2, sharex=True)
    plt.xlabel('Latitudinal Distance from HCS at 8.75 R$_\odot$', **big_font)

    # ALL DEFLECTIONS
    # Plot STA points
    axarr[0].scatter(da.lat_dist_hcs, da.pa_change, color='r', label='STEREO-A')
    axarr[0].errorbar(da.lat_dist_hcs, da.pa_change,
                 xerr=[abs(da.hcs_e_p), abs(da.hcs_e_n)],
                       yerr=da.pa_change_e.values, color='r', ls='none', label=None)
    # Plot STB points
    axarr[0].scatter(db.lat_dist_hcs, db.pa_change, color='b', label='STEREO-B')
    axarr[0].errorbar(db.lat_dist_hcs, db.pa_change,
                 xerr=[abs(db.hcs_e_p), abs(db.hcs_e_n)],
                       yerr=db.pa_change_e.values, color='b', ls='none', label=None)
    axarr[0].set_ylabel('Deflection (degrees)', **big_font)
    axarr[0].legend(loc=0, scatterpoints=1)
    axarr[0].axhline(0, color='k')
    axarr[0].axvline(0, color='k')
    # linear fit
    x = np.append(da.lat_dist_hcs, db.lat_dist_hcs)
    y = np.append(da.pa_change, db.pa_change)
    slope, intercept, r_value, p_value, slope_std_error = sps.linregress(x, y)
    predict_y = intercept + slope * x
    print 'x-value when y=0: %s' % (-intercept/slope)
    axarr[0].plot(x, predict_y, 'k-')
    print 'p-value: %s' % (p_value)
    print 'intercept: %s' % (intercept)

    # LIMB EVENTS ONLY
    # Plot STA points
    axarr[1].scatter(lda.lat_dist_hcs, lda.pa_change, color='r', label='STEREO-A')
    axarr[1].errorbar(lda.lat_dist_hcs, lda.pa_change,
                 xerr=[abs(lda.hcs_e_p), abs(lda.hcs_e_n)],
                       yerr=lda.pa_change_e.values, color='r', ls='none', label=None)
    # Plot STB points
    axarr[1].scatter(ldb.lat_dist_hcs, ldb.pa_change, color='b', label='STEREO-B')
    axarr[1].errorbar(ldb.lat_dist_hcs, ldb.pa_change,
                 xerr=[abs(ldb.hcs_e_p), abs(ldb.hcs_e_n)],
                       yerr=ldb.pa_change_e.values, color='b', ls='none', label=None)
    axarr[1].set_ylabel('Deflection (degrees)', **big_font)
    axarr[1].legend(loc=0, scatterpoints=1)
    axarr[1].axhline(0, color='k')
    axarr[1].axvline(0, color='k')
    # linear fit
    x = np.append(lda.lat_dist_hcs, ldb.lat_dist_hcs)
    y = np.append(lda.pa_change, ldb.pa_change)
    slope, intercept, r_value, p_value, slope_std_error = sps.linregress(x, y)
    predict_y = intercept + slope * x
    print 'x-value when y=0: %s' % (-intercept/slope)
    axarr[1].plot(x, predict_y, 'k-')
    print 'p-value: %s' % (p_value)
    print 'intercept: %s' % (intercept)

    # Save plot into figure directory
    plt.tight_layout()
    plt.savefig(fig_dir + 'Deflection_Plots' + '.png')
    