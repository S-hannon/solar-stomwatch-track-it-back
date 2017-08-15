import pandas as pd
import matplotlib.pyplot as plt
import ssw_comparisons as cp


def import_ssw_cme_list(file_route, root):
    """Creates a pandas dataframe from the data in the file. file_route is the
    name of the file in the thresholding_data folder to investigate.
    """
    # Get the file
    data = root + file_route
    d = {'tb_utc': pd.to_datetime, 'tm_utc': pd.to_datetime,
     'tm_e': pd.to_timedelta, 'tb_e': pd.to_timedelta}
    df = pd.read_csv(data, index_col=0, converters=d)
    # Change the tm and tb columns to UTC time instead of Julian date
    df['tm'] = pd.Series(df.tm_utc, index=df.index)
    df['tb'] = pd.Series(df.tb_utc, index=df.index)
    return df

    
def create_cor1_summary(runs, th_dir):
    """Creates a summary of the runs to trial with thresholds specified. runs
    should be in the format e.g. [2.0, 1.0, 15.0, 1.0, 15.0], which specifies a
    rolling time window of 1 hour, a rolling position angle of 15 degrees, and
    a minimum of 5 classifications per CME cluster.
    
    The function saves the results to a csv file with columns:
    dp : rolling position angle width of the run
    dt : rolling time window width of the run
    run : the function input e.g. 'dt01_dp15_t05',
    th : the minimum number of classifications per CME cluster
    total_ssw : the number of CMEs the code finds from the COR1 classification
        data using the thresholds specified.
    """
    # Empty lists to store data
    run = []
    dt = []
    dp = []
    th = []
    total_ssw = []
    # Populate fields with relevant data
    for i in range(0, len(runs)):
        name = 'dt' + str(int(runs[i][3])).zfill(2) + '_dp' + str(int(runs[i][2])).zfill(2) + '_t' + str(int(runs[i][4])).zfill(2)
        run.append(name)
        dp.append(runs[i][2])
        dt.append(runs[i][3])
        th.append(runs[i][4])
        # Get run data for each trial
        name_a = '/Data\COR1A_CMES_' + name + '.csv'
        name_b = '/Data\COR1B_CMES_' + name + '.csv'
        data_a = import_ssw_cme_list(name_a, th_dir)
        data_b = import_ssw_cme_list(name_b, th_dir)
        total_ssw.append(len(data_a) + len(data_b))
        
    print run
    # Convert into a pandas dataframe
    df = pd.DataFrame({'run': run, 'dt' : dt, 'dp' : dp, 'th' : th,
                       'total_ssw': total_ssw})
    # Save!
    filename = th_dir + r'\COR1_Summary.csv'
    df.to_csv(filename, sep=',')
    return df
    

def seeds_cont_values(cor2a, cor2b, data_dir, value=False):
    """ Matches the events from a given SSW event list to events in the SEEDS
    catalogue with a start time up to 12 hours before the SSW mid-point time
    given the position angles match within 45 degrees. Returns the number of
    SSW events, the number of SEEDS events within the matching time frame, and
    the number of matches. If value is set to True the function will print the
    indexes and times of the SSW events which do not match to SEEDS.
    """
    # Find total SSW events
    total_ssw = len(cor2a) + len(cor2b)
    # Get SEEDS COR2 catalogue data
    seeds_a_all, seeds_b_all = cp.get_seeds_data(data_dir)
    # Find total SEEDS events in time frame
    seeds_a_t = seeds_a_all
    seeds_a_t = seeds_a_t.set_index(['time'])
    seeds_a_tf = seeds_a_t.loc['2007-2-28':'2010-2-12']
    seeds_b_t = seeds_b_all
    seeds_b_t = seeds_b_t.set_index(['time'])
    seeds_b_tf = seeds_b_t.loc['2007-2-28':'2010-2-12']
    total_seeds = len(seeds_a_tf) + len(seeds_b_tf)
    # Match the events from SSW to comparison data
    index_list_a, index_list_b = cp.match_to(cor2a, cor2b, seeds_a_all,
                                             seeds_b_all)
    # How many matches are there?
    match_times_a = cp.get_cme_data(index_list_a, seeds_a_all.time)
    match_times_b = cp.get_cme_data(index_list_b, seeds_b_all.time)
    # Find total matched events
    c2a_unmatched = cp.check_match_times(cor2a.tm, match_times_a, value=value)
    c2b_unmatched = cp.check_match_times(cor2b.tm, match_times_b, value=value)
    total_matches = total_ssw - c2a_unmatched - c2b_unmatched
    # Print contingency table values
    print 'Total events in SSW and SEEDS: %s' % (total_matches)
    print 'Total events in SSW not SEEDS: %s' % (total_ssw - total_matches)
    print 'Total events in SEEDS not SSW: %s' % (total_seeds - total_matches)
    return total_matches, total_ssw, total_seeds


def cactus_cont_values(cor2a, cor2b, data_dir, value=False):
    """ Matches the events from a given SSW event list to events in the CACTus
    catalogue with a start time up to 12 hours before the SSW mid-point time
    given the position angles match within 45 degrees. Returns the number of
    SSW events, the number of CACTus events within the matching time frame, and
    the number of matches. If value is set to True the function will print the
    indexes and times of the SSW events which do not match to CACTus.
    """
    # Find total SSW events
    total_ssw = len(cor2a) + len(cor2b)
    # Get CACTus COR2 catalogue data
    cactus_a_all, cactus_b_all = cp.get_cactus_data(data_dir)
    # Find total CACTus events in time frame
    cactus_a_t = cactus_a_all
    cactus_a_t = cactus_a_t.set_index(['time'])
    cactus_a_tf = cactus_a_t.loc['2007-2-28':'2010-2-12']
    cactus_b_t = cactus_b_all
    cactus_b_t = cactus_b_t.set_index(['time'])
    cactus_b_tf = cactus_b_t.loc['2007-2-28':'2010-2-12']
    total_cactus = len(cactus_a_tf) + len(cactus_b_tf)
    # Match the events from SSW to comparison data
    index_list_a, index_list_b = cp.match_to(cor2a, cor2b, cactus_a_all,
                                             cactus_b_all)
    # How many matches are there?
    match_times_a = cp.get_cme_data(index_list_a, cactus_a_all.time)
    match_times_b = cp.get_cme_data(index_list_b, cactus_b_all.time)
    # Find total matched events
    c2a_unmatched = cp.check_match_times(cor2a.tm, match_times_a, value=value)
    c2b_unmatched = cp.check_match_times(cor2b.tm, match_times_b, value=value)
    total_matches = total_ssw - c2a_unmatched - c2b_unmatched
    # Print contingency table values
    print 'Total events in SSW and CACTus: %s' % (total_matches)
    print 'Total events in SSW not CACTus: %s' % (total_ssw - total_matches)
    print 'Total events in CACTus not SSW: %s' % (total_cactus - total_matches)
    return total_matches, total_ssw, total_cactus


def create_cor2_summary(runs, data_dir, th_dir):
    """Creates a summary of the runs to trial with thresholds specified.runs
    should be in the format e.g. 'dt01_dp15_t05', which specifies a rolling
    time window of 1 hour, a rolling position angle of 15 degrees, and a
    minimum of 5 classifications per CME cluster. This shows the number of
    matched events between the SSW events and the events in the SEEDS and
    CACTus catalogues.

    The function saves the results to a csv file with columns:
    dp : rolling position angle width of the run
    dt : rolling time window width of the run
    run : the function input e.g. 'dt01_dp15_t05',
    th : the minimum number of classifications per CME cluster
    total_ssw : the number of CMEs the code finds from the COR1 classification
        data using the thresholds specified.
    total_cactus : the number of CMEs the CACTus CME catalogue lists during
        the same time period as the SSW classifications.
    total_seeds : the number of CMEs the SEEDS CME catalogue lists during
        the same time period as the SSW classifications.
    ssw_cactus_matches : number of even matched between the CACTus and SSW
        catalogues
    ssw_seeds_matches : number of even matched between the SEEDS and SSW
        catalogues
    seeds_cactus_matches : number of even matched between the CACTus and SEEDS
        catalogues
    ssw_not_seeds : the number of events that SSW identifies which do not match
        to an event in the SEEDS catalogue
    seeds_not_ssw : the number of events that SEEDS identifies which do not
        match to an event in the SSW catalogue
    ssw_not_cactus : the number of events that SSW identifies which do not
        match to an event in the CACTus catalogue
    cactus_not_ssw : the number of events that CACTus identifies which do not
        match to an event in the SSW catalogue
    ssw_seeds_ratio : the ratio of events matched between the SSW and SEEDS
        catalogues to non-matched events
    ssw_cactus_ratio : the ratio of events matched between the SSW and CACTus
        catalogues to non-matched events
    seeds_cactus_ratio : the ratio of events matched between the SEEDS and
        CACTus catalogues to non-matched events
    cactus_seeds_ratio : the ratio of events matched between the CACTus and
        SEEDS catalogues to non-matched events
    """
    # Empty lists to store data
    run = []
    dt = []
    dp = []
    th = []
    ssw_seeds_matches = []
    total_ssw = []
    total_seeds = []
    ssw_cactus_matches = []
    total_cactus = []

    # Add in the data for each run
    for i in range(0, len(runs)):
        # Get run data
        name = 'dt' + str(int(runs[i][3])).zfill(2) + '_dp' + str(int(runs[i][2])).zfill(2) + '_t' + str(int(runs[i][4])).zfill(2)
        run.append(name)
        dp.append(runs[i][2])
        dt.append(runs[i][3])
        th.append(runs[i][4])
        print 'Thresholds: %s' % (name)
        name_a = '/Data\COR2A_CMES_' + name + '.csv'
        name_b = '/Data\COR2B_CMES_' + name + '.csv'
        data_a = import_ssw_cme_list(name_a, th_dir)
        data_b = import_ssw_cme_list(name_b, th_dir)
        # Get contingency values for this run
        seeds = seeds_cont_values(data_a, data_b, data_dir)
        ssw_seeds_matches.append(seeds[0])
        total_ssw.append(seeds[1])
        total_seeds.append(seeds[2])
        cactus = cactus_cont_values(data_a, data_b, data_dir)
        ssw_cactus_matches.append(cactus[0])
        total_cactus.append(cactus[2])
        print

    # Convert into a pandas dataframe
    df = pd.DataFrame({'run': run, 'dt' : dt, 'dp' : dp, 'th' : th,
                       'ssw_seeds_matches': ssw_seeds_matches,
                       'total_ssw': total_ssw,
                       'total_seeds': total_seeds,
                       'ssw_cactus_matches': ssw_cactus_matches,
                       'total_cactus': total_cactus})

    # Add in some extra columns
    df['ssw_not_seeds'] = pd.Series(df.total_ssw - df.ssw_seeds_matches,
                                    index=df.index)
    df['seeds_not_ssw'] = pd.Series(df.total_seeds - df.ssw_seeds_matches,
                                    index=df.index)
    df['ssw_not_cactus'] = pd.Series(df.total_ssw - df.ssw_cactus_matches,
                                     index=df.index)
    df['cactus_not_ssw'] = pd.Series(df.total_cactus - df.ssw_cactus_matches,
                                     index=df.index)
    df['ssw_seeds_ratio'] = pd.Series(df.ssw_not_seeds / df.ssw_seeds_matches,
                                  index=df.index)
    df['ssw_cactus_ratio'] = pd.Series(df.ssw_not_cactus / df.ssw_cactus_matches,
                                   index=df.index)

    # Get SEEDS COR2 catalogue data
    seeds_a_all, seeds_b_all = cp.get_seeds_data(data_dir)
    # SEEDS vs CACTus contingency values
    sc_matches = cactus_cont_values(seeds_a_all, seeds_b_all, data_dir)
    # Add to dataframe
    df['seeds_cactus_matches'] = pd.Series(sc_matches[0], index=df.index)
    df['seeds_cactus_ratio'] = pd.Series((df.total_seeds - df.seeds_cactus_matches) / df.seeds_cactus_matches,
                                          index=df.index)
    df['cactus_seeds_ratio'] = pd.Series((df.total_cactus - df.seeds_cactus_matches) / df.seeds_cactus_matches,
                                          index=df.index)

    # Save to csv
    file_name = th_dir + r'/Contingency_Values_COR2.csv'
    df.to_csv(file_name, sep=',')
    return df


def cor2_combined_threshold_plot(df, fig_dir):
    """Creates three subplots of thresholding values from df.
    """
    # Select correct data
    dt_ok = df[df.dt == '01']
    dt_dp_ok = dt_ok[dt_ok.dp == '15']
    dt_dp_ok['th'] = dt_dp_ok['th'].astype('float')
    df = dt_dp_ok

    # Change figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size

    # Font properties
    font = {'size':'14'}

    # Three plots
    f, axarr = plt.subplots(3, sharex=True)
    plt.xlabel('Number of observations required for an event', **font)

    # First subplot: Events matched between the SSW and SEEDS / CACTus lists
    axarr[0].scatter(df.th.values, df.ssw_seeds_matches.values, color='r',
                     label='SEEDS')
    axarr[0].scatter(df.th.values, df.ssw_cactus_matches.values, color='b',
                     label='CACTus')
    axarr[0].legend(loc=1, scatterpoints=1)
    axarr[0].set_ylabel('SSW events matched', **font)

    # Second subplot: SSW events not matched to the SEEDS / CACTus lists
    axarr[1].scatter(df.th.values, df.ssw_not_seeds.values, color='r',
                     label='SEEDS')
    axarr[1].scatter(df.th.values, df.ssw_not_cactus.values, color='b',
                     label='CACTus')
    axarr[1].legend(loc=1, scatterpoints=1)
    axarr[1].set_ylabel('SSW events not matched', **font)

    # Final subplot: Ratio of matched to non-matched events
    axarr[2].scatter(df.th.values, df.ssw_seeds_ratio.values, color='r',
                     label='SSW/SEEDS')
    axarr[2].scatter(df.th.values, df.ssw_cactus_ratio.values, color='b',
                     label='SSW/CACTus')
    axarr[2].axhline(df.seeds_cactus_ratio.values[0], color='g',
                     label='SEEDS/CACTus')
    axarr[2].axhline(df.cactus_seeds_ratio.values[0], color='y',
                     label='CACTus/SEEDS')
    axarr[2].legend(loc=1, scatterpoints=1)
    axarr[2].set_ylabel('Ratio matched:non-matched', **font)

    # Save plots into figure directory
    plt.tight_layout(pad=0.001)
    plt.savefig(fig_dir + '/SS_Thresholding_Values.png')
