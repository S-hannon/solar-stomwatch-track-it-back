import ssw_clustering_cor1_cor2
import ssw_thresholding
import ssw_clustering_matching_euvi
import ssw_comparisons
import ssw_analysis

# Data directories to use:
# Directory which should contain all raw datafiles
# Final catalogue CSV files will be stored here
data_dir =
# For storing figures
fig_dir =
# For storing the thresholding data
th_dir = 
# For storing the clustering plots
c_fig_dir = 

###############################################################################
# Investigate the thresholds for the COR1 and COR2 event clustering

# COR2
# Specify thresholds to investiagate for clustering code
cor2_runs = [[2.0, 1.0, 15.0, 1.0, 15.0], [2.0, 1.0, 15.0, 1.0, 05.0],
             [2.0, 1.0, 15.0, 1.0, 10.0], [2.0, 1.0, 10.0, 1.0, 10.0],
             [2.0, 1.0, 10.0, 1.0, 15.0], [2.0, 1.0, 10.0, 1.0, 12.0],
             [2.0, 1.0, 12.0, 1.0, 12.0], [2.0, 1.0, 12.0, 1.0, 10.0],
             [2.0, 1.0, 15.0, 2.0, 12.0], [2.0, 1.0, 15.0, 2.0, 15.0]]
# Run COR2 clustering code with thresholds specified above
for i in range(0, len(cor2_runs)):
    ssw_clustering_cor1_cor2.run_cor2(cor2_runs[i][0], cor2_runs[i][1],
                                      cor2_runs[i][2], cor2_runs[i][3], 
                                      cor2_runs[i][4], th_dir, c_fig_dir)
# Create csv file with the results
cor2_results = ssw_thresholding.create_cor2_summary(cor2_runs, data_dir, th_dir)
# Plot the results
ssw_thresholding.cor2_combined_threshold_plot(cor2_results, fig_dir)

# COR1
# Specify thresholds to investiagate for clustering code
cor1_runs = [[2.0, 1.0, 15.0, 1.0, 05.0], [2.0, 1.0, 15.0, 1.0, 07.0],
             [2.0, 1.0, 15.0, 1.0, 10.0], [2.0, 1.0, 15.0, 1.0, 12.0],
             [2.0, 1.0, 15.0, 1.0, 15.0]]
# Run COR1 clustering code with thresholds specified above
for i in range(0, len(cor1_runs)):
    ssw_clustering_cor1_cor2.run_cor1(cor1_runs[i][0], cor1_runs[i][1],
                                      cor1_runs[i][2], cor1_runs[i][3],
                                      cor1_runs[i][4], th_dir, c_fig_dir)
# Create csv file with the results
cor1_results = ssw_thresholding.create_cor1_summary(cor1_runs, th_dir)

###############################################################################
# Get final event lists

# Run the clustering code on the COR1 and COR2 data using the final thresholds
ssw_clustering_cor1_cor2.run_cor2(2.0, 1.0, 15.0, 1.0, 12.0, data_dir, fig_dir)
ssw_clustering_cor1_cor2.run_cor1(2.0, 1.0, 15.0, 1.0, 07.0, data_dir, fig_dir)
# Now match events between the COR1 and COR2 catalogues
ssw_clustering_matching_euvi.run_matching(data_dir)
# And add the relevant EUVI data to complete the catalogue
ssw_clustering_matching_euvi.run_euvi(data_dir)

###############################################################################
# Analyse

# Import all the relevant data
# Get the SSW catalogue data from the csv files
c1a, c1b, c2a, c2b, euvia, euvib = ssw_analysis.get_ssw_data(data_dir)
# Get lists of matched events in the SEEDS and CACTus catalogues
seeds_a, seeds_b, cactus_a, cactus_b = ssw_comparisons.get_matched_lists(c2a, c2b, data_dir)
# Get deflections data - locations of HCS from MAS model
da, db, lda, ldb = ssw_analysis.get_deflections_data(c1a, c1b, c2a, c2b,
                                                     data_dir)

# Create the plots to analyse the data
ssw_analysis.timing_uncertainties_plot(c2a, c2b, fig_dir)
ssw_analysis.latitude_plot(c1a, c1b, c2a, c2b, fig_dir)
ssw_analysis.width_plot(c1a, c1b, c2a, c2b, fig_dir)
ssw_analysis.speed_plot(c1a, c1b, c2a, c2b, fig_dir)
ssw_analysis.time_dist_all_plots(euvia, euvib, c1a, c1b, c2a, c2b, fig_dir)
ssw_analysis.seeds_cactus_comparison_plot(c2a, c2b, seeds_a, seeds_b, cactus_a,
                                          cactus_b, fig_dir)
ssw_analysis.deflections_plot(da, db, lda, ldb, fig_dir)
