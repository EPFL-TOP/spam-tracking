import h5py
import tifffile
import os
import numpy as np

# Directory containing the TIFF files
tiff_dir = '/mnt/h/PROJECTS-03/clement/Clock_end/20240418_202031_Experiment/Position 1_Settings 1/'
tiff_dir = '/mnt/h/PROJECTS-03/Feyza/240925-NcadGFPxH2Bch-HIGHRES/20240925_151619_20240925_NcadxH2B_05z_timelapse/Position 4_Settings 1'
out_dir = '/mnt/d/Clement/'
# HDF5 file to create
hdf5_path = os.path.join(out_dir,'output_feyza_5tp.h5')

# Define the parameters of your dataset
num_timepoints = 5  # Number of time points
starting_point = 4
num_channels = 2     # Number of channels

# Initialize an HDF5 file with a 4D dataset
with h5py.File(hdf5_path, 'w') as hdf_file:
    # Assuming all TIFF files have the same shape
    sample_tiff = tifffile.imread(os.path.join(tiff_dir, 't0001_Channel 1.tif'))
    z, y, x = sample_tiff.shape  # 3D shape of each TIFF

    # Create a 5D dataset for (time, channel, z, y, x)
    dset = hdf_file.create_dataset(
        "image_data",
        shape=(num_timepoints, num_channels, 180, 400, 400),
        #shape=(num_timepoints, z, y, x),
        #shape=(num_timepoints, z, y, x),
        dtype=sample_tiff.dtype
    )
    #z:43->223
    #y

    # Loop over each time point and channel to read and store the data
    for t in range(num_timepoints):
        print('timepoint ',t+1+starting_point)
        filename = os.path.join(tiff_dir, f't{t+1+starting_point:04}_Channel {2}.tif')
#        data = tifffile.imread(filename)
#        data = data[50:230,980:1380,840:1240]
#        dset[t] = data 
        for ch in range(num_channels):
#            # Construct the filename pattern for each time point and channel
            print('  channel: ',ch+1)
            filename = os.path.join(tiff_dir, f't{t+1+starting_point:04}_Channel {ch+1}.tif')
            data = tifffile.imread(filename)
            data = data[50:230,980:1380,840:1240]
            print('shape ',data.shape)
            dset[t, ch] = data  # Insert data into HDF5

print("HDF5 dataset created successfully.")
