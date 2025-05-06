import h5py
import os
import numpy as np
import time

np.random.seed(0)

data_folder = "./data/taxi"
stem = "yellow_tripdata_"
years = ["2009", "2010", "2011", "2012", "2013", "2014", "2015"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

total_rows = 0
n_samples = 334_000_000

X_list = []
Y_list = []

# Open .h5py files for each month
for year in years:
    for month in months:
        filename = stem + year + "-" + month + ".h5py"
        print(f"Processing {filename}")
        with h5py.File(os.path.join(data_folder, filename), "r") as f:
            data = f["X"][()]
            labels = f["Y"][()]

            X_list.append(data)
            Y_list.append(labels)
            total_rows += data.shape[0]

dim = X_list[0].shape[1]
max_chunk_size = 2 * 2**20  # 2MB
chunk_x = int(max_chunk_size / dim / 8)
chunk_y = chunk_x

# Turn the .h5py files into one big .h5py file
with h5py.File(
    os.path.join(data_folder, "full_taxi_data.h5py"), "w", libver="latest"
) as f:
    Xdset = f.create_dataset(
        "X",
        (total_rows, dim),
        dtype="float64",
        compression="gzip",
        chunks=(chunk_x, dim),
    )
    Ydset = f.create_dataset("Y", (total_rows, 1), dtype="int32")
    current_i = 0
    for X, Y in zip(X_list, Y_list):
        t_s = time.time()
        X = np.ascontiguousarray(X)
        Y = Y.reshape((-1, 1))
        Xdset.write_direct(X, dest_sel=np.s_[current_i : current_i + X.shape[0], :])
        Ydset.write_direct(Y, dest_sel=np.s_[current_i : current_i + Y.shape[0], :])
        current_i += X.shape[0]
        print("i: %d/%d in %.2fs" % (current_i, total_rows, time.time() - t_s))

# Open the big .h5py file for subsampling
with h5py.File(os.path.join(data_folder, "full_taxi_data.h5py"), "r") as f:
    print(f.keys())

    print(f["X"])
    print(f["Y"])

    X_full = f["X"][()]
    y_full = f["Y"][()]

# Sample the data
sample_indices = np.random.choice(X_full.shape[0], n_samples, replace=False)
X_small = X_full[sample_indices]
y_small = y_full[sample_indices]

# Save the sampled data
with h5py.File(os.path.join(data_folder, "data.h5py"), "w", libver="latest") as f:
    Xdset = f.create_dataset(
        "X",
        (n_samples, dim),
        dtype="float64",
        compression="gzip",
        chunks=(chunk_x, dim),
    )
    Ydset = f.create_dataset("Y", (n_samples, 1), dtype="int32")
    current_i = 0
    t_s = time.time()
    X_small = np.ascontiguousarray(X_small)
    y_small = y_small.reshape((-1, 1))
    Xdset.write_direct(
        X_small, dest_sel=np.s_[current_i : current_i + X_small.shape[0], :]
    )
    Ydset.write_direct(
        y_small, dest_sel=np.s_[current_i : current_i + y_small.shape[0], :]
    )
    current_i += X_small.shape[0]
    print("i: %d/%d in %.2fs" % (current_i, n_samples, time.time() - t_s))
