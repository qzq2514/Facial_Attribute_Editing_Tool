import os
import numpy as np

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def latent_post_process(latent):
    if latent.shape[0]==1:
        return latent
    return np.expand_dims(latent,axis=0)

def get_file_name(path, keep_ext = True):
    name = os.path.basename(path)
    return name if keep_ext else os.path.splitext(name)[0]


if __name__ == '__main__':
    print(get_file_name("../data/2020_09_21_13_06_52.npy", True))