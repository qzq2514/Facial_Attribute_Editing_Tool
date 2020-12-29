import os
import sys
from scipy import misc
sys.path.append("..")
from Utils import check_dir, get_file_name
from ATMGAN_edit_config import edit_attr_info
from inference import get_sesssion,inference_core
import numpy as np

# save_dir = check_dir("local_edit_res/")
latent_path = "data/2020_09_21_13_06_52.npy"
factor_dict = {"Curly_Hair": 0, "Mouth_Open": 0, "Wide_Nose": 0,
                "Big_Eyes": 0, "Eyeglasses": -15.5, "Closing_Eyes": 0}

sess, latent_places, image_place = get_sesssion()
FmapInput_places = {}
resolutions = [8, 16, 32, 64, 128, 256, 512]
for reso in resolutions:
    FmapInput_places[reso] = sess.graph.get_tensor_by_name('Gs/G_synthesis/{}x{}/FmapInput:0'.format(reso,reso))

def get_fmap_input(edit_attr_info, factor_dict):
    fmap_input={}
    fmap_input_flat = {}
    for reso,tensor in FmapInput_places.items():
        shape = tensor.shape[1:]
        fmap_input[reso] = np.zeros(shape=shape)
        fmap_input_flat[reso] = np.zeros(shape=(shape[0]*shape[1],shape[2]))

    for attr_name, factor in factor_dict.items():
        attr_info = edit_attr_info[attr_name]
        fmap = attr_info["fmap"]
        reso = attr_info["resolution"]
        location = attr_info["location"]

        for fmap_index in fmap:
            fmap_input_flat[reso][location,fmap_index]+=factor

    for reso,flat_fmap in fmap_input_flat.items():
        fmap_input[reso] = np.reshape(flat_fmap, fmap_input[reso].shape)[np.newaxis,:]
    return fmap_input

#仅仅获得与ATMGAN有关的编辑信息，与latent无关
def get_ATMGAN_feed(edit_attr_info, factor_dict):

    fmap_input = get_fmap_input(edit_attr_info, factor_dict)
    input_feed = {}
    if fmap_input is not None:
        for key, input_data in fmap_input.items():
            input_feed[FmapInput_places[key]] = input_data
    return input_feed

# edit_attr_info是一个字典格式，具体形式见"ATMGAN_edit_config.py"
def ATMGAN_edit(latent, edit_attr_info, factor_dict, latent_type="W"):

    edit_info = ""
    # 字符串拼接所有属性的编辑因子,为了后续编辑结果图像的命名
    for attr_name, factor in factor_dict.items():
        if factor!=0:
            edit_info += "_{}_{:.2f}".format(attr_name, factor)

    ATMGAN_feed = get_ATMGAN_feed(edit_attr_info, factor_dict)
    image = inference_core(latent, ATMGAN_feed, latent_type)

    return image, edit_info

if __name__ == '__main__':
    edit_attr_info = edit_attr_info
    latent = np.load(latent_path)
    image, edit_info = ATMGAN_edit(latent, edit_attr_info, factor_dict)

    misc.imsave("ATMGAN_edit_res.jpg", image)
