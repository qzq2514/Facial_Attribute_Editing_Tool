import os
import sys
sys.path.append("..")
from scipy import misc

import numpy as np
import tensorflow as tf
from Utils import check_dir, get_file_name
from inference import get_sesssion, inference_core

# Bangs: boundaries/for_latent_w/5_Bangs_w/Bangs_boundary.npy
# Receding_Hairline: boundaries/for_latent_w/28_Receding_Hairline_w/Receding_Hairline_boundary.npy
# Eyeglasses: boundaries/for_latent_w/15_Eyeglasses_w/Eyeglasses_boundary.npy
# Big Nose: boundaries/for_latent_w/7_Big_Nose_w/Big_Nose_boundary.npy
# Wave Hair: boundaries/for_latent_w/33_Wavy_Hair_w/Wavy_Hair_boundary.npy
# Brown Hair: boundaries/for_latent_w/11_Brown_Hair_w/Brown_Hair_boundary.npy
# Mouth Open: boundaries/for_latent_w/21_Mouth_Slightly_Open_w/Mouth_Slightly_Open_boundary.npy
# Male(Gender): boundaries/for_latent_w/20_Male_w/Male_boundary.npy
# Young(age): boundaries/for_latent_w/39_Young_w/Young_boundary.npy
# Pale_skin: boundaries/for_latent_w/26_Pale_Skin_w/Pale_Skin_boundary.npy

# save_dir = check_dir("golbal_edit_res/")
edit_factor = 1.0   #主要编辑方向的编辑步长
main_boundary_path = "models/boundaries/for_latent_w/21_Mouth_Slightly_Open_w/Mouth_Slightly_Open_boundary.npy"
#限制条件即对应的限制程度
cond_boundary_dict_org = {"models/boundaries/for_latent_w/5_Bangs_w/Bangs_boundary.npy": 1.0}

latent_path = "data/2020_09_21_13_06_52.npy"

sess, latent_place, image_place = get_sesssion()

'''
edit_boundary_info格式:
{'main_boundary': {
                   'name': 'Mouth_Slightly_Open', 
                   'factor': 1.0, 
                   'main_boundary':np.array()},
 'cond_boundaries': {
                     'Bangs':{
                              'cond_factor': 1.0, 
                              'cond_boundary': np.array()},
                     'Eyeglasses':{
                              'cond_factor': 2.0, 
                              'cond_boundary': np.array()},
                      ...}
}
'''
#上面edit_boundary_info格式为InterFaceGAN的一组编辑信息，包括一个主要编辑属性和多个约束属性
#根据一组编辑信息获得Z/W空间内移动后的向量
def get_InterFaceGAN_move_direction(latent, edit_boundary_info):
    edit_info = ""
    main_boundary_info = edit_boundary_info["main_boundary"]
    main_name = main_boundary_info["name"]
    main_factor = main_boundary_info["factor"]
    edit_info += "_{}_{:.2f}".format(main_name, main_factor)
    main_boundary = main_boundary_info["main_boundary"].copy()   #注意这里要copy一下,因为python是引用传递,不然可能会修改原boundary

    move_boundary = main_boundary
    for cond_name, cond_boundary_info in edit_boundary_info["cond_boundaries"].items():
        cond_factor = cond_boundary_info["cond_factor"]
        cond_boundary = cond_boundary_info["cond_boundary"]
        edit_info += "_{}_{:.2f}".format(cond_name, cond_factor)
        # 条件操作的核心: boundary1-factor*(np.matmul(boundary1, boundary2.T)) * boundary2
        move_boundary -= cond_factor * (np.matmul(main_boundary, cond_boundary.T)) * cond_boundary

    # 要注意:main_factor是乘在最终的move_boundary上面的
    moved_latent = latent + main_factor * move_boundary
    return moved_latent, edit_info

def InterFaceGAN_edit(latent, edit_boundary_info):
    moved_latent, edit_info = get_InterFaceGAN_move_direction(latent, edit_boundary_info)

    image_out = inference_core(moved_latent, None)
    return image_out,  edit_info #if main_factor!=0 else "_noEdit"

if __name__ == '__main__':
    latent = np.load(latent_path)
    main_name = get_file_name(main_boundary_path, keep_ext=False).replace("_boundary", "")
    edit_boundary_info = {"main_boundary":{"name": main_name, "factor": edit_factor,
                                           "main_boundary": np.load(main_boundary_path)}}
    cond_boundaries_info = {}
    for boundary_path, cond_factor in cond_boundary_dict_org.items():
        boundary_name = get_file_name(boundary_path,keep_ext=False).replace("_boundary","")
        boundary_info = {"cond_factor":cond_factor, "cond_boundary":np.load(boundary_path)}
        cond_boundaries_info[boundary_name]=boundary_info
    edit_boundary_info["cond_boundaries"] = cond_boundaries_info

    image, edit_info = InterFaceGAN_edit(latent, edit_boundary_info)
    misc.imsave("InterFaceGAN_edit_res.jpg", image)
