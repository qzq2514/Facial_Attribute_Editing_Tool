from scipy import misc
import numpy as np
from inference import inference_core

def get_mixing_dlatent(latent1, latent2, K=8):
    # 通过w空间得到
    # 这里w+空间的delatent1不是latent1简单复制18份得到的(可能找的张量名不太对,但是不影响编辑)
    delatent1 = inference_core(latent1, None, latent_type="W", out_node="W+")
    delatent2 = inference_core(latent2, None, latent_type="W", out_node="W+")

    # 0,2, 4, 6, 8, 10, 12, 14,  16
    # [4,8,16,32,64,128,256,512,1024]
    # Coarse styles: 0:4  high-level aspects such as pose, general hair style, face shape, and eyeglasses
    # Middle styles: 4:8  hair style, eyes open/closed
    # Fine styles: 8:-1   color scheme and microstructure
    delatent1[:, K:-1, :] = delatent2[:, K:-1, :]
    return delatent1

def Making_up(latent1, latent2, K=8):

    mixing_dlatent = get_mixing_dlatent(latent1, latent2, K)
    makeUp_res = inference_core(mixing_dlatent, None, latent_type="W+")
    return makeUp_res

if __name__ == '__main__':
    latent_con = np.load("data/2020_09_21_13_06_52.npy")
    latent_ref = np.load("MakeUp/makeup_latents/2020_10_16_17_18_51.npy")
    makeUp_res = Making_up(latent_con, latent_ref)
    misc.imsave("makeUp_res.jpg",makeUp_res)
