import time
from scipy import misc
import numpy as np
import tensorflow as tf

model_path = "StyleGAN_edit/models/ffhq.pb"

stylegan2_sess = tf.Session()
with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    stylegan2_sess.graph.as_default()
    tf.import_graph_def(graph_def, name="")

stylegan2_sess.run(tf.global_variables_initializer())
z_latent_place = stylegan2_sess.graph.get_tensor_by_name('Gs/G_mapping/new_latent:0')                # Z空间[1,512]
w_latent_place = stylegan2_sess.graph.get_tensor_by_name('Gs/G_mapping/Dense7/mul_2:0')              # W空间[1,512]
wPlus_latent_place = stylegan2_sess.graph.get_tensor_by_name("Gs/G_mapping/Broadcast/Tile:0")        # W+空间[1,18,512](W空间简单复制18份)
# half_wPlus_latent_place = stylegan2_sess.graph.get_tensor_by_name("Gs/G_synthesis/dlatents_in:0")    # W+空间每个元素减半[1,18,512](暂时可不用)
image_place = stylegan2_sess.graph.get_tensor_by_name('Gs/G_synthesis/output:0')

'''
#z空间(随机高斯分布[1,512]): 'Gs/G_mapping/new_latent:0'
#w空间(映射网络的输出结果[1,512]): 'Gs/G_mapping/Dense7/mul_2:0'
#W+空间([在w的第二维度增加并broadcast得到[1,18,512]]) 'Gs/G_mapping/Broadcast/Tile:0'
latent_z_place = sess.graph.get_tensor_by_name('Gs/G_mapping/new_latent:0')
latent_w_place = sess.graph.get_tensor_by_name('Gs/G_mapping/Dense7/mul_2:0')
latent_w_plus_place = sess.graph.get_tensor_by_name('Gs/G_mapping/Broadcast/Tile:0')
'''

def get_sesssion():
    return stylegan2_sess,w_latent_place,image_place

# 从w空间开始
# 初始化: 4.745085954666138 s
# 预热后: 3.3608760833740234 s
# 参数中的input_feed只是ATMGAN中需要用的,还没牵扯到latent
def inference_core(latent, input_feed=None, latent_type="W", out_node="Image"):
    if input_feed is None:
        input_feed={}

    if latent_type=="Z":
        input_feed[z_latent_place] = latent
    elif latent_type=="W":
        input_feed[w_latent_place] = latent
    else:
        input_feed[wPlus_latent_place] = latent

    out_place = image_place
    if out_node=="Z":
        out_place = z_latent_place
    elif out_node=="W":
        out_place =  w_latent_place
    elif out_node=="W+":
        out_place = wPlus_latent_place

    # start_time = time.time()
    inference_out = stylegan2_sess.run(out_place, feed_dict=input_feed)
    # end_time = time.time()
    # print("inference_core time:", end_time - start_time)

    if out_node!="Image":
        return inference_out

    with open("inference_core.jpg", "wb") as fw:
        fw.write(inference_out)
    inference_out = misc.imread("inference_core.jpg")
    return inference_out

# 从z空间开始
# 初始化: 14.806763887405396 s
# 预热后: 3.0977349281311035 s
def random_gen():
    start_time = time.time()
    latent, image = stylegan2_sess.run([w_latent_place,image_place])
    end_time = time.time()
    print("sess_run time:", end_time - start_time)
    with open("random_gen.jpg", "wb") as fw:
        fw.write(image)
    image_stylized = misc.imread("random_gen.jpg")
    return latent, image_stylized

if __name__ == '__main__':
    latent_path = "data/Leonardo_W+.npy"
    latent = np.load(latent_path)
    # print(latent[:,:10])
    w_plus = inference_core(latent[np.newaxis,:], None, latent_type="W+")
    # print("-----------")
    # print(w_plus.shape, np.sum(w_plus[0, 1, :]))
    # print(w_plus)

    # for _ in range(10):
    #     # random_gen()
    #     inference_core(latent, None, "W", "Image")
