# ------------------------------------------------------------
# Real-Time Style Transfer Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Logan Engstrom
# Email: sbkim0407@gmail.com
# ------------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict

from style_transfer import Transfer
import utils as utils

def style_transform(image, checkpoint_dir):

    img_shape = image.shape
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), tf.Session(config=soft_config) as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=[None, *img_shape], name='img_placeholder')

        model = Transfer()
        pred = model(img_placeholder)

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('No checkpoint found...')
        else:
            saver.restore(sess, checkpoint_dir)

        img = np.asarray([image]).astype(np.float32)
        _pred = sess.run(pred, feed_dict={img_placeholder: img})
        return _pred[0]

if __name__ == '__main__':
    image = utils.imread("test_faces/mouth_open.jpg")
    for model_name in os.listdir("checkpoints"):
        if model_name == ".DS_Store":
            continue
        print("checkpoints/{}".format(model_name))
        trans_res = style_transform(image, "checkpoints/{}".format(model_name))
        utils.imsave("results/res{}.jpg".format(model_name), trans_res)
