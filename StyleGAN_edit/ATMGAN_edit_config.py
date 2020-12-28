import numpy as np

edit_attr_info = {}

#卷发性质:
curly_hair={}
curly_hair["name"] = "Curly_Hair"
curly_hair["reslotion"] = 8
curly_hair["fmap"] = np.array([26,92])
curly_hair["range"] = [-30,35]   #[-15,20]
curly_hair["location"] = np.array([2,3,4,5,
                                   9,10,11,12,13,14,
                                   17,18,21,22,
                                   25,30,# 25,30,31,
                                   # 32,33,38,39,
                                   # 40,41,46,47,
                                   # 48,49,54,55,
                                   # 56,57,58,62,63
                                   ])
edit_attr_info["Curly_Hair"] = curly_hair

#嘴巴张开
mouth_open = {}
mouth_open["name"] = "Mouth_Open"
mouth_open["reslotion"] = 16
mouth_open["fmap"] = np.array([317,381])
mouth_open["range"] = [-15,10]    #[-15,10]
mouth_open["location"] = np.array([182,183,184,185,
                                   198,199,200,201])

edit_attr_info["Mouth_Open"] = mouth_open

#鼻子变宽
nose_widen = {}
nose_widen["name"] = "Wide_Nose"
nose_widen["reslotion"] = 32
nose_widen["fmap"] = np.array([155,253])
nose_widen["range"] = [-10,10]
nose_widen["location"] = np.array([463,464,
                                   495,496,
                                   527,528,
                                   559,560,
                                   590,591,592,593,
                                   621,622,623,624,625,626,
                                   653,654,655,656,657,658,
                                   686,687,688,689])
edit_attr_info["Wide_Nose"] = nose_widen

#眼睛变大变黑
eyes_bigger = {}
eyes_bigger["name"] = "Big_Eyes"
eyes_bigger["reslotion"] = 16
eyes_bigger["fmap"] = np.array([253,290])
eyes_bigger["range"] = [-20,40]
eyes_bigger["location"] = np.array([117,118,
                                    121,122])
edit_attr_info["Big_Eyes"] = eyes_bigger

#戴眼镜
glasses_wear = {}
glasses_wear["name"] = "Eyeglasses"
glasses_wear["reslotion"] = 16
glasses_wear["fmap"] = np.array([317,81,6])   #np.array([481,501])
glasses_wear["range"] = [-20,20]
glasses_wear["location"] = np.array([115,116,117,118,119,120,121,122,123,124,
                                     132,133,134,135,136,  #132,133,134
                                     137,138,139])

edit_attr_info["Eyeglasses"] = glasses_wear

# 闭眼
eyes_close = {}
eyes_close["name"] = "Closing_Eyes"
# eyes_close["reslotion"] = 16
# eyes_close["fmap"] = np.array([6, 78])
# eyes_close["range"] = [-200, 200]
# eyes_close["location"] = np.array([ 117, 118, 121, 122])

eyes_close["reslotion"] = 32
eyes_close["fmap"] = np.array([2, 257])
eyes_close["range"] = [-80, -10]
eyes_close["location"] = np.array([    # 427, 428,           435, 436,
                                   458, 459, 460, 461, 466, 467, 468, 469,
                                   490, 491, 492, 493, 498, 499, 500, 501,
                                   522, 523, 524, 525, 530, 531, 532, 533])

edit_attr_info["Closing_Eyes"] = eyes_close
