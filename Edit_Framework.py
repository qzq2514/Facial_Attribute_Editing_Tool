import os
import time
import numpy as np
from StyleGAN_edit.ATMGAN_edit_config import edit_attr_info as ATMGAN_org_info
from StyleGAN_edit.ATMGAN import get_ATMGAN_feed
from NST.transfer_inference import style_transform
from StyleGAN_edit.inference import random_gen,inference_core
from StyleGAN_edit.InterFaceGAN import get_InterFaceGAN_move_direction
from StyleGAN_edit.MakeUp import get_mixing_dlatent
from Utils import get_file_name
from scipy import misc
from tkinter import ttk   #用ttk的按钮，不会出现文字不显示的问题
from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk

image_height,image_width = 880, 880
small_image_shape = (256, 256)
org_path = "StyleGAN_edit/data/2020_09_21_13_06_52.jpg"
latent_path = "StyleGAN_edit/data/2020_09_21_13_06_52.npy"

class Edit_Framework():
    def __init__(self,master):
        self.tkObj = master
        self.tkObj.title("人脸属性编辑框架")
        self.tkObj['width'] = 1880
        self.tkObj['height'] = 1100
        # self.tkObj.resizable(width=False, height=False)

        #初始的一些配置
        self.image_exp_NST = self.cur_image = self.org_image = misc.imread(org_path)
        self.cur_latent = self.org_lantent = np.load(latent_path)
        self.ATMGAN_edit_info = {}   #存放ATMGAN编辑需要的固定的属性配置
        self.ATMGAN_factors = {}     #存放ATMGAN编辑每个属性属性的动态编辑因子

        left_frame = Frame(self.tkObj, bg="#CCFFFF")
        left_frame.pack(side=LEFT)
        # 图片上方的加载/生成/更新等按钮
        head_frame = Frame(left_frame)
        self.create_head(head_frame, color="#CCFFFF")
        # 原始图片显示区域
        image_frame = Frame(left_frame)
        image_frame.pack(side=TOP, expand=NO)
        self.imgPIL_org = ImageTk.PhotoImage(Image.fromarray(self.org_image).resize((image_width, image_height)))
        self.face_ImagePanel = self.create_ImagePanel(image_frame,self.imgPIL_org)

        # 中间使用ATMGAN和InterFaceGAN进行人脸属性编辑的功能区域
        mid_frame = Frame(self.tkObj)
        mid_frame.pack(side=LEFT)
        ATMGAN_frame = Frame(mid_frame, bg="#FFFFCC")
        ATMGAN_frame.pack(side=TOP, pady=10, ipady=20)
        self.create_ATMGAN(ATMGAN_frame, color="#FFFFCC")

        InterFaceGAN_frame = Frame(mid_frame, bg="#CCFFCC")
        InterFaceGAN_frame.pack(side=TOP, ipady=20)
        self.create_InterFaceGAN(InterFaceGAN_frame, color="#CCFFCC")

        # 最右侧进行风格迁移和妆容迁移
        right_frame = Frame(self.tkObj)
        right_frame.pack(side=LEFT)
        NST_frame = Frame(right_frame, bg="#66CCFF")
        NST_frame.pack(side=TOP, ipady=10)
        self.create_NST(NST_frame, color="#66CCFF")
        MakeUp_frame = Frame(right_frame, bg="#FF99CC")
        MakeUp_frame.pack(side=TOP, pady=10, ipady=10)
        self.create_MakeUp(MakeUp_frame, color="#FF99CC")

    def update(self):
        # 依次整合InterFaceGAN(Z/W空间)、Style_Mixing(W+空间)和ATMGAN(激活张量空间)的编辑信息,然后一次性进行编辑

        # 1.在Z/W空间获得根据InterFaceGAN移动后的隐向量
        InterFaceGAN_edit_info = self.get_InterFaceGAN_edit_info()
        # 让使用者决定是从原始开始编辑还是从当前InterFaceGAN的编辑结果进一步编辑,改动的话,这里只需要将self.org_lantent变为self.cur_latent就行
        moved_latent, edit_info = get_InterFaceGAN_move_direction(self.org_lantent, InterFaceGAN_edit_info)
        # 单独使用InterFaceGAN进行编辑(inference时候不考虑Style_Mixing和ATMGAN)
        # # 这里暂时让InterFaceGAN都是最原始的latent上进行编辑,后续可能考虑添加一个选项,
        # # 让使用者决定是从原始开始编辑还是从当前InterFaceGAN的编辑结果进一步编辑,改动的话,这里只需要将self.org_lantent变为self.cur_latent就行
        # edit_res, latent, edit_info = InterFaceGAN_edit(self.org_lantent, InterFaceGAN_edit_info)
        self.cur_latent = moved_latent  #更新W空间的latent

        # 3.在激活张量空间获取使用ATMGAN进行编辑所需要的信息
        self.ATMGAN_feed = get_ATMGAN_feed(self.ATMGAN_edit_info,self.ATMGAN_factors)

        start_time = time.time()
        edit_res = inference_core(self.cur_latent, self.ATMGAN_feed)
        end_time = time.time()

        #最终的编辑结果收到InterFaceGAN、Style_Mixing和AMTGAN的综合影响
        self.cur_image = edit_res
        self.update_imagePanel(end_time - start_time)

    def start_makeUp(self):
        # 进行妆容迁移前，要保留之前InterFace和ATMGAN编辑的结果
        # InterFaceGAN的编辑必要信息就保存在self.cur_latent中
        # ATMGAN的编辑必要信息就保存在self.ATMGAN_feed

        # 开始在W+空间进行Style_Mixing得到混合后的隐向量
        makeUp_name = self.makeUp_comb.get()
        makeUp_latent = np.load("StyleGAN_edit/MakeUp/makeup_latents/{}.npy".format(makeUp_name))

        mixing_dlatent = get_mixing_dlatent(self.cur_latent, makeUp_latent)

        start_time = time.time()
        edit_res = inference_core(mixing_dlatent, self.ATMGAN_feed, latent_type="W+")
        end_time = time.time()
        self.cur_image = edit_res
        self.update_imagePanel(end_time - start_time)

    def get_ATMGAN_attr_names(self):
        attr_names=[]
        for attr_name, info in ATMGAN_org_info.items():
            attr_names.append(attr_name)
        return attr_names

    def create_ATMGAN(self, frame, color):
        #ttk.Style()命名必须以Txxxx结尾,其中xxxx是button/Label之类，当然还可以在之前加自定义的参数，如"my2."
        ttk.Style().configure('AMTGAN.TButton', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('AMTGAN.TLabel', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('AMTGAN.TNotebook', font=('Helvetica', 20), foreground="black", background=color)

        ATMGAN_lab = ttk.Label(frame, text="ATMGAN", style="AMTGAN.TLabel")
        ATMGAN_lab.pack(side=TOP,pady=10)

        def change_ATMGAN_attr_show(event):
            attr_name = ATMGAN_attrs_comb.get()
            attr_info = ATMGAN_org_info[attr_name]
            ATMGAN_attr_intro_lab.config(text="属性名称:{0}\n张量分辨率:{1}x{1}\n特征图:[{2}]".
                           format(attr_name, attr_info["reslotion"], ",".join([str(x) for x in attr_info["fmap"]])))

        attr_names = self.get_ATMGAN_attr_names()
        ATMGAN_attrs_comb = ttk.Combobox(frame)
        ATMGAN_attrs_comb['value'] = attr_names
        ATMGAN_attrs_comb.pack(side=TOP, pady=10)
        ATMGAN_attrs_comb.current(2)
        ATMGAN_attrs_comb.bind("<<ComboboxSelected>>", change_ATMGAN_attr_show)

        ATMGAN_attr_intro_lab = Label(frame, text="属性名称:{0}\n张量分辨率:{1}x{1}\n特征图:[{2}]".
                           format(attr_names[0],ATMGAN_org_info[attr_names[0]]["reslotion"],",".join([str(x) for x in ATMGAN_org_info[attr_names[0]]["fmap"]])),bg=color)
        ATMGAN_attr_intro_lab.pack(side=TOP)
        # 滑动条:orient，控制滑块的方位，HORIZONTAL（水平），VERTICAL（垂直）,通过resolution选项可以控制分辨率（步长），通过tickinterval选项控制刻度
        ATMGAN_scale = Scale(frame, from_=-100, to=100, orient=HORIZONTAL, resolution=0.1, length=400, bg=color)
        ATMGAN_scale.pack();ATMGAN_scale.set(0)

        temp_frame1 = Frame(frame, bg=color)
        temp_frame1.pack(side=TOP, pady=10)


        def change_comb(event):
            selection = self.ATMGAN_attrs_listbox.curselection()
            attr_name = self.ATMGAN_attrs_listbox.get(selection)
            attr_name = attr_name[:attr_name.find(":")]
            for ind, name in enumerate(attr_names):
                if name.startswith(attr_name):
                    ATMGAN_attrs_comb.current(ind)

        self.ATMGAN_attrs_listbox = Listbox(temp_frame1, width=30, height=8)
        self.ATMGAN_attrs_listbox.pack(side=LEFT, padx=10)
        self.ATMGAN_attrs_listbox.bind("<Double-Button-1>", change_comb)


        #添加ATMGAN待编辑的属性
        def add_ATMGAN_edit_attr():
            attr_name = ATMGAN_attrs_comb.get()
            attr_factor = ATMGAN_scale.get()
            attr_info = ATMGAN_org_info[attr_name]
            # 1.listbox中添加相关信息
            attr_nums = self.ATMGAN_attrs_listbox.size()
            insert_ind = END
            for ind in range(attr_nums):
                if self.ATMGAN_attrs_listbox.get(ind).startswith(attr_name):
                    self.ATMGAN_attrs_listbox.delete(ind)
                    insert_ind = ind
                    break
            self.ATMGAN_attrs_listbox.insert(insert_ind, "{0}: {1}--->{2}x{2}   [{3}]".format(attr_name, attr_factor, attr_info["reslotion"],
                                                                                       ",".join([str(x) for x in attr_info["fmap"]])))
            # 2.ATMGAN_edit_info添加相关信息
            self.ATMGAN_edit_info[attr_name] = attr_info
            # 3.ATMGAN_factors添加相关信息
            self.ATMGAN_factors[attr_name] = attr_factor
        add_btn = ttk.Button(temp_frame1, style="AMTGAN.TButton", text="添加/更新", command=add_ATMGAN_edit_attr)
        add_btn.pack(side=TOP, pady=10)

        #移除ATMGAN待编辑的属性
        def remove_ATMGAN_edit_attr():
            #1.移除listbox中显示的
            selection = self.ATMGAN_attrs_listbox.curselection()
            if len(selection) == 0:
                return
            attr_name = self.ATMGAN_attrs_listbox.get(selection).split(":")[0]
            self.ATMGAN_attrs_listbox.delete(selection)
            #2.移除ATMGAN_edit_info中该属性信息
            self.ATMGAN_edit_info.pop(attr_name)
            #3.移除ATMGAN_factors中该属性的编辑因子
            self.ATMGAN_factors.pop(attr_name)
        remove_btn = ttk.Button(temp_frame1, style="AMTGAN.TButton", text="移除",command = remove_ATMGAN_edit_attr)
        remove_btn.pack(side=TOP, pady=10)

    def update_imagePanel(self, spend_time, update_from_NST=False):
        #防止风格迁移的效果不断累加,同时能够保证self.cur_image一直都是大窗口显示的人脸结果
        if not update_from_NST:
            self.image_exp_NST = self.cur_image
        self.imgPIL_org = ImageTk.PhotoImage(Image.fromarray(self.cur_image).resize((image_width, image_height)))
        self.face_ImagePanel.create_image(0, 0, image=self.imgPIL_org, anchor=NW)
        self.time_show_lab.config(text="用时:{:.2f} s".format(spend_time))

    def get_InterFaceGAN_edit_info(self):
        # 获得InterFaceGAN的编辑信息
        # 1.获取主编辑属性
        main_attr_factor = self.main_attr_scale.get()
        boundary_name = self.main_attr_comb.get()
        boundary = np.load(os.path.join("StyleGAN_edit/models/boundaries/for_latent_w", "{}_w/{}_boundary.npy".
                                        format(boundary_name, boundary_name[boundary_name.find("_") + 1:])))
        InterFaceGAN_edit_info = {"main_boundary": {'name': 'Mouth_Slightly_Open',
                                                    'factor': main_attr_factor,
                                                    'main_boundary': boundary},
                                  "cond_boundaries": {}}
        # 2.获取约束属性
        cond_nums = self.cond_attrs_listbox.size()
        for ind in range(cond_nums):
            edit_info = self.cond_attrs_listbox.get(ind)
            cond_boundary_name, cond_factor = edit_info.split(":")
            cond_boundary_name_no_num = cond_boundary_name[cond_boundary_name.find("_") + 1:]
            cond_boundary = np.load(os.path.join("StyleGAN_edit/models/boundaries/for_latent_w", "{}_w/{}_boundary.npy".
                                                 format(cond_boundary_name, cond_boundary_name_no_num)))
            # print(cond_boundary_name, cond_factor, cond_boundary_name_no_num)  # 5_Bangs 0.71 Bangs

            InterFaceGAN_edit_info["cond_boundaries"][cond_boundary_name_no_num] = {'cond_factor': float(cond_factor),
                                                                                    'cond_boundary': cond_boundary}
        return InterFaceGAN_edit_info

    def get_makeUp_names(self):
        makeUp_root = "StyleGAN_edit/MakeUp"
        style_names = []
        for image_name in os.listdir(os.path.join(makeUp_root, "makeup_pics")):
            style_names.append(get_file_name(image_name, keep_ext=False))
        return style_names

    def create_MakeUp(self, frame, color):

        ttk.Style().configure('MakeUp.TButton', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('MakeUp.TLabel', font=('Helvetica', 20), foreground="black", background=color)

        NST_lab = ttk.Label(frame, text="妆容迁移", style="MakeUp.TLabel")
        NST_lab.pack(side=TOP, pady=10)

        NST_lab = ttk.Label(frame, text="妆容引导图", style="MakeUp.TLabel")
        NST_lab.pack(side=TOP, pady=10)

        style_names = self.get_makeUp_names()
        self.makeUp_imagePIL = ImageTk.PhotoImage(
            Image.fromarray(misc.imread("StyleGAN_edit/MakeUp/makeup_pics/{}.jpg"
                            .format(style_names[0]))).resize(small_image_shape))

        self.makeUp_comb = ttk.Combobox(frame, width=200)
        def change_makeUp_show(event):
            self.makeup_imagePIL = ImageTk.PhotoImage(
                Image.fromarray(misc.imread("StyleGAN_edit/MakeUp/makeup_pics/{}.jpg".format(self.makeUp_comb.get()))).resize(small_image_shape))
            self.makeUp_imagePannel.create_image(0, 0, image=self.makeup_imagePIL, anchor=NW)

        self.makeUp_comb.bind("<<ComboboxSelected>>", change_makeUp_show)
        self.makeUp_comb['value'] = style_names

        self.makeUp_comb.pack(side=TOP, pady=10)
        self.makeUp_comb.current(0)

        temp_frame = Frame(frame, bg =color)
        temp_frame.pack(side=TOP)

        self.makeUp_imagePannel = self.create_ImagePanel(temp_frame, self.makeUp_imagePIL, side=LEFT, shape=small_image_shape)

        add_btn = ttk.Button(temp_frame, style="MakeUp.TButton", text="开始上妆", command = self.start_makeUp)
        add_btn.pack(side=LEFT, pady=10)

    def start_NST(self):
        style_name = self.styles_comb.get()
        start_time = time.time()
        #风格迁移永远都是对当前窗口下的最新人脸进行风格化
        tran_res = style_transform(self.image_exp_NST, "NST/checkpoints/{}".format(style_name))
        end_time = time.time()
        tran_res = np.clip(tran_res, 0, 255).astype(np.uint8) #Image.fromarray接收uint8格式的图片

        self.cur_image = tran_res
        self.update_imagePanel(end_time-start_time, True)

    def get_style_pics(self):
        NST_root = "NST"
        style_names = []
        for image_name in os.listdir(os.path.join(NST_root, "style_pics")):
            style_names.append(get_file_name(image_name, keep_ext=False))
        return style_names

    def create_NST(self, frame, color):

        ttk.Style().configure('NST.TButton', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('NST.TLabel', font=('Helvetica', 20), foreground="black", background=color)

        NST_lab = ttk.Label(frame, text="风格迁移", style="NST.TLabel")
        NST_lab.pack(side=TOP, pady=10)

        NST_lab = ttk.Label(frame, text="风格图:", style="NST.TLabel")
        NST_lab.pack(side=TOP, pady=10)

        style_names = self.get_style_pics()
        self.style_imagePIL = ImageTk.PhotoImage(
            Image.fromarray(misc.imread("NST/style_pics/{}.jpg".format(style_names[0]))).resize(small_image_shape))

        self.styles_comb = ttk.Combobox(frame,width=200)
        def chane_style_show(event):
            self.style_imagePIL = ImageTk.PhotoImage(
                Image.fromarray(misc.imread("NST/style_pics/{}.jpg".format(self.styles_comb.get()))).resize(small_image_shape))
            self.style_imagePanel.create_image(0, 0, image=self.style_imagePIL, anchor=NW)

        self.styles_comb.bind("<<ComboboxSelected>>", chane_style_show)
        self.styles_comb['value'] = style_names

        self.styles_comb.pack(side=TOP, pady=10)
        self.styles_comb.current(0)

        temp_frame = Frame(frame, bg= color)
        temp_frame.pack(side=TOP)

        self.style_imagePanel = self.create_ImagePanel(temp_frame, self.style_imagePIL, side=LEFT, shape=small_image_shape)

        add_btn = ttk.Button(temp_frame, style="NST.TButton", text="开始迁移", command= self.start_NST)
        add_btn.pack(side=LEFT, pady=10)

    def create_ImagePanel(self, frame, imagePIL, side=TOP, shape=(image_height,image_width)):

        ImgPanel = Canvas(frame, cursor='tcross')
        ImgPanel.pack(side=side)
        ImgPanel.config(height=shape[0], width=shape[1])
        ImgPanel.create_image(0, 0, image=imagePIL, anchor=NW)
        return ImgPanel

    def get_boundaries(self):
        dir = "StyleGAN_edit/models/boundaries/for_latent_w"
        boundary_names = []
        for boundary_name in os.listdir(dir):
            boundary_names.append(boundary_name[:boundary_name.rfind("_")])
        boundary_names.sort(key=lambda x:int(x[:x.find("_")]))
        return boundary_names

    def create_InterFaceGAN(self, frame, color):
        ttk.Style().configure('InterFaceGAN.TButton', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('InterFaceGAN.TLabel', font=('Helvetica', 20), foreground="black", background=color)

        InterFaceGAN_lab = ttk.Label(frame, text="InterFaceGAN", style="InterFaceGAN.TLabel")
        InterFaceGAN_lab.pack(side=TOP, pady=10)
        boundary_names = self.get_boundaries()

        # 主要编辑属性
        temp_frame1 = Frame(frame,bg=color)
        temp_frame1.pack(side=TOP)
        main_attr_lab = ttk.Label(temp_frame1, text="主编辑属性:", style="InterFaceGAN.TLabel")
        main_attr_lab.pack(side=LEFT, padx=10)

        self.main_attr_comb = ttk.Combobox(temp_frame1)
        self.main_attr_comb['value'] = boundary_names
        self.main_attr_comb.pack(side=LEFT)
        self.main_attr_comb.current(0)

        main_attr_lab = ttk.Label(frame, text="主属性的编辑步长:", style="InterFaceGAN.TLabel")
        main_attr_lab.pack(side=TOP,pady=10)

        self.main_attr_scale = Scale(frame, from_=-10, to=10, orient=HORIZONTAL, resolution=0.01, length=400, bg="#CCFFCC")
        self.main_attr_scale.pack(side=TOP);self.main_attr_scale.set(0)

        # 约束属性
        temp_frame2 = Frame(frame, bg=color)
        temp_frame2.pack(side=TOP, pady=10)
        cond_attr_lab = ttk.Label(temp_frame2, text="约束属性:", style="InterFaceGAN.TLabel")
        cond_attr_lab.pack(side=LEFT, padx=10)

        self.cond_attr_comb = ttk.Combobox(temp_frame2)
        self.cond_attr_comb['value'] = boundary_names
        self.cond_attr_comb.pack(side=LEFT)
        self.cond_attr_comb.current(0)

        temp_frame3 = Frame(frame, bg=color)
        temp_frame3.pack(side=TOP)
        cond_attr_lab = ttk.Label(temp_frame3, text="约束因子:", style="InterFaceGAN.TLabel")
        cond_attr_lab.pack(side=LEFT, padx=10)

        self.cond_attr_scale = Scale(temp_frame3, from_=-10, to=10, orient=HORIZONTAL, resolution=0.01, length=200, bg=color)
        self.cond_attr_scale.pack(side=LEFT);self.cond_attr_scale.set(0)

        temp_frame4 = Frame(frame, bg=color)
        temp_frame4.pack(side=TOP,pady=20)

        def change_comb(event):
            selection = self.cond_attrs_listbox.curselection()
            attr_name = self.cond_attrs_listbox.get(selection)
            attr_name = attr_name[:attr_name.find(":")]
            for ind,boundary_name in enumerate(boundary_names):
                if boundary_name.startswith(attr_name):
                    self.cond_attr_comb.current(ind)

        self.cond_attrs_listbox = Listbox(temp_frame4,width=30, height=5)
        self.cond_attrs_listbox.pack(side=LEFT, padx=10)
        self.cond_attrs_listbox.bind("<Double-Button-1>", change_comb)

        #添加条件属性的相关操作
        def add_condi_attr():
            cond_attr_name = self.cond_attr_comb.get()
            cond_attr_factor = self.cond_attr_scale.get()
            cond_nums = self.cond_attrs_listbox.size()
            insert_ind = END
            for ind in range(cond_nums):
                if self.cond_attrs_listbox.get(ind).startswith(cond_attr_name):
                    self.cond_attrs_listbox.delete(ind)
                    insert_ind = ind
                    break
            self.cond_attrs_listbox.insert(insert_ind, "{}:{}".format(cond_attr_name, cond_attr_factor))
        add_btn = ttk.Button(temp_frame4, style="InterFaceGAN.TButton", text="添加/更新", command=add_condi_attr)
        add_btn.pack(side=TOP, pady=10, expand=NO)

        #移除条件属性的相关操作
        def delete_condi_attr():
            selection = self.cond_attrs_listbox.curselection()
            if len(selection) == 0:
                return
            self.cond_attrs_listbox.delete(selection)
        remove_btn = ttk.Button(temp_frame4, style="InterFaceGAN.TButton", text="移除", command=delete_condi_attr)
        remove_btn.pack(side=TOP, pady=10, expand=NO)

    def random_gen(self):
        start_time = time.time()
        latent, image = random_gen()
        end_time = time.time()

        self.cur_image = self.org_image = image
        self.cur_latent = self.org_lantent = latent

        self.update_imagePanel(end_time-start_time)

    def load_latent(self):

        file_path = askopenfilename(title='选择隐向量',initialdir="StyleGAN_edit/data")  # initialdir=(os.path.expanduser("")

        start_time = time.time()
        latent = np.load(file_path)
        image = inference_core(input_feed=latent)
        end_time = time.time()

        # 加载隐向量那么久重新绘图,会修改self.cur_image和self.org_image,
        # 也会修改self.cur_latent和self.org_lantent
        self.cur_image = self.org_image = image
        self.cur_latent = self.org_lantent = latent

        self.update_imagePanel(end_time-start_time)

    def create_head(self, frame, color):
        frame.pack(side=TOP)
        ## Tkinter 中的fg,bg 在ttk中并不被支持，ttk是通过style这个对象来实现的。
        ttk.Style().configure('head.TButton', font=('Helvetica', 20), foreground="black", background=color)
        self.time_show_lab = Label(frame, text= "用时:0.00s")
        self.time_show_lab.pack(side=LEFT, padx=20)

        randomGen_btn = ttk.Button(frame, style="head.TButton", text="随机生成人脸图像", command= self.random_gen)
        randomGen_btn.pack(side=LEFT)

        loadLatent_btn = ttk.Button(frame, style="head.TButton", text="加载隐向量",command= self.load_latent)
        loadLatent_btn.pack(side=LEFT)

        self.latent_choice = IntVar()
        self.latent_choice.set(1)
        Radiobutton(frame, bg=color, variable=self.latent_choice, text="Z空间", value=0).pack(side=LEFT)
        Radiobutton(frame, bg=color, variable=self.latent_choice, text="W空间", value=1).pack(side=LEFT)
        Radiobutton(frame, bg=color, variable=self.latent_choice, text="W+空间", value=2).pack(side=LEFT)
        update_btn = ttk.Button(frame, style="head.TButton", text="更新", command=self.update)
        update_btn.pack(side=LEFT)


if __name__ == '__main__':
    root = Tk()
    tool = Edit_Framework(root)
    root.mainloop()

