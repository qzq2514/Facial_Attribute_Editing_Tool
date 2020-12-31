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
from Utils import check_dir
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
        self.temp_save_dir = check_dir("temp_dir")
        self.image_exp_NST = self.cur_image = self.org_image = misc.imread(org_path) #图像都是原1024 x 1024大小的，只不过到时候在panel中显示的时候会有缩放
        self.cur_latent = self.org_lantent = np.load(latent_path)
        self.ATMGAN_edit_info = {}   #存放ATMGAN编辑需要的固定的属性配置(ATMGAN_org_info的子集)
        self.ATMGAN_factors = {}     #存放ATMGAN编辑每个属性属性的动态编辑因子
        self.ATMGAN_grid_info = {}   #存放当前需要使用ATMGAN编辑的属性中对应的激活张量位置和图像画的rect的id
                                     #其内元素形式为:
        # {'Curly_Hair':
        #              {'resolution': 8,
        #              'location': {loca1:id1,loca2:id2...}},
        #  "Eyeglasses":
        #              {'resolution': 16,
        #              'location': {loca1:id1,loca2:id2...}}}

        self.InterFace_edit_info = {}

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

        self.face_ImagePanel.bind("<Button-1>", self.facePanelClick)  # 绑定鼠标左键(<Button-1>)单击事件
        self.face_ImagePanel.bind("<Button-2>", self.facePanelClick_right)  # 绑定鼠标右键(<Button-1>)单击事件

        # 中间使用ATMGAN和InterFaceGAN进行人脸属性编辑的功能区域
        mid_frame = Frame(self.tkObj)
        mid_frame.pack(side=LEFT)

        mid_top_frame = Frame(mid_frame)
        mid_top_frame.pack(side=TOP)

        ttk.Style().configure('update_save.TButton', font=('Helvetica', 40), foreground="black")
        update_btn = ttk.Button(mid_top_frame, style="update_save.TButton", text="更新", command=self.update)
        update_btn.pack(side=LEFT,padx=10)
        save_btn = ttk.Button(mid_top_frame, style="update_save.TButton", text="保存", command=self.save)
        save_btn.pack(side=LEFT)

        ATMGAN_frame = Frame(mid_frame, bg="#FFFFCC")
        ATMGAN_frame.pack(side=TOP, pady=10, ipady=10)
        self.create_ATMGAN(ATMGAN_frame, color="#FFFFCC")

        InterFaceGAN_frame = Frame(mid_frame, bg="#CCFFCC")
        InterFaceGAN_frame.pack(side=TOP, ipady=10)
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
        main_attr_lab.pack(side=TOP, pady=10)

        self.main_attr_scale = Scale(frame, from_=-10, to=10, orient=HORIZONTAL, resolution=0.01, length=400,
                                     bg="#CCFFCC")
        self.main_attr_scale.pack(side=TOP, pady=10);self.main_attr_scale.set(0)

        temp_frame3 = Frame(frame,bg=color)
        temp_frame3.pack(side=TOP)

        def change_main_attr(event):
            # 双击主属性中集合中的某个主属性发生的事件
            # 1.更新self.main_attr_comb和scale
            selection = InterFaceGAN_main_attrs_listbox.curselection()
            selection_str = InterFaceGAN_main_attrs_listbox.get(selection)
            main_attr_name_with_ind = selection_str[:selection_str.find(":")]
            scale = float(selection_str[selection_str.find(":")+1:selection_str.find("--->")])

            self.main_attr_scale.set(scale)
            for ind, boundary_name in enumerate(boundary_names):
                if boundary_name.startswith(main_attr_name_with_ind):
                    self.main_attr_comb.current(ind)

            # 2.更新下面的约束属性集合的显示
            # 2.1删除约束属性集合的原来所有元素
            self.cond_attrs_listbox.delete(0,END)
            # 2.2添加当前主属性下的所有约束属性
            condi_attrs = self.InterFace_edit_info[main_attr_name_with_ind]["cond_boundaries"]
            for condi_attr_name,condi_attr_info in condi_attrs.items():
                self.cond_attrs_listbox.insert(END,"{}: {}".format(condi_attr_name,condi_attr_info["cond_factor"]))


            # 2.更新约束属性集合显示
        InterFaceGAN_main_attrs_listbox = Listbox(temp_frame3,width = 30,height=5)
        InterFaceGAN_main_attrs_listbox.pack(side=LEFT, padx=10)
        InterFaceGAN_main_attrs_listbox.bind("<Double-Button-1>", change_main_attr)

        def add_main_attr():
            #添加InterFaceGAN的主要编辑属性
            selection = self.main_attr_comb.get()
            attr_name = selection[selection.find("_")+1:]
            scale = self.main_attr_scale.get()
            main_attrs_num = InterFaceGAN_main_attrs_listbox.size()

            #先判断是不是已经有该属性信息了,有则先删除,然后统一更新
            insert_ind = END
            have_main_attr=False
            for ind in range(main_attrs_num):
                if InterFaceGAN_main_attrs_listbox.get(ind).startswith(selection):
                    InterFaceGAN_main_attrs_listbox.delete(ind)
                    insert_ind = ind
                    have_main_attr=True
                    break

            #1.向InterFaceGAN_main_attrs_listbox添加信息
            InterFaceGAN_main_attrs_listbox.insert(insert_ind, "{0}: {1}".format(selection, scale))

            #2.向self.InterFace_edit_info添加信息
            boundary = np.load(os.path.join("StyleGAN_edit/models/boundaries/for_latent_w", "{}_w/{}_boundary.npy".
                                            format(selection, attr_name)))
            self.InterFace_edit_info[selection]={'main_boundary': {'name': attr_name,
                                                                   'factor': scale,
                                                                   'main_boundary': boundary},
                                                 'cond_boundaries': {}}
            # 3.如果是添加了新的删除约束属性集合内所有的元素(因为换新的主编辑元素了)
            if not have_main_attr:
                self.cond_attrs_listbox.delete(0, END)

        add_btn = ttk.Button(temp_frame3, style="InterFaceGAN.TButton", text="添加/更新", command=add_main_attr)
        add_btn.pack(side=TOP, pady=10, expand=NO)

        def delete_main_attr():
            # 删除InterFaceGAN的编辑属性
            # 1. 先删除InterFaceGAN_main_attrs_listbox中对应的属性
            selection = InterFaceGAN_main_attrs_listbox.curselection()
            if len(selection) == 0:
                return
            attr_name_with_ind = InterFaceGAN_main_attrs_listbox.get(selection).split(":")[0]
            print("attr_name_with_ind:", attr_name_with_ind)
            InterFaceGAN_main_attrs_listbox.delete(selection)

            # 2.删除self.InterFace_edit_info中对应的属性信息
            self.InterFace_edit_info.pop(attr_name_with_ind)

        remove_btn = ttk.Button(temp_frame3, style="InterFaceGAN.TButton", text="移除", command=delete_main_attr)
        remove_btn.pack(side=TOP, expand=NO)

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
        temp_frame4.pack(side=TOP,pady=10)

        def change_comb(event):
            # 双击属性约束集合中的某个元素
            # 1.更新cond_attr_comb的显示
            selection = self.cond_attrs_listbox.curselection()
            selection_str = self.cond_attrs_listbox.get(selection)
            attr_name = selection_str[:selection_str.find(":")]
            for ind,boundary_name in enumerate(boundary_names):
                if boundary_name.startswith(attr_name):
                    self.cond_attr_comb.current(ind)

            # 2.更新self.cond_attr_scale
            scale = float(selection_str[selection_str.find(":") + 1:selection_str.find("--->")])
            self.cond_attr_scale.set(scale)

        self.cond_attrs_listbox = Listbox(temp_frame4,width=30, height=5)
        self.cond_attrs_listbox.pack(side=LEFT, padx=10)
        self.cond_attrs_listbox.bind("<Double-Button-1>", change_comb)

        #添加条件属性的相关操作
        def add_condi_attr():
            # 为当前主要编辑属性添加约束属性
            # 获取当前约束属性相关
            cond_attr_name = self.cond_attr_comb.get()
            cond_attr_factor = self.cond_attr_scale.get()
            cond_nums = self.cond_attrs_listbox.size()

            main_attr_name_with_ind = self.main_attr_comb.get()
            # 当前添加的约束属性必须是属于self.main_attr_comb所显示的主属性，如果不存在则不做任何事
            if main_attr_name_with_ind not in self.InterFace_edit_info.keys():
                return

            # 如果确实存在对应的主属性,那么开始做事
            # 1.先向self.cond_attrs_listbox中添加并更新属性
            insert_ind = END
            for ind in range(cond_nums):
                if self.cond_attrs_listbox.get(ind).startswith(cond_attr_name):
                    self.cond_attrs_listbox.delete(ind)
                    insert_ind = ind
                    break
            self.cond_attrs_listbox.insert(insert_ind, "{}:{}".format(cond_attr_name, cond_attr_factor))

            # 2.向self.InterFace_edit_info添加该约束属性的信息
            cond_attr_name_no_num = cond_attr_name[cond_attr_name.find("_")+1:]
            cond_boundary = np.load(os.path.join("StyleGAN_edit/models/boundaries/for_latent_w", "{}_w/{}_boundary.npy".
                                                 format(cond_attr_name, cond_attr_name_no_num)))
            self.InterFace_edit_info[main_attr_name_with_ind]["cond_boundaries"][cond_attr_name_no_num] = {'cond_factor': float(cond_attr_factor),
                                                                                                            'cond_boundary': cond_boundary}

        add_btn = ttk.Button(temp_frame4, style="InterFaceGAN.TButton", text="添加/更新", command=add_condi_attr)
        add_btn.pack(side=TOP, pady=10, expand=NO)

        #移除条件属性的相关操作
        def delete_condi_attr():
            selection = self.cond_attrs_listbox.curselection()
            if len(selection) == 0:
                return
            # 1.从self.cond_attrs_listbox中删除显示
            condi_attr_name_with_ind = self.cond_attrs_listbox.get(selection).split(":")[0]
            self.cond_attrs_listbox.delete(selection)

            # 2.从self.InterFace_edit_info中删除该约束属性的信息
            main_attr_name_with_ind = self.main_attr_comb.get()

            # 2.1当前删除的约束属性的主属性必须是属于self.main_attr_comb所显示的主属性，并且如果该不存在InterFace_edit_info中，则不做任何事
            if main_attr_name_with_ind not in self.InterFace_edit_info.keys():
                return
            # 2.2如果存在那么就从该主属性中删除该约束属性
            condi_attr_name = condi_attr_name_with_ind[condi_attr_name_with_ind.find("_")+1:]
            print(condi_attr_name_with_ind,condi_attr_name)
            self.InterFace_edit_info[main_attr_name_with_ind]["cond_boundaries"].pop(condi_attr_name)

        remove_btn = ttk.Button(temp_frame4, style="InterFaceGAN.TButton", text="移除", command=delete_condi_attr)
        remove_btn.pack(side=TOP, pady=10, expand=NO)

    # def get_InterFaceGAN_edit_info(self):
    #     # 获得InterFaceGAN的编辑信息
    #     # 1.获取主编辑属性
    #     main_attr_factor = self.main_attr_scale.get()
    #     boundary_name = self.main_attr_comb.get()
    #     boundary = np.load(os.path.join("StyleGAN_edit/models/boundaries/for_latent_w", "{}_w/{}_boundary.npy".
    #                                     format(boundary_name, boundary_name[boundary_name.find("_") + 1:])))
    #     InterFaceGAN_edit_info = {"main_boundary": {'name': 'Mouth_Slightly_Open',
    #                                                 'factor': main_attr_factor,
    #                                                 'main_boundary': boundary},
    #                               "cond_boundaries": {}}
    #     # 2.获取约束属性
    #     cond_nums = self.cond_attrs_listbox.size()
    #     for ind in range(cond_nums):
    #         edit_info = self.cond_attrs_listbox.get(ind)
    #         cond_boundary_name, cond_factor = edit_info.split(":")
    #         cond_boundary_name_no_num = cond_boundary_name[cond_boundary_name.find("_") + 1:]
    #         cond_boundary = np.load(os.path.join("StyleGAN_edit/models/boundaries/for_latent_w", "{}_w/{}_boundary.npy".
    #                                              format(cond_boundary_name, cond_boundary_name_no_num)))
    #         # print(cond_boundary_name, cond_factor, cond_boundary_name_no_num)  # 5_Bangs 0.71 Bangs
    #
    #         InterFaceGAN_edit_info["cond_boundaries"][cond_boundary_name_no_num] = {'cond_factor': float(cond_factor),
    #                                                                                 'cond_boundary': cond_boundary}
    #     return InterFaceGAN_edit_info

    # 鼠标单击人脸图像界面发生的事件-画某个单元格或清除某个单元格
    def facePanelClick(self, event):
        # 获得当前画网格是在哪个局部属性下面
        attr_name = self.ATMGAN_attrs_comb.get()
        # 刚初始化或者删除了某个属性后再点击，这时候self.ATMGAN_grid_info中还没内容
        if attr_name not in self.ATMGAN_grid_info.keys():
            return
        reso = self.ATMGAN_grid_info[attr_name]["resolution"]
        x, y = event.x, event.y  #获取当前鼠标位置
        grid_height, grid_width = image_height // reso, image_width // reso
        location_id = y//grid_height*reso+x//grid_width
        #说明原来已经画过了,那么这里就不再画一次(后期可能考虑对不同位置的特征值引入不同的权重，这时候下面就要修改了，更复杂了，以后再说)
        cur_locations = self.ATMGAN_grid_info[attr_name]["location"]
        if location_id in cur_locations.keys() and cur_locations[location_id]!=-1:
            return

        #没画过说明原来ATMGAN_grid_info中该属性的激活值不包括这一个，那么就把该位置激活值添加进去并画网格
        self.ATMGAN_grid_info[attr_name]["location"][location_id] = self.draw_single_grid(reso, location_id)
        # print("Add:", len(self.ATMGAN_grid_info[attr_name]["location"]),"-----", self.ATMGAN_grid_info[attr_name]["location"])

    # 右击清除单元格
    def facePanelClick_right(self, event):
        attr_name = self.ATMGAN_attrs_comb.get()
        if attr_name not in self.ATMGAN_grid_info.keys():
            return

        reso = self.ATMGAN_grid_info[attr_name]["resolution"]
        x, y = event.x, event.y  # 获取当前鼠标位置
        grid_height, grid_width = image_height // reso, image_width // reso
        location_id = y // grid_height * reso + x // grid_width
        cur_locations = self.ATMGAN_grid_info[attr_name]["location"]
        #如果本来就没有网格，那么就不用删除了
        if location_id not in cur_locations.keys() or cur_locations[location_id]==-1:
            return

        #否则删除！！
        self.face_ImagePanel.delete(self.ATMGAN_grid_info[attr_name]["location"][location_id])
        self.ATMGAN_grid_info[attr_name]["location"].pop(location_id)
        # print("Delete:", len(self.ATMGAN_grid_info[attr_name]["location"]),"-----", self.ATMGAN_grid_info[attr_name]["location"])

    # 根据特征图的分辨率和特征值的下标(行优先)画出当前的grid
    def draw_single_grid(self, reso, location):
        grid_height, grid_width = image_height // reso, image_width // reso
        row_ind, col_ind = location // reso, location % reso
        rect_id = self.face_ImagePanel.create_rectangle(grid_width * col_ind, grid_height * row_ind,
                                                        grid_width * (col_ind + 1), grid_height * (row_ind + 1), outline="green",)
        return rect_id

    def update_ATMGAN_edit_info(self):
        for attr_name,edit_info in self.ATMGAN_edit_info.items():
            new_location_ids = [k for k,v in self.ATMGAN_grid_info[attr_name]["location"].items()]
            # print("new_location_ids:",new_location_ids)
            self.ATMGAN_edit_info[attr_name]["location"] = new_location_ids

    def save(self):
        save_name = time.strftime("%Y_%m_%d_%H_%M_%S.jpg", time.localtime(time.time()))
        misc.imsave(os.path.join(self.temp_save_dir, save_name), self.cur_image)

    def update(self):
        # 依次整合InterFaceGAN(Z/W空间)、Style_Mixing(W+空间)和ATMGAN(激活张量空间)的编辑信息,然后一次性进行编辑

        # 1.从self.InterFace_edit_info中多组属性中获得最终在Z/W空间获得根据InterFaceGAN移动后的隐向量
        moved_latent = self.org_lantent
        for selection, attr_edit_info in self.InterFace_edit_info.items():
            moved_latent, edit_info = get_InterFaceGAN_move_direction(moved_latent, attr_edit_info)
        # 单独使用InterFaceGAN进行编辑(inference时候不考虑Style_Mixing和ATMGAN)
        # # 这里暂时让InterFaceGAN都是最原始的latent上进行编辑,后续可能考虑添加一个选项,
        # # 让使用者决定是从原始开始编辑还是从当前InterFaceGAN的编辑结果进一步编辑,改动的话,这里只需要将self.org_lantent变为self.cur_latent就行
        # edit_res, latent, edit_info = InterFaceGAN_edit(self.org_lantent, InterFaceGAN_edit_info)
        self.cur_latent = moved_latent  #更新W空间的latent

        # 2.在激活张量空间获取使用ATMGAN进行编辑所需要的信息
        # 2.1先根据之前再人脸图像中画各个属性下的网格更新ATMGAN_edit_info
        self.update_ATMGAN_edit_info()
        # 2.2然后再使用ATMGAN编辑结果图
        self.ATMGAN_feed = get_ATMGAN_feed(self.ATMGAN_edit_info,self.ATMGAN_factors)

        start_time = time.time()
        # print("self.cur_latent:",self.cur_latent[0][:10])
        edit_res = inference_core(self.cur_latent, self.ATMGAN_feed)
        end_time = time.time()
        # print("----------")

        #最终的编辑结果收到InterFaceGAN、Style_Mixing和AMTGAN的综合影响
        self.cur_image = edit_res
        self.update_imagePanel(end_time - start_time,draw_grid=True)

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

    def draw_edit_grid(self,attr_name):
        # attr_info就是ATMGAN_edit_config.py中每个属性的编辑初始信息
        if attr_name not in self.ATMGAN_grid_info.keys():
            return
        reso = self.ATMGAN_grid_info[attr_name]["resolution"]
        locations = self.ATMGAN_grid_info[attr_name]["location"].keys()
        for loca in locations:
            # 如果原来已经画过了，那么就要删除该方框
            cur_rect_id = self.ATMGAN_grid_info[attr_name]["location"][loca]
            if cur_rect_id != -1:
                self.face_ImagePanel.delete(cur_rect_id)
            self.ATMGAN_grid_info[attr_name]["location"][loca] = self.draw_single_grid(reso, loca)
        # print(self.ATMGAN_grid_info)

    def create_ATMGAN(self, frame, color):
        #ttk.Style()命名必须以Txxxx结尾,其中xxxx是button/Label之类，当然还可以在之前加自定义的参数，如"my2."
        ttk.Style().configure('AMTGAN.TButton', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('AMTGAN.TLabel', font=('Helvetica', 20), foreground="black", background=color)
        ttk.Style().configure('AMTGAN.TNotebook', font=('Helvetica', 20), foreground="black", background=color)

        ATMGAN_lab = ttk.Label(frame, text="ATMGAN", style="AMTGAN.TLabel")
        ATMGAN_lab.pack(side=TOP, pady=10)

        def change_ATMGAN_attr_show(event=None):
            attr_name = self.ATMGAN_attrs_comb.get()
            attr_info = ATMGAN_org_info[attr_name]
            ATMGAN_attr_intro_lab.config(text="属性名称:{0}\n张量分辨率:{1}x{1}\n特征图:[{2}]".
                           format(attr_name, attr_info["resolution"], ",".join([str(x) for x in attr_info["fmap"]])))

        attr_names = self.get_ATMGAN_attr_names()
        self.ATMGAN_attrs_comb = ttk.Combobox(frame)
        self.ATMGAN_attrs_comb['value'] = attr_names
        self.ATMGAN_attrs_comb.pack(side=TOP, pady=10)
        self.ATMGAN_attrs_comb.current(0)
        self.ATMGAN_attrs_comb.bind("<<ComboboxSelected>>", change_ATMGAN_attr_show)

        ATMGAN_attr_intro_lab = Label(frame, text="属性名称:{0}\n张量分辨率:{1}x{1}\n特征图:[{2}]".
                           format(attr_names[0],ATMGAN_org_info[attr_names[0]]["resolution"],",".join([str(x) for x in ATMGAN_org_info[attr_names[0]]["fmap"]])),bg=color)
        ATMGAN_attr_intro_lab.pack(side=TOP)
        # 滑动条:orient，控制滑块的方位，HORIZONTAL（水平），VERTICAL（垂直）,通过resolution选项可以控制分辨率（步长），通过tickinterval选项控制刻度
        ATMGAN_scale = Scale(frame, from_=-100, to=100, orient=HORIZONTAL, resolution=0.1, length=400, bg=color)
        ATMGAN_scale.pack(side=TOP, pady=10);ATMGAN_scale.set(0)

        temp_frame1 = Frame(frame, bg=color)
        temp_frame1.pack(side=TOP, pady=0)

        def change_comb(event):
            # 双击ATMGAN中代表当前待编辑属性集合的Listbox中的元素，会发生以下的事情
            # 1.更新下拉框的显示和ATMGAN_attr_intro_lab标签的显示
            selection = self.ATMGAN_attrs_listbox.curselection()
            selection_str = self.ATMGAN_attrs_listbox.get(selection)
            attr_name = selection_str[:selection_str.find(":")]
            for ind, name in enumerate(attr_names):
                if name.startswith(attr_name):
                    self.ATMGAN_attrs_comb.current(ind)
                    break
            change_ATMGAN_attr_show()

            # 2.先删除原panel中所有的图像和矩形并画上原来的人脸图
            # (其实就是清除原panel中所有的矩形，保留人脸图)
            self.face_ImagePanel.delete(ALL)
            self.update_imagePanel(0.00,draw_grid=True)

            # 3.重新绘制下拉框所指属性对应的grids
            # (其实这时候self.ATMGAN_grid_info[attr_name]中已经有信息了，只不过线条被清除了，所以需要重新画)
            self.draw_edit_grid(attr_name)

            # 4.把滑动条更新到当前选择的属性的编辑因子
            info_without_name = selection_str[selection_str.find(":")+1:]
            attr_scale = info_without_name[:info_without_name.find("--->")]
            print(attr_scale)
            attr_scale = float(attr_scale)
            ATMGAN_scale.set(attr_scale)

        self.ATMGAN_attrs_listbox = Listbox(temp_frame1, width=30, height=5)
        self.ATMGAN_attrs_listbox.pack(side=LEFT, padx=10)
        self.ATMGAN_attrs_listbox.bind("<Double-Button-1>", change_comb)

        #添加ATMGAN待编辑的属性
        def add_ATMGAN_edit_attr():
            attr_name = self.ATMGAN_attrs_comb.get()
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
            self.ATMGAN_attrs_listbox.insert(insert_ind, "{0}: {1}--->{2}x{2}   [{3}]".format(attr_name, attr_factor, attr_info["resolution"],
                                                                                       ",".join([str(x) for x in attr_info["fmap"]])))
            # 2.ATMGAN_edit_info添加相关信息
            self.ATMGAN_edit_info[attr_name] = attr_info
            # 3.ATMGAN_factors添加相关信息
            self.ATMGAN_factors[attr_name] = attr_factor
            # 4.删除图像界面中已经有的所有网格
            self.face_ImagePanel.delete(ALL)  #先把Canves上的内容全删了delete(ALL), 然后再重新画上人脸图
            self.update_imagePanel(0.00,draw_grid=True)

            # 5.在图像中添加网格表示该属性的待编辑区域
            # 初始化self.ATMGAN_grid_info[attr_name]中的"resolution"信息和"location"信息
            self.ATMGAN_grid_info[attr_name]={"resolution": attr_info["resolution"]}
            self.ATMGAN_grid_info[attr_name]["location"]={k:-1 for k in attr_info["location"]}  #初始值全部为-1，表明还没开始画
            self.draw_edit_grid(attr_name)  #重新绘制self.ATMGAN_grid_info[attr_name]中的包含的网格信息

        add_btn = ttk.Button(temp_frame1, style="AMTGAN.TButton", text="添加/更新", command=add_ATMGAN_edit_attr)
        add_btn.pack(side=TOP, pady=5)

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
            #4.移除ATMGAN_grid_info中该属性的编辑网格信息
            self.ATMGAN_grid_info.pop(attr_name)
            #5.如果图像画的正是当前删除属性的网格，那么也删除图中的网格
            if attr_name == self.ATMGAN_attrs_comb.get():
                self.update_imagePanel(0.00, draw_grid=False)

        remove_btn = ttk.Button(temp_frame1, style="AMTGAN.TButton", text="移除",command = remove_ATMGAN_edit_attr)
        remove_btn.pack(side=TOP, pady=0)

    def update_imagePanel(self, spend_time, update_from_NST=False, draw_grid = False):
        #防止风格迁移的效果不断累加,同时能够保证self.cur_image一直都是大窗口显示的人脸结果
        if not update_from_NST:
            self.image_exp_NST = self.cur_image
        self.imgPIL_org = ImageTk.PhotoImage(Image.fromarray(self.cur_image).resize((image_width, image_height)))
        # 更新界面:
        # 1.先把原来界面上所有图像，矩形全部删除
        self.face_ImagePanel.delete(ALL)
        # 2.画上新人脸图
        self.face_ImagePanel.create_image(0, 0, image=self.imgPIL_org, anchor=NW)
        # 3.如果需要再画上原图中的矩形

        if draw_grid:
            attr_name = self.ATMGAN_attrs_comb.get()
            self.draw_edit_grid(attr_name)

        self.time_show_lab.config(text="用时:{:.2f} s".format(spend_time))

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


if __name__ == '__main__':
    root = Tk()
    tool = Edit_Framework(root)
    root.mainloop()

