#!/usr/bin/python
# -*- coding: UTF-8 -*-
# python2  使用 Tkinter
from tkinter import *


# python3 使用 tkinter
# from tkinter import *
def say_hi():
    print("hello ~ !")

'''
1、pack函数布局的时候，默认先使用的放到上面，然后依次向下排列，默认方式它会给我们的组件一个自认为合适的位置和大小。
2、pack函数也可以接受几个参数，side参数，指定了它停靠在哪个方向，可以为LEFT,TOP,RIGHT,BOTTOM,分别代表左，上，右，下，
   它的fill参数可以是X,Y,BOTH,NONE即在水平方向填充，竖直方向填充，水平和竖直方向填充和不填充。
3、它的expand参数可以是YES 和 NO,它的anchor参数可以是N,E,S,W(这里的NESW分别表示北东南西，这里分别表示上右下左)以及他们的组合或者是CENTER(表示中间)
4、它的ipadx表示的是内边距的x方向，它的ipady表示的是内边距的y的方向，padx表示的是外边距的x方向，pady表示的是外边距的y方向。

https://blog.csdn.net/superfanstoprogram/article/details/83713196
'''
root = Tk()
root['width'] = 1880
root['height'] = 1300
# root.resizable(width=False, height=False)
root.title("tkinter frame")

frame1 = Frame(root, bg="blue",height=400,width=300)
# frame1.resizable(width=False, height=False)
frame1.pack()


# Label(frame1, text="Label",bg="red",width=10,height=5).pack(side = LEFT)
# Label(frame1, text="Label",bg="green",width=10,height=5).pack()

Label(frame1, text="Label1",bg="red",width=10,height=5).pack(side='left', expand='no', anchor='w', fill='y', padx=5, pady=5)
Label(frame1, text="Label2",bg="yellow",width=10,height=5).pack(side='top')
Label(frame1, text="Label5",bg="pink",width=10,height=5).pack(side='right')
Label(frame1, text="Label4",bg="orange",width=10,height=5).pack(side='bottom')

# hi_there = Button(root, text="say hi~", command=say_hi)
# hi_there.pack(side = LEFT,padx=20)



root.mainloop()