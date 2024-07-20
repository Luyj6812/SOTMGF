# show_pkl.py

import torch
path = '/media/dell/新加卷/Lu/code_kongzhuan2/stMGAC_mouse_brain/model_concat.pkl'
f = open(path,'rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu
#print(data)
#####data.__dict__
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
#plt.rcParams['font.sans-serif']='SimHei'#设置中文显示，必须放在sns.set之后
np.random.seed(0)
uniform_data = data['transformer.fc.weight'] #设置二维矩阵
np.savetxt("transformer.fc.weight.csv",data['transformer.fc.weight'],delimiter=',')
np.savetxt("cluster_layer.csv",data['cluster_layer'],delimiter=',')

f, ax = plt.subplots(figsize=(100, 6))
uniform_data = np.array(uniform_data)
#heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
#参数annot=True表示在对应模块中注释值
# 参数linewidths是控制网格间间隔
#参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
#参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
sns.heatmap(uniform_data, ax=ax,vmin=-1.5,vmax=1.5,cmap='bwr',annot=True,linewidths=2,cbar=True)

ax.set_title('fc_weight') #plt.title('热图'),均可设置图片标题
ax.set_ylabel('class')  #设置纵轴标签
ax.set_xlabel('feature')  #设置横轴标签

#设置坐标字体方向，通过rotation参数可以调节旋转角度
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')

plt.show()

###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#设置绘图风格
plt.style.use('ggplot')
#处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
# plt.rcParams['axes.unicode_minus']=False
# 读取数据
tips = pd.read_csv('/media/dell/新加卷/Lu/code_kongzhuan2/stMGAC_mouse_brain/data/murine_breast_cancer/weight_transformer.csv',header=0, index_col=None)
# 绘制分组小提琴图
sns.violinplot(x = "class", # 指定x轴的数据
               y = "modality_weight", # 指定y轴的数据
               hue = "modality", # 指定分组变量
               data = tips, # 指定绘图的数据集
               order = ['1','2','3','4','5'], # 指定x轴刻度标签的顺序
               scale = 'count', # 以男女客户数调节小提琴图左右的宽度
               split = True, # 将小提琴图从中间割裂开，形成不同的密度曲线；
               palette = 'RdBu' # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
              )
# 添加图形标题
plt.title('modality weight')
# 设置图例
plt.legend(loc = 'upper left', ncol = 2)
# 显示图形
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#设置绘图风格
plt.style.use('ggplot')
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
# 读取数据
tips = pd.read_excel(r'酒吧消费数据.xlsx')
# 绘制分组小提琴图
sns.violinplot(x = "day", # 指定x轴的数据
               y = "total_bill", # 指定y轴的数据
               hue = "sex", # 指定分组变量
               data = tips, # 指定绘图的数据集
               order = ['Thur','Fri','Sat','Sun'], # 指定x轴刻度标签的顺序
               scale = 'count', # 以男女客户数调节小提琴图左右的宽度
               split = True, # 将小提琴图从中间割裂开，形成不同的密度曲线；
               palette = 'RdBu' # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
              )
# 添加图形标题
plt.title('每天不同性别客户的酒吧消费额情况')
# 设置图例
plt.legend(loc = 'upper right', ncol = 2)
#控制横纵坐标的值域
plt.axis([-1,4,-10,70])
# 显示图形
plt.show()