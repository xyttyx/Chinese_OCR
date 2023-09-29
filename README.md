# Chinese_OCR
手写中文文本行识别，使用CRNN+CTC，在HWBD2.x数据集上accuracy达到**0.95**

# 训练方式
请在当前工程文件夹的上一级文件夹中，创建数据集文件夹，默认文件夹名称为**Chinese_OCR_data**。此文件夹默认结构如下
>Chinese_OCR_data
>
>  |--datasets
>
>  |  |--Test_images
>
>  |  |--Test_label
>
>  |  |--Train_images
>
>  |  |--Train_label
>
>  |--model  

请确认图片文件夹和标签文件夹下图片与标签名称相同，内容对应。
请注意，默认标签为.txt文件，图片为.jpg文件。如果使用HWBD2.x数据集，则可以使用utils下的trans_dgrl.py将.dgrl文件分离为图片和标签。
**如需使用trans_dgrl.py，请确认并修改文件中指向的路径**

在命令行中输入
`python train.py`
开始训练

# 测试方式
使用evaluate.py，会使用测试集中的数据进行评估。评估所输出的accuracy计算公式如下
*accuracy = average(1 - 编辑距离/标签字符串长度)*
编辑距离的定义和计算方式可自行搜索，实现方式放在util.editDistance之中
