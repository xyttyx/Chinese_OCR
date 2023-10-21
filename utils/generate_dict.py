import os
import pickle

def generate_dict():
    save_folder = '.\\dictionary'#vscode运行时当前文件夹为utils的上一级文件夹，如果运行时当前文件夹为utils则需修改save_folder，下面的data同理
    i2c_dict_path = os.path.join(save_folder,'i2c_dict.pkl')
    c2i_dict_path = os.path.join(save_folder,'c2i_dict.pkl')
    
    if os.path.exists(i2c_dict_path):
        with open(i2c_dict_path,'rb') as file:
            i2c_dict = pickle.load(file)
            assert isinstance(i2c_dict,dict)
    else:
        i2c_dict = dict()
    if os.path.exists(c2i_dict_path):
        with open(c2i_dict_path,'rb') as file:
            c2i_dict = pickle.load(file)
            assert isinstance(c2i_dict,dict)
    else:
        c2i_dict = dict()

    if len(c2i_dict) == 0 and len(i2c_dict) == 0:
        pass
    elif len(c2i_dict) != 0 and len(c2i_dict) != 0:
        for key in c2i_dict.keys():
            key_ = c2i_dict[key]
            if i2c_dict[key_] == key:
                continue
            else:
                print('fatal dict load')
                return 
    else:
        print('fatal dict load')
        return
    
    update_dict(c2i_dict, i2c_dict)

    with open(c2i_dict_path,'wb') as file:
        pickle.dump(c2i_dict,file)
    with open(i2c_dict_path,'wb') as file:
        pickle.dump(i2c_dict,file)
    print('lenght of dict is {}'.format(len(c2i_dict)))
    
    return 



def update_dict(c2i_dict: dict, i2c_dict: dict):
    i_number = len(c2i_dict) + 1 # 序号0不分配，留给CTCloss的占位符
    list_string = generate_list_string()
    for string in list_string:
        for char in string:
            if char not in c2i_dict.keys():
                c2i_dict[char] = i_number
                i2c_dict[i_number] = char
                i_number += 1
    return



def generate_list_string():
    data_folder_path = '..\\Chinese_OCR_data\\datasets'
    txt_folder_name = 'Train_label'
    txt_folder_path = os.path.join(data_folder_path, txt_folder_name)
    list_string = list()
    for _0, _1, files in os.walk(txt_folder_path):
        for file in files:
            txt_file_path = os.path.join(txt_folder_path, file)
            with open(txt_file_path,'r',encoding='utf-8') as txt_file:
                context = txt_file.read()
                list_string.append(context)
    return list_string



if __name__ == '__main__':
    generate_dict()
