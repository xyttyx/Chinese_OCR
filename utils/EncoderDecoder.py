import os
import pickle

import torch

class EncoderDecoder():
    def __init__(self,c2i_pkl_dict:str|dict = None, i2c_pkl_dict:str|dict = None):
        if isinstance(c2i_pkl_dict,str) and os.path.exists(c2i_pkl_dict):
            with open(c2i_pkl_dict,'rb') as c2i_file: 
                self.c2i_dict = pickle.load(c2i_file)
                assert isinstance(self.c2i_dict, dict)
        elif isinstance(c2i_pkl_dict,dict):
            self.c2i_dict = c2i_pkl_dict.copy()
        else:
            self.c2i_dict = None
        if isinstance(i2c_pkl_dict,str) and os.path.exists(i2c_pkl_dict):
            with open(i2c_pkl_dict,'rb') as i2c_file: 
                self.i2c_dict:dict = pickle.load(i2c_file)
                assert isinstance(self.i2c_dict, dict)
        elif isinstance(i2c_pkl_dict,dict):
            self.i2c_dict = i2c_pkl_dict.copy()
        else:
            self.i2c_dict = None

    def StringEncode(self, string:str)->torch.Tensor:
        assert self.c2i_dict is not None
        enc = torch.zeros(len(string), dtype = torch.int32)
        for i in range(len(string)):
            enc[i] = self.c2i_dict[string[i]] if string[i] in self.c2i_dict.keys() else 0
        return enc
    
    def StringDecode(self, enc:torch.Tensor|list)->str:
        assert self.i2c_dict is not None
        string = str()
        for i in range(len(enc)):
            key = int(enc[i])
            if key == 0 or key > len(self.i2c_dict):
                string += '#'
            else:
                string += self.i2c_dict[key]
        return string
    
    def TensorDecode(self,probs:torch.Tensor)->str: #(T C)
        assert self.i2c_dict is not None
        T, C = probs.size()
        enc = torch.zeros(T, dtype=torch.int32)
        for i in range(T):
            T_prob = probs[i,:]
            max_index =int(torch.argmax(T_prob))
            enc[i] = max_index
        tmp_string = self.StringDecode(enc)
        string = str()
        c = ''
        for i in range(T):
            if c != tmp_string[i]:
                c = tmp_string[i]
                string += c
        return string.replace('#', '')