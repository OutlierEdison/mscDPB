from django.shortcuts import render
#from django.http import HttpResponse
from keras.models import load_model
import os
import numpy as np
import keras
keras.backend.clear_session()
from sklearn.metrics import roc_curve, auc
from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from django.http import FileResponse  
from django.shortcuts import render,redirect,HttpResponse,HttpResponseRedirect  
from django.contrib.auth import authenticate
from django.urls import reverse
import keras.preprocessing.sequence as kps
# Create your views here.


def download_template(request):
    file = open('static/files/train_1.data', 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="train_1.data"'
    return response
def download_test(request):
    file = open('static/files/test_1.data', 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="test_1.data"'
    return response
def download_name(request):
    file = open('static/files/690_datasets.xlsx', 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="690_datasets.xlsx"'
    return response
def download_readme(request):
    file = open('static/files/readme.txt', 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="readme.txt"'
    return response


def mscDPB(request):
    return render(request,"mscDPB.html")
 
class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
   '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs *(K.tanh(K.softplus(inputs)))
get_custom_objects().update({'Mish': Mish(mish)})

#反向互补序列编码
def DNA_complement(sequence):
    sequence = sequence.upper()#这里是在前一条序列的基础上进行替换，一定要注意，不能都是大写
    sequence = sequence.replace('A', 't')
    sequence = sequence.replace('T', 'a')
    sequence = sequence.replace('C', 'g')
    sequence = sequence.replace('G', 'c')
    return sequence.upper()


def DNA_reverse(sequence):
    sequence = sequence.upper()
    return sequence[::-1]

def seq_code():
    alphabet = ['A','C','G','T']
    temp=['A','C','G','T']
    mapper=['A','C','G','T']
    for base in range(len(temp)):
        for letter in alphabet:
            mapper.append(temp[base] + letter)
    code = np.eye(len(mapper), dtype = int)
    encoder = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i,:])
    return encoder

def sequence_dict(seq):
    '''
    if seq[0] == '>':  # or line.startswith('>')
        name = line[1:].upper()  # discarding the initial >
        seq_dict[name] = ''
    else:
        seq_dict[name] = seq_dict[name] + line
    return seq_dict
    '''
    seq_dict= {}
    name = ''
    seq = seq.rstrip()  
    name = seq.split(' ')[0]
    seq_dict[name] = seq.split(' ')[1]
    return seq_dict

def feature_code(seq_dict):
    one_hot_data = []
    for k, v in seq_dict.items():
        seq_content=v
    temp_one_data=[]
    encode=seq_code()
    for i in seq_content:
        if i in encode.keys():
            temp_one_data.append(encode[i])
    #单个碱基的编码
    
    temp_multinucleotide = []
    seq_length = len(seq_content)
    windows_length=2
    for i in range(seq_length-windows_length+1):
        multinucleotide = seq_content[i:i+windows_length]
        if multinucleotide in encode.keys():    
            temp_multinucleotide.append(encode[multinucleotide])
       #双碱基的编码
    
    rev_one_data=[]
    seq_reverse=DNA_reverse(DNA_complement(seq_content))
    str_list = [seq_content, seq_reverse]
    a = ''
    temp_seq= a.join(str_list)
    for j in temp_seq:
        if j in encode.keys():
            rev_one_data.append(encode[j])
    #反向互补的编码
    forant=np.concatenate([temp_one_data,temp_multinucleotide],axis=0)
    back=np.concatenate([forant,rev_one_data],axis=0)
    one_hot_data.append(back)
    
    post_seq = kps.pad_sequences(one_hot_data, maxlen=403, dtype='int32',
                           padding='post', truncating='post')
    test_data = post_seq.reshape((-1,403, 20))
    data = np.array(test_data)   #训练集数据
    return data

def test_action(request):
    seq=request.POST.get("sequences","").strip()
    mname=request.POST.get("model_name")
    if seq =='' or mname=='' or ">" not in seq:
        return HttpResponse('Message: please input again.')
    else:    
    #seq='>chr1:44497068-44497168 AGGCTCTGTGCCGCGCCGAGTTCGCCCGCCCCGCCGCGCCGCTCGCAGCTCTTCCACAGCCTGTTGTGTTTTGGTTTCGGGGAGGCGGGGGCTAAGAGTTT'
        seq_dict=sequence_dict(seq)
        data_test=feature_code(seq_dict)
        number=mname[1:]
        path1 = "mscDPBmodel/mm"
        file_read_path = "{}/{}{}.{}".format(path1,"model",str(number), "hdf5")
        mod = load_model(file_read_path)
        proba = mod.predict(data_test)#决策值
        K.clear_session()
        for i in proba:
            proba_value=i
        predict = (proba > 0.5).astype('int32') 
        for j in predict:
            predict_label=j
        #title='Boosting DNA-protein binding prediction with multi-scale complementary feature from Chip-seq'
        list_seq='%s'%(seq)
        id_seq = seq.rstrip()  
        id = id_seq.split(' ')[0]
        list_pro_value=' %.4f'%(proba_value)
        threshold=0.5
        if float(list_pro_value)<0.5:
            label='N'
        else:
            label='B'
        list_predict_label=' %d'%(predict_label)
        #return HttpResponseRedirect("/result_show/")
        #return render(request,'result_show.html')    
    return render(request,'result_show.html',{'id':id,'list_seq':list_seq,'list_pro_value':list_pro_value,'threshold':threshold,'list_predict_label':list_predict_label,'label':label})

    


    