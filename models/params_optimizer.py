# self.hyper_param =	{
#       "lr": 0.2,          # learning rate
#       "batch_size": 5,
#       'filter_num': [4,4],
#       'layers_num': 2,
#       'filter_size': [2,2],
#       'fc_size': [128,128],
#       'input_channel' : 1,
#       'concession': 0.02  # error within 2%
#     }
import decimal
import itertools
from .cnn_model import cnn 

def cnn_optimize(X,Y):
    lr = [decimal.Decimal(i) / decimal.Decimal(10) for i in range(1, 3)]
    # filter_num = list(range(50,100))
    keys = ['lr','filter_num','layers_num']
    filter_num =[i for i in range(8,160,8)]
    layers_num = list(range(1,5))
    filter_dim = [ list(itertools.permutations(filter_num, i)) for i in layers_num ]
    combs = []
    for l in layers_num:
        combs = combs + list(itertools.product(lr,filter_dim[l - 1],[l]))    # l -1 is not a good practice change it later ^^
    param_list = [dict(zip(keys, c)) for c in combs]
    
    print('param list length ',len(param_list))
    # start KCV ..
    precision = dict()
    demo_cnn = cnn(X,Y,param_list[0])
    print('good ',param_list[100000])
    for i in range(100000,100100):
        print(param_list[i])
        precision[str(i)] = demo_cnn.kcv(param_list[i])
        print('training idx : ' + str(i))
        print('precision: ' , precision[str(i)])

    # # write report
    # with open('CNN_KCV_REPORT.txt', 'w') as f:
    #     for idx,p in enumerate(precision):
    #         f.write("%s\t%s" % precision % param_list[idx])



# cnn_optimize() # ez debug