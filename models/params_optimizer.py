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

def cnn_optimize(){
    lr = [decimal.Decimal(i) / decimal.Decimal(10) for i in range(0, 3)]
    filter_num = list(range(10,200))
    layers_num = list(range(1,5))
    
}