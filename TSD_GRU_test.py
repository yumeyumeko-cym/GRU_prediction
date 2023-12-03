import pandas
import numpy as np
import torch
from torch import nn
import csv
from time import *
import Functions

begin = time()
# input data
Trend, Cycle, Effect, train_size, test_size = Functions.import_TSD_data()

# Hyperparameters
LOOK_BACK = 6
INPUT_FEATURES_NUM = 2*LOOK_BACK + 1
HIDDEN_SIZE = 8
OUTPUT_FEATURES_NUM = 1
NUM_LAYERS = 1
max_epochs = 1000
LEARNING_RATE = 0.02
circle = 5

# GRU Neural Networks
class GRU_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.gru = nn.GRU((input_size - 1), hidden_size, num_layers)
        self.forwardProcessing = nn.Linear((hidden_size + 1), output_size)

    # override initial forward
    def forward(self, x_input):
        # x_input is the input matrix
        # size (seq length, batch, input_size)
        x = x_input[:, :, 0:(INPUT_FEATURES_NUM-1)]
        e = x_input[:, :, (INPUT_FEATURES_NUM-1)].reshape(1, x_input.shape[1], 1)
        x, _ = self.gru(x)
        x = torch.cat([x, e], dim=2)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.forwardProcessing(x)
        x = x.view(s, b, -1)
        return x
    



# TSD-GRU formulation
def TSD_GRU(gru_model, trend, cycle, effect):

    # define training and testing sets after applying Time-Series-Decomposition
    train_trend = trend[:train_size]
    test_trend = trend[train_size:]
    train_cycle = cycle[:train_size]
    test_cycle = cycle[train_size:]
    effect = effect.reshape(-1, 1)
    train_effect = effect[LOOK_BACK+LOOK_BACK:train_size]
    test_effect = effect[LOOK_BACK+LOOK_BACK+train_size:]
    

    # create training set
    train_trend, train_y = Functions.create_RNNs_dataset(train_trend, look_back=LOOK_BACK)
    _, train_cycle = Functions.create_cycle_dataset(train_cycle, look_back=LOOK_BACK)
    
    train_x = np.concatenate((train_trend, train_cycle, train_cycle), axis = 1)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    
    batch_size = train_x.shape[0]

    input_feature = train_x.shape[1]
    train_x_tensor = train_x.reshape(1, batch_size, input_feature)
    train_x_tensor = torch.tensor(train_x_tensor, dtype=torch.float32)
    train_y_tensor = train_y.reshape(1, batch_size, OUTPUT_FEATURES_NUM)
    train_y_tensor = torch.tensor(train_y_tensor, dtype=torch.float32)


    # create testing sets
    test_trend, test_y = Functions.create_RNNs_dataset(test_trend, look_back=LOOK_BACK)
    _, test_cycle = Functions.create_cycle_dataset(test_cycle, look_back=LOOK_BACK)

    test_x = np.concatenate((test_trend, test_cycle, test_effect), axis=1)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    test_batch_size = test_x.shape[0]

    test_input_feature = test_x.shape[1]
    test_x_tensor = test_x.reshape(1, test_batch_size, test_input_feature)
    test_x_tensor = torch.tensor(test_x_tensor, dtype=torch.float32)
    test_y_tensor = test_y.reshape(1, test_batch_size, OUTPUT_FEATURES_NUM)
    test_y_tensor = torch.tensor(test_y_tensor, dtype=torch.float32)

    # evaluation matrices
    MSE_list = np.zeros([circle])
    RMSE_list = np.zeros([circle])
    MAPE_list = np.zeros([circle])
    RAE_list = np.zeros([circle])
    R2_list = np.zeros([circle])



    # training process
    for i in range(circle):
        print('circle =', i)

        """
        Model parameters are initilized randomly. This is done by creating random weights and biases for GRU
        layers and the fully connected layer. It resets the model's state for each cycle, allowing for an
        independent evaluation of the model's performance across cycles. 

        *refer to the model of GRU

        """
        
        gru_weight1 = torch.randn(size=[3*HIDDEN_SIZE, (INPUT_FEATURES_NUM-1)])
        gru_weight2 = torch.randn(size=[3*HIDDEN_SIZE, HIDDEN_SIZE])
        gru_bias1 = torch.randn(size=[3*HIDDEN_SIZE])
        gru_bias2 = torch.randn(size=[3*HIDDEN_SIZE])
        fc_weight = torch.randn(size=[OUTPUT_FEATURES_NUM, (HIDDEN_SIZE+1)])
        fc_bias = torch.randn(size=[OUTPUT_FEATURES_NUM])
        weight_bias = [gru_weight1, gru_weight2, gru_bias1, gru_bias2, fc_weight, fc_bias]        

        for epoch in range(max_epochs):
            # initialize the model
            weight_bias_index = 0
            for name, parameter in gru_model.named_parameters():
                if parameter.requires_grad:
                    parameter.data = weight_bias[weight_bias_index]
                    weight_bias_index += 1
            output = gru_model(train_x_tensor)
            loss = loss_function(output, train_y_tensor)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # visualize the process
            if (epoch + 1) % 100 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))



        # prediction process
        pre_y = gru_model(test_x_tensor)
        y_hat = torch.reshape(pre_y, [test_batch_size]).detach().numpy()
        y_act = torch.reshape(test_y_tensor, [test_batch_size]).detach().numpy()

        # error calculation
        error = y_hat - y_act
        abs_error = list(map(abs, error))
        squared_err = np.multiply(error, error)

        MSE_list[i] = np.mean(squared_err)
        MAPE_list[i] = np.mean(abs_error / y_act)
        RAE_list[i] = np.sum(abs(y_hat - y_act)) / np.sum(abs(np.mean(y_act) - y_act))

        # calculation of R2
        # R-squared is a statistical measure that represents the goodness of fit of a regression model.
        # sum of squared residuals
        RSS = np.sum(np.multiply((y_hat - y_act), (y_hat - y_act)))
        # total sum of squares
        TSS = np.sum(np.multiply((y_act - np.mean(y_act)), (y_act - np.mean(y_act))))
        R2_list[i] = 1 - RSS / TSS
        print(MAPE_list[i])   

    av_MSE = np.mean(MSE_list)
    av_MAPE = np.mean(MAPE_list)
    av_RMSE = np.sqrt(av_MSE)
    av_RAE = np.mean(RAE_list)
    av_R2 = np.mean(R2_list)
    return av_MSE, av_RMSE, av_MAPE, av_RAE, av_R2









# main functions
gru_model = GRU_model(input_size=INPUT_FEATURES_NUM, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_FEATURES_NUM, num_layers=NUM_LAYERS)
for name, parameter in gru_model.named_parameters():
    if parameter.requires_grad:
        print(parameter.shape)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
MSE_list = np.zeros([30])
RMSE_list = np.zeros([30])
MAPE_list = np.zeros([30])
RAE_list = np.zeros([30])
R2_list = np.zeros([30])

for k in range(30):
    MSE_list[k], RMSE_list[k], MAPE_list[k], RAE_list[k], R2_list[k]  = TSD_GRU(gru_model, Trend[k], Cycle[k], Effect[k])

print('RMSE=', RMSE_list, '\nMAPE=', MAPE_list, '\nRAE=', RAE_list, '\nR2=', R2_list)

# output metrics
output_list = [MSE_list, RMSE_list, MAPE_list, RAE_list, R2_list]
f = open('30min_TSD-GRU_output_list.csv', 'w', newline='')
csv_writer = csv.writer(f)
for l in output_list:
    csv_writer.writerow(l)
    print('writing')
f.close()

end = time()
run_time = end - begin
print('run_time=', run_time)
