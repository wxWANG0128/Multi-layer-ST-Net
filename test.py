import util
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

processor = 'cuda:0'
data = 'data/BAY_NEW'
adjdata='data/sensor_graph/adj_mx_bay.pkl'
seq_length = 12
nhid = 32
batch_size = 64
dropout = 0.3
checkpoint = 'garage/bay23_net2/_exp1_best_1.22.pth'
plotheatmap = 'True'
TCN_kernel_size = 2
TCN_dilation = 2
save = 'plot/bay_new/'

num_nodes = np.load(data + '/val.npz')['x'][0,0,:,0].shape[0]
in_dim = np.load(data + '/val.npz')['x'][0,0,0,:].shape[0]

def main():
    device = torch.device(processor)
    adj_mx = util.load_adj(adjdata)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    model = net2(device, num_nodes, TCN_kernel_size, TCN_dilation, dropout, supports=supports, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
    model.to(device)
    if processor == 'cpu':
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(data, batch_size, batch_size, batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))

    if plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="hot_r",vmin=0.3, vmax=1)
        plt.savefig(save+"emb-la" + '.png',dpi=300)

    y12 = realy[2000,:,11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[2000,:,11]).cpu().detach().numpy()

    y3 = realy[2000,:,2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[2000,:,2]).cpu().detach().numpy()

    y = realy.cpu().detach().numpy()
    yhat = scaler.inverse_transform(yhat).cpu().detach().numpy()

    np.save(save+'y',y)
    np.save(save+'yhat',yhat)

    np.save(save+'y12',y12)
    np.save(save+'yhat12',yhat12)
    np.save(save+'y3',y3)
    np.save(save+'yhat3',yhat3)




if __name__ == "__main__":
    main()
