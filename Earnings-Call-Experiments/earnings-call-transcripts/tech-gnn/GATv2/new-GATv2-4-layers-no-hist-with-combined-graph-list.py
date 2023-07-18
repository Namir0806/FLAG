import os
import time
import dgl
import torch as th
import torch.nn as nn
from dgl.nn import GATv2Conv
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.nn.functional import relu
import psutil
num_of_attn_heads = int(sys.argv[9])
num_of_layers = 4

try:
    os.mkdir('tracking-files')
    os.mkdir('mse-results')
    os.mkdir('best-valid-losses')
    os.mkdir('saved-model-weights')
    os.mkdir('training-logs')
except OSError as error:
    print(error)

device = th.device("cuda")
class Net(nn.Module):
    def __init__(self,num_heads):
        super(Net, self).__init__()
        #self.initial_lin = nn.Linear(300, 768)
        #self.num_attn_heads = num_attn_heads
        #self.layer1 = GraphConv(768, int(768/num_attn_heads), num_heads = num_attn_heads, activation = relu)
        self.num_attn_heads = num_heads
        self.layer0 = GATv2Conv(768,int(768/num_heads),num_heads=num_heads)
        self.layer1 = GATv2Conv(768,int(768/num_heads),num_heads=num_heads)
        self.layer2 = GATv2Conv(768,int(768/num_heads),num_heads=num_heads)
        self.layer3 = GATv2Conv(768,int(768/num_heads),num_heads=num_heads)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu =  nn.LeakyReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,600)

        #self.hist_fc = nn.Linear(1,256)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(600,1)

        #self.fc3 = nn.Linear(1,1)
    
    def forward(self, g, features):
        #features = self.initial_lin(features)
        #features =self.leakyrelu(features)
        #features = self.dropout(features)
        #g.to(device)
        #features.to(device)
        #g = dgl.add_self_loop(g)
        g = dgl.add_self_loop(g)
        x = self.layer0(g, features)
        attn_heads = [x[:,i,:] for i in range(self.num_attn_heads)]
        x = th.cat(attn_heads,dim=1)
        
        x = self.layer1(g, x)
        attn_heads = [x[:,i,:] for i in range(self.num_attn_heads)]
        x = th.cat(attn_heads,dim=1)
        #print(x.shape)
        x = self.layer2(g, x)
        attn_heads = [x[:,i,:] for i in range(self.num_attn_heads)]
        x = th.cat(attn_heads,dim=1)
        #print(x.shape)
        x = self.layer3(g, x)
        attn_heads = [x[:,i,:] for i in range(self.num_attn_heads)]
        x = th.cat(attn_heads,dim=1)

        x = x[0]
        #print(x.shape)

        x = self.fc1(x)
        x =self.leakyrelu(x)
        x = self.dropout(x) 

        #hist = hist.unsqueeze(0)
        #hist = self.fc3(hist)

        #print(x.shape)
        #print(hist.shape)
        #print(amber)
        #x = th.cat((x, hist))


        y = self.fc2(x)
        #y = self.leakyrelu(y)
        del g
        del features
        return x, y
net = Net(num_of_attn_heads).to(device)
#print(net)

def get_graph(index):
    return dgl.load_graphs(f'../../amrs-to-graphs/tech-2010-to-2018-plan-{sys.argv[6]}.graphs',[index])[0][0]

sec = sys.argv[1]

df = pd.read_csv('../../new-tech-2010-to-2018-result.csv')

bv = sys.argv[2]

hist = sys.argv[3]

#glist = dgl.load_graphs("train-result_ibm_graphs_list_hk-finbert_plan_"+sys.argv[6]+".graphs")[0]

#print(len(glist))
#print(len(df['prev_'+bv]))

train_index, valid_index, train_labels, valid_labels = train_test_split(list(range(len(df[bv]))),
    df[bv],
    shuffle=False,
    train_size=0.8)


#valid_index = valid_index.reset_index(drop=True)
#print(train_index)
#print(valid_index)
#print(amber)
#del glist
#train_hist = th.tensor(train_hist.tolist()).to(device)
#valid_hist = th.tensor(valid_hist.tolist()).to(device)
df_test = pd.read_csv('../../new-tech-2019-result.csv')
test_index = list(range(len(df_test[bv])))

test_labels = df_test[bv]

train_y = th.tensor(train_labels.tolist()).to(device)
valid_y = th.tensor(valid_labels.tolist()).to(device)
test_y = th.tensor(test_labels.tolist()).to(device)
#print(amber)
mse_loss  = nn.MSELoss() 
# function to train the model
def train(model,epoch):

    memory_file = open('tracking-files/memory_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_ep_'+str(epoch)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+sys.argv[3]+'.txt', 'a+')
    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds = []

    total_hist = []

    xs = []


    # iterate over list of documents
    for i in range(len(train_index)):

        memory_file.write("doc num: "+str(i)+"\n")
        #memory_file.write("doc num: "+str(i)+" before train: "+str(int(th.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
        #memory_file.write("doc num: "+str(i)+" before train: "+str(int(th.cuda.memory_allocated()/1024/1024))+' mem allocated\n')
        #memory_file.write('RAM memory % used:'+str(psutil.virtual_memory()[2])+' \n')
        memory_file.flush()

        sent_id = get_graph(train_index[i])
        features = sent_id.ndata['h']
        #hist = train_hist[i] 
        labels = train_y[i].unsqueeze(0)
        sent_id = sent_id.to(device)
        features = features.to(device)
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        x, preds = model(sent_id,features)
        #print(preds)
        #print(len(preds))
        #print(labels)

        # compute the loss between actual and predicted values
        #loss = huber_loss(preds, labels)
        loss = mse_loss(preds, labels)
        preds = preds.detach().cpu().numpy()
        x = x.detach().cpu().numpy().ravel()


        # model predictions are stored on GPU. So, push it to CPU
        #preds = preds.detach().cpu().numpy()
        #x = x.detach().cpu().numpy().ravel()

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # append the model predictions
        total_preds.append(preds)
        xs.append(x)

        #memory_file.write("doc num: "+str(i)+" after train: "+str(int(th.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
        #memory_file.write("doc num: "+str(i)+" after train: "+str(int(th.cuda.memory_allocated()/1024/1024))+' mem allocated\n')
        #memory_file.flush()

        #loss.detach().cpu()

        
        
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_index)

    xs = np.array(xs)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    #total_hist = np.concatenate(total_hist, axis=0)
    memory_file.close()
    #returns the loss and predictions
    return avg_loss, total_preds, xs

def train_x(model):

    #memory_file = open('memory_finbert_'+sec+'_'+bv+'_ep'+str(epoch)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.txt', 'a+')
    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds = []

    total_hist = []

    xs = []


    # iterate over list of documents
    for i in range(len(train_index)):

        #memory_file.write("doc num: "+str(i)+"\n")
        #memory_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')

        sent_id = get_graph(train_index[i])
        features = sent_id.ndata['h']
        #hist = train_hist[i] 
        labels = train_y[i].unsqueeze(0)
        sent_id = sent_id.to(device)
        features = features.to(device)
        # clear previously calculated gradients 
        #model.zero_grad()        
        with th.no_grad():
            # get model predictions for the current batch
            x, preds = model(sent_id,features)

            preds = preds.detach().cpu().numpy()
            x = x.detach().cpu().numpy().ravel()

            # append the model predictions
            total_preds.append(preds)
            xs.append(x)


    xs = np.array(xs)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    #total_hist = np.concatenate(total_hist, axis=0)
    #memory_file.close()
    #returns the loss and predictions
    return xs, total_preds

# function for evaluating the model
def evaluate(model):

    #print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0.0, 0.0
  
    # empty list to save the model predictions
    total_preds = []

    total_xs = []

    # iterate over list of documents
    for i in range(len(valid_index)):

        sent_id = get_graph(valid_index[i])
        features = sent_id.ndata['h']
        #mask = valid_mask[i]
        #hist = valid_hist[i]
        labels = valid_y[i].unsqueeze(0)
        sent_id = sent_id.to(device)
        features = features.to(device)
        # deactivate autograd
        with th.no_grad():
      
            #with autocast():
            # model predictions
            x, preds = model(sent_id,features )
            
            # compute the validation loss between actual and predicted values
            loss = mse_loss(preds,labels)
            preds = preds.detach().cpu().numpy()

            total_loss = total_loss + loss.item()

            #preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

            x = x.detach().cpu().numpy().ravel()

            total_xs.append(x)
        #loss.detach().cpu()

        

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(valid_index) 

    total_xs = np.array(total_xs)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, total_xs

def test(model,test_index):
    #memory_file = open('new-GATv2-MSE-results/memory_test_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+sys.argv[3]+'.txt', 'a+')

    model.eval()

    # empty list to save the model predictions
    total_xs = []

    total_preds=[]
    
    total_loss = 0.0

    for i in range(len(test_index)):

        #memory_file.write("doc num: "+str(i)+"\n")
        #memory_file.write("doc num: "+str(i)+" before train: "+str(int(th.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
        #memory_file.write("doc num: "+str(i)+" before train: "+str(int(th.cuda.memory_allocated()/1024/1024))+' mem allocated\n')
        #memory_file.write('RAM memory % used:'+str(psutil.virtual_memory()[2])+' \n')
        #memory_file.flush()

        sent_id = get_graph(test_index[i])
        features = sent_id.ndata['h']
        #mask = test_mask[i]
        #hist = test_hist[i]
        labels = test_y[i].unsqueeze(0)
        sent_id = sent_id.to(device)
        features = features.to(device)
        with th.no_grad():
            #with autocast():
            x, preds = model(sent_id, features)
            
            #preds = preds.detach().cpu().numpy()

            

            loss = mse_loss(preds,labels)
            preds = preds.detach().cpu().numpy()
            total_loss = total_loss + loss.item()

            x = x.detach().cpu().numpy().ravel()
            total_preds.append(preds)
            total_xs.append(x)


            
    # reshape the predictions in form of (number of samples, no. of classes)
    total_xs = np.array(total_xs)
    avg_loss = total_loss / len(test_index) 
    total_preds = np.concatenate(total_preds, axis=0)
    #memory_file.close()
    return avg_loss,total_preds, total_xs

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
learning_rate = float(sys.argv[4])
optimizer = th.optim.Adam(net.parameters(), lr=learning_rate)
dur = []

total_epochs = int(sys.argv[5])
start_epoch = int(sys.argv[7])
end_epoch = int(sys.argv[8])
epochs = end_epoch - start_epoch + 1
# set initial loss to previous best
if (os.path.isfile('best-valid-losses/best_valid_loss_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt')):
    with open('best-valid-losses/best_valid_loss_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt') as f:
        lines = f.readlines()
    best_valid_loss = float(lines[0])
    best_epoch = int(lines[1])
else:
    best_valid_loss = float('inf')
    best_epoch = 0

if (os.path.isfile('saved-model-weights/saved_weights_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt')):
    path = 'saved-model-weights/saved_weights_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt'
    net.load_state_dict(th.load(path))
training_log = open('training-logs/training_log_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt', 'a+')
for epoch in range(start_epoch,end_epoch):
    start_time = time.time()
    #train model
    train_loss, _,_ = train(net,epoch)
    
    # append training and validation loss
    #train_losses.append(train_loss)
    #valid_losses.append(valid_loss)
            
    #print(f'\nTraining Loss: {train_loss:.10f}')
    #print(f'Validation Loss: {valid_loss:.10f}')
    #evaluate model
    valid_loss,_ ,_= evaluate(net)

    end_time = time.time()
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        #print(f'\nTraining Loss: {train_loss:.3f}')
        #xs_train = xs_final
        model_to_save = net.module if hasattr(net, 'module') else net
        th.save(model_to_save.state_dict(), 'saved-model-weights/saved_weights_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt')
        #torch.save(model_to_save.state_dict(), 'saved_weights_finbert_'+str(max_length)+'_'+sec+'_'+bv+'_epoch'+str(start_epoch+epoch)+'_of_'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm_hist.pt')
    print("Target: "+bv+" | Layers {:03d}| Heads {:03d}| Epoch {:03d} | Train Loss {:.9f} | Valid Loss {:.9f}| lr {:.9f} | time {:.3f}".format(
            num_of_layers,num_of_attn_heads, epoch, train_loss, valid_loss, learning_rate, end_time-start_time))
    training_log.write("Target: "+bv+" | Layers {:03d}| Heads {:03d}| Epoch {:03d} | Train Loss {:.9f} | Valid Loss {:.9f}| lr {:.9f} | time {:.3f}\n".format(
            num_of_layers,num_of_attn_heads, epoch, train_loss, valid_loss, learning_rate, end_time-start_time))
    training_log.flush()
training_log.close()
valid_loss_file = open('best-valid-losses/best_valid_loss_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt', 'w')
valid_loss_file.write(str(best_valid_loss)+"\n")
valid_loss_file.write(str(best_epoch))
valid_loss_file.close()
if end_epoch == total_epochs:
    #del train_text
    model = Net(num_of_attn_heads).to(device)

    path = 'saved-model-weights/saved_weights_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt'
    model.load_state_dict(th.load(path))

    _ , preds, xs_valid = evaluate(model)
    preds = np.asarray(preds)
    valid_y = valid_y.cpu().data.numpy()
    valid_mse = mean_squared_error(valid_y, preds)
    
    # get predictions for test data
    #years = ["2001","2002","2003","2004","2005","2006"]
    #for year in years:
    #th.cuda.empty_cache()
    #valid_y = th.tensor(valid_labels.tolist()).to(device)

    #sec = year

    #test_fname = year+"-result.csv"

    #TEST

    #df_test = pd.read_csv(test_fname)

    #test_index = list(range(len(df_test[bv])))
    #test_hist = df_test['prev_'+bv]
    #test_labels = df_test[bv]

    #test_hist = th.tensor(test_hist.tolist()).to(device)
    #test_y = th.tensor(test_labels.tolist()).to(device)

    _, preds,xs_test= test(model,test_index)
    preds = np.asarray(preds)
    np.savetxt('mse-results/test_preds_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.csv',preds, delimiter=',')
    test_y = test_y.cpu().data.numpy()
    np.savetxt('mse-results/test_trues_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.csv',test_y, delimiter=',')
    test_mse = mean_squared_error(test_y, preds)

    print("GATv2 "+str(num_of_layers)+" layers "+str(num_of_attn_heads)+" heads mse: "+str(test_mse)+'---bare---'+str(valid_mse))

    mse_file = open('mse-results/mse_GATv2_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt', "w")

    mse_file.write(str(test_mse)+'---bare---'+str(valid_mse)+"\n")
    mse_file.write(str(best_valid_loss)+" after epoch: "+str(best_epoch)+"\n")   

    mse_file.close()
