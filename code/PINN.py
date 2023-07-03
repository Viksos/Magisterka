import torch
import torch.nn as nn

class PINN(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        
        activation = nn.Tanh
        
        self.lambda_1 = nn.Sequential(*[
                        nn.Linear(1, 1,bias=False),
                        nn.ReLU()])
        
        self.lambda_2 = nn.Sequential(*[
                        nn.Linear(1, 1,bias=False),
                        nn.ReLU()])
    
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN+2, N_HIDDEN+2),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN+2, N_OUTPUT)
         
    def forward(self, x):
        
        x = self.fcs(x)
        l1 = self.lambda_1(torch.Tensor([1]))
        l2 = self.lambda_1(torch.Tensor([1]))

        x = torch.cat((x,torch.ones(len(x),1)*l1,torch.ones(len(x),1)*l2), dim=1)

        x = self.fch(x)
        
        x = self.fce(x)
        
        return x

    #   class Data():    


class Data_and_Plot():

    import matplotlib.pyplot as plt
    import os
    

    def load_data(self,path:str):
         
        import numpy as np
        import scipy
        
        data = scipy.io.loadmat(path)
        
        U_star = data['U_star'] # N x 2 x T
        P_star = data['p_star'] # N x T
        t_star = data['t'] # T x 1
        X_star = data['X_star'] # N x 2

        N = X_star.shape[0]
        T = t_star.shape[0]

        # Rearrange Data 
        XX = np.tile(X_star[:,0:1], (1,T)) # N x T
        YY = np.tile(X_star[:,1:2], (1,T)) # N x T
        TT = np.tile(t_star, (1,N)).T # N x T

        UU = U_star[:,0,:] # N x T
        VV = U_star[:,1,:] # N x T
        PP = P_star # N x T

        x_e = XX.flatten()[:,None] # NT x 1
        y_e = YY.flatten()[:,None] # NT x 1
        t_e = TT.flatten()[:,None] # NT x 1

        u_e = UU.flatten()[:,None] # NT x 1
        v_e = VV.flatten()[:,None] # NT x 1
        p_e = PP.flatten()[:,None] # NT x 1
        
        return(x_e,y_e,t_e,u_e,v_e,p_e)

    def plot(self,model,x_e,y_e,t_e,p_e,path,id):

        x_w = torch.Tensor(x_e).view(-1,1)
        y_w = torch.Tensor(y_e).view(-1,1)
        t_w = torch.Tensor(t_e).view(-1,1)
        
        with torch.no_grad():
            p_psi = model(torch.cat([x_w,y_w,t_w],1))
            p_pre = p_psi[:,0]
            psi_pre = p_psi[:,1]
    
        p_train = torch.Tensor(p_e).view(-1,1)

        fig = self.plt.figure()

        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.scatter(x_w.view(1,-1).numpy()[0].tolist()[::1000], y_w.view(1,-1).numpy()[0].tolist()[::1000], p_train.view(1,-1).numpy()[0].tolist()[::1000])

        ax_1 = fig.add_subplot(1,2,2, projection='3d')
        ax_1.scatter(x_w.view(1,-1).numpy()[0].tolist()[::1000], y_w.view(1,-1).numpy()[0].tolist()[::1000], p_pre.view(1,-1).numpy()[0].tolist()[::1000])

        self.plt.savefig(path+"/graph_"+str(id+1)+".jpg")
        self.plt.close()
    
    
    def plot_loss_in_time_life(self,path,name_of_activation_function_on_the_output,loss_lst):
    
        self.plt.plot(range(len(loss_lst)),loss_lst)
        self.plt.savefig(path+"/loss_of_"+name_of_activation_function_on_the_output+"_function.jpg")
        self.plt.close()
    
    def delete_temp_plots(self,path):
        for file in self.os.listdir(path):
            self.os.remove(path+"/"+file)   
