import torch
import torch.nn as nn
import PINN

torch.manual_seed(123)

Data_and_plot = PINN.Data_and_Plot()
model                   = PINN.PINN(3,2,10,5)
x_e,y_e,t_e,u_e,v_e,p_e = Data_and_plot.load_data('')#path to data file

model.load_state_dict(torch.load('')) # model file

time_spl = 0.2
x_train = torch.Tensor(x_e[t_e<=time_spl]).view(-1,1).requires_grad_(True)
y_train = torch.Tensor(y_e[t_e<=time_spl]).view(-1,1).requires_grad_(True)
t_train = torch.Tensor(t_e[t_e<=time_spl]).view(-1,1)
u_train = torch.Tensor(u_e[t_e<=time_spl]).view(-1,1)
v_train = torch.Tensor(v_e[t_e<=time_spl]).view(-1,1)
p_train = torch.Tensor(p_e[t_e<=time_spl]).view(-1,1)

x_grad = torch.Tensor(x_e[range(0,100000,100),:]).view(-1,1).requires_grad_(True)
y_grad = torch.Tensor(y_e[range(0,100000,100),:]).view(-1,1).requires_grad_(True)
t_grad = torch.Tensor(t_e[range(0,100000,100),:]).view(-1,1).requires_grad_(True)
u_grad = torch.Tensor(u_e[range(0,100000,100),:]).view(-1,1).requires_grad_(True)
v_grad = torch.Tensor(v_e[range(0,100000,100),:]).view(-1,1).requires_grad_(True)



optimizer           = torch.optim.Adam(model.parameters(),lr=1e-4)
alpha               = 10**(-4)
epochs              = 10**(6)
loss_lst            = []
losses              = {"loss1":[],"loss2":[]}

for i in range(epochs):
    optimizer.zero_grad()

    lambda_1        = list(model.parameters())[0]
    lambda_2        = list(model.parameters())[1]
    
    p_psi           = model(torch.cat([x_train,y_train,t_train],1))
    p_mt            = p_psi[:,0]
    psi_mt          = p_psi[:,1]
    
    u_mt            = torch.autograd.grad(psi_mt, y_train, torch.ones_like(psi_mt), create_graph=True)[0]
    v_mt            = -torch.autograd.grad(psi_mt, x_train, torch.ones_like(psi_mt), create_graph=True)[0]  
    
    loss_1          = torch.mean((p_mt -p_train)**2)+torch.mean((v_mt -v_train)**2)+torch.mean((u_mt -u_train)**2)
    
      
    p_psi = model(torch.cat([x_grad,y_grad,t_grad],1))
    p = p_psi[:,0]
    psi = p_psi[:,1]
 
    u = torch.autograd.grad(psi, y_grad, torch.ones_like(psi), create_graph=True)[0]
    v = -torch.autograd.grad(psi, x_grad, torch.ones_like(psi), create_graph=True)[0]  
        
    u_t = torch.autograd.grad(u, t_grad, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_grad, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_grad, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_grad, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_grad, torch.ones_like(u_y), create_graph=True)[0]
        
    v_t = torch.autograd.grad(v, t_grad, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_grad, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_grad, torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_grad, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_grad, torch.ones_like(v_y), create_graph=True)[0]
        
    p_x = torch.autograd.grad(p, x_grad, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_grad, torch.ones_like(p), create_graph=True)[0]
    
    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
    
    loss_2 = torch.sum(torch.square(f_u)) + \
                    torch.sum(torch.square(f_v))
    
    losses["loss1"].append(loss_1.detach().numpy());losses["loss2"].append(loss_2.detach().numpy())                 
    loss             = (1-alpha)*loss_1+alpha*loss_2   

    if(loss.detach().numpy()<0.015):
        break
    
    print(" epoch "+str(i)+" loss"+str(loss)+"p"+str(torch.mean((p_mt -p_train)**2))+"V"+str(torch.mean((v_mt -v_train)**2))+"u"+str(torch.mean((u_mt -u_train)**2)))
    
    loss_lst.append(loss.tolist())
    loss.backward()
    optimizer.step()
    
torch.save(model.state_dict(), '')#model save path
