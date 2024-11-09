import torch
import collections
net_G0 = torch.load(f"../checkpoints/0/0_net_G.pth")
net_G1 = torch.load(f"../checkpoints/1/0_net_G.pth")
print(net_G0.state_dict().keys())
netG0 = net_G0.state_dict()
netG1 = net_G1.state_dict()
net_G = {}
for key in netG0.keys():
    net_G[key] = (netG0[key] + netG1[key]) / float(2)

save_path_G = f"../checkpoints/mean/0_net_G.pth"
torch.save(net_G, save_path_G)
model.load_networks('mean')
netG = torch.load(save_path_G)
print(netG.state_dict().keys())