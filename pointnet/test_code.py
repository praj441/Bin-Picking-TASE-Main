import torch

gqs_map = torch.randn((2,3,2))

PX = torch.tensor([[1,1],[0,0]])
PY = torch.tensor([[2,1],[1,0]])
seed_pixels = PY*2+PX

gqs_target = torch.gather(gqs_map.view(gqs_map.shape[0],-1),1,seed_pixels.to(torch.long))


print('PX',PX)
print('PY',PY)
print('seed_pixels',seed_pixels)
print('gqs_map',gqs_map)
print('gqs_target',gqs_target)