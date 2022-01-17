import torch

from torchvision.models import resnet18

model=resnet18(pretrained=True)

script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

dummy=torch.randn(1,3,224,224,dtype=torch.float)[:5]

print(dummy.shape)
assert torch.allclose(model(dummy), script_model(dummy))


