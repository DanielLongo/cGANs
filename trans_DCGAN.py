import torch
from cDCGAN import ConditionalGenerator
pretrained_generator_filepath = "test_g.pt"
pretrained_generator = Generator()
pretrained_generator.load_state_dict(torch.load(pretrained_generator_filepath))
# pretrained_generator = torch.load(pretrained_generator_filepath)
print("type", pretrained_generator.type)
# print(pretrained_generator)
print("layer2", pretrained_generator.layer2)
print("output of layer2", (pretrained_generator.layer2(torch.ones((22,512,22,512)))).shape)
# print(model.load_state_dict(pretrained_generator))
# print(pretrained_generator.layer2)
