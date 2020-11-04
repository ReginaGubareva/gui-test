# Не нужно до конца тренировать CNN
# Нужно только извлечь данные из изображения,
# чтобы передать их LSTM and A3C
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from A3C_LSTM import A3Clstm


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

input_image = Image.open(fr"D:\guit-test-python\resources\sber1.PNG").convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

action_space = ['L', 'R']
print(A3Clstm(torch.nn.functional.softmax(output[0], dim=0), action_space)

# model = models.alexnet(pretrained=True)
# features = model.features
# exact_list = ["conv1","layer1","avgpool"]
# extractor = FeatureExtractor.forward()


# def get_vector(image_name):
#     # 1. Load the image with Pillow library
#     img = Image.open(image_name)
#
#     # 2. Create a PyTorch Variable with the transformed image
#     t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
#
#     # 3. Create a vector of zeros that will hold our feature vector
#     #    The 'avgpool' layer has an output size of 512
#     my_embedding = torch.zeros(512)
#
#     # 4. Define a function that will copy the output of a layer
#     def copy_data(m, i, o):
#         my_embedding.copy_(o.data)
#
#     # 5. Attach that function to our selected layer
#     h = layer.register_forward_hook(copy_data)
#
#     # 6. Run the model on our transformed image
#     model(t_img)
#
#     # 7. Detach our copy function from the layer
#     h.remove()
#
#     # 8. Return the feature vector
#     return my_embedding
#
#
#
# # Load the pretrained model
# model = models.resnet18(pretrained=True)
#
# # Use the model object to select the desired layer
# layer = model._modules.get('avgpool')
#
# # Set model to evaluation mode
# model.eval()
#
# scaler = transforms.Scale((224, 224))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# to_tensor = transforms.ToTensor()
#
# pic_one_vector = get_vector(fr"D:\gui-test-python\resources\sber1.PNG")
#
# print(pic_one_vector.data)


# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         self.submodule = submodule
#
#     def forward(self, x):
#         outputs = []
#         for name, module in self.submodule._modules.items():
#             x = module(x)
#             if name in self.extracted_layers:
#                 outputs += [x]
#         return outputs + [x]
