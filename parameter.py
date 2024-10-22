import torch
from torchsummary import summary
from model.attention.SGE import  SpatialGroupEnhance
from models.model import CNNModel
# Example model
model = CNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the summary of the model
summary(model, input_size=(3, 224, 224))  # Get the number of parameters
