import torch
from torch import nn
from torchvision.models import resnet18

class Encoder(nn.Module):
    '''
        To get the embedding, simply use the network as usual, the returned
        output would be a avgpool embedding
    '''
    def __init__(self, model, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.model = model
        self.model.eval()
        self.target_layer = self.model._modules.get('avgpool')
        self.embedding = torch.zeros(batch_size, 512)
        self.hook = self.target_layer.register_forward_hook(self._pullHook)
        
    def _pullHook(self, m, i, o):
        self.embedding.copy_(o.data.view(self.batch_size, 512))
   
    def forward(self, x):
        _ = self.model(x)
        return self.embedding

    def get_embedding(self, x):
        self.model(x)
        return self.embedding