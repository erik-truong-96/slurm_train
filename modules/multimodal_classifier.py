import torch
import torch.nn as nn
import lightning as L
import torchvision.models as models
from transformers import DistilBertModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class BERT_CNN_Classifier(L.LightningModule):
    
    def __init__(self, input_size, num_layers, num_channels, classes):
        super().__init__()

        ######### Learning rate #########
        self.lr = 0.01

        ######### Parameters #########
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.classes = classes

        ######### Criterion #########
        self.criterion = nn.CrossEntropyLoss()


        ########## BERT #########
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        bert_out_size = self.bert_model.config.hidden_size
    

        ######### CNN #########
        #models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # models.efficientnet.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT) 
        self.cnn_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        cnn_out_size = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()

        #cnn_out_size = self.cnn_model.classifier[-1].in_features
        #self.cnn_model.fc.out_features = nn.Identity()   #.classifier = nn.Identity()

        ######### Dropout #########
        self.dropout = nn.Dropout(0.2)

        ######### Fully connected layers #########
        self.fc1 = nn.Linear(cnn_out_size+bert_out_size, classes)


    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataloader    

    def training_step(self, batch, batch_idx):
        tokens, images, labels = batch
       
        predictions = self(tokens, images)
        loss = self.criterion(predictions, labels)

        # calculate accuracy
        #total = labels.size(0)
        predicted = torch.argmax(predictions, dim=1)
        # correct = torch.eq(predicted, labels).sum().item()
        # accuracy = correct / total
        # self.log("train-accuracy", accuracy)
        #wandb.log({"train-accuracy": accuracy})

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='weighted',zero_division=1)
        accuracy = accuracy_score(labels, predicted)

        # Log metrics
        self.log("train-loss", loss, prog_bar=True)
        self.log("train-accuracy", accuracy, prog_bar=True)
        self.log("train-precision", precision, prog_bar=True)
        self.log("train-recall", recall, prog_bar=True)
        self.log("train-f1", f1, prog_bar=True)

        return loss
    
    def forward(self, tokens, image):

        ####### Forward pass through CNN #####
        cnn_x = self.cnn_model(image)
        cnn_x = cnn_x.view(cnn_x.size(0), -1)
        #####################################
        
        ####### Forward pass through BERT #####
        attention_mask = (tokens != 0).to(torch.long)
        bert_x = self.bert_model(tokens, attention_mask=attention_mask).last_hidden_state[:,-1,:]
        #####################################
        
        # Concatenate the outputs of the two models
        concat_x = torch.cat((cnn_x, bert_x), dim=1)

        # Add dropout layer
        concat_x = self.dropout(concat_x)
        
        # Fully connected layer
        out = self.fc1(concat_x)
        
        return out
    
if __name__ == "__main__":
    pass
    