from typing import Any, Dict, Optional, Iterable, Iterator
import math
import copy

import torch
import torch.nn as nn

from transformers import BertModel, BertConfig, BertForSequenceClassification
from attention import CustomBertSelfAttention
from get_embed import EmbeddedFeatures

class BERTClassifierCostum(nn.Module):
    def __init__(self, transformer_config, eval_config, segment_config, computing_device='cpu'):
        super().__init__()

        self.computing_device = computing_device

        self.brush_embedding_flag = False
        if eval_config['Brush'] == 'None':
            self.brush_embedding_flag = True


        self.input_size = len(eval_config["data_fields"])
        if eval_config['use_euler_angles'] == True:
          self.input_size += 3
    
        self.hidden_size_transformer = transformer_config["hidden_size"]  #+transformer_config["position_embedding_dim"]
        # if self.brush_embedding_flag == True:
        #     self.hidden_size_transformer += transformer_config["brush_embedding_dim"]
        
        self.__init_upsample(transformer_config, eval_config)
        self.__init_embed(transformer_config, segment_config)
        self.__init_encoder(transformer_config, segment_config)
        self.__init_layer_drop(transformer_config["layer_drop_prob"])
        # self.__init_weight()

        self.activation = nn.ReLU()


        self.fc1 = nn.Linear(self.hidden_size_transformer, transformer_config["hidden_size"]//2)
        self.fc2 = nn.Linear(transformer_config["hidden_size"]//2, eval_config['num_dental_regs'])

    def __init_upsample(self, transformer_config, eval_config) -> None:
        upsample_choice = transformer_config["upsample_unit"]
        if upsample_choice.lower() == "cnn":
            # ! This is equivalent to fully connected network
            self.upsample = nn.Conv1d(
                self.input_size,
                transformer_config["hidden_size"],
                kernel_size=transformer_config['kernel_size'],
                bias=True
            )
        else:
            self.upsample = nn.Identity()
            
    def __init_embed(self, transformer_config, segment_config) -> None:
        self.embed_features = EmbeddedFeatures(segment_config['segment_length'], transformer_config['hidden_size'], \
            self.brush_embedding_flag, computing_device=self.computing_device) # transformer_config['position_embedding_dim']
    
    def __init_encoder(self, transformer_config: Dict[str, Any], segment_config) -> None:
        
        bert_config = BertConfig(
            hidden_size=self.hidden_size_transformer,
            intermediate_size=transformer_config["intermediate_size"],
            num_attention_heads=transformer_config["num_attention_heads"],
            hidden_dropout_prob=transformer_config["hidden_dropout_prob"],
            max_position_embeddings=segment_config["segment_length"] + 1,
            attention_probs_dropout_prob=transformer_config["attention_probs_dropout_prob"],
            hidden_act=transformer_config["hidden_activation"],
            num_hidden_layers=transformer_config["num_hidden_layers"],
        )
        self.encoder = BertModel(bert_config).encoder

      
    def __init_layer_drop(self, layer_drop_prob):
        if layer_drop_prob > 0:              
            layers = LayerDrop(p=layer_drop_prob)
        else:
            layers = nn.ModuleList([])

        for layer in self.encoder.layer:
            layers.append(layer)

        # if transformer_config["attention_option"] == "custom":
        #     transformer_config["max_position_embeddings"] = bert_config.max_position_embeddings
        #     for layer in layers:
        #         layer.attention.self = CustomBertSelfAttention(transformer_config)

    def __init_weight(self) -> None:
        nn.init.xavier_uniform_(
            self.upsample.weight, gain=nn.init.calculate_gain("relu")
        )


    def forward(
        self,
        inputs: torch.Tensor,
        brush_type = None,
        is_left_handed = 0
    ) -> torch.Tensor:  #is_left_handed=0
        # ! Upsample the inputs before feeding into the transformers
        
        embedded_input = self.embed_features(self.upsample(inputs.permute(0, 2, 1)).permute(0, 2, 1), brush_type, is_left_handed) #, is_left_handed

        outputs = self.encoder(hidden_states=embedded_input)
        batch = self.activation(self.fc1(outputs[0][:,0]))
        batch = self.fc2(batch)

        return batch


class LSTM(nn.Module):
    def __init__(self, input_size, eval_config, lstm_config=None, **kwargs):
        super().__init__()
        
        
        self.input_size = input_size
        if eval_config['use_euler_angles'] == True:
          self.input_size += 3
    
        self.num_classes = eval_config['num_dental_regs']
        if lstm_config is None:
            self.num_hidden_units = 20
            self.dropout_prob = 0.1
        else:
            self.num_hidden_units = lstm_config['num_hidden_units']
            self.dropout_prob = lstm_config['dropout_prob']

        # self.fc0 = nn.Linear(input_size, upscale_size)
        # self.fc0 = nn.Conv1d(12, hidden_size, kernel_size=7, padding=3)

        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.num_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.dp = nn.Dropout(p=self.dropout_prob)

        self.fc1 = nn.Linear(self.num_hidden_units*(1+self.encoder.bidirectional), self.num_classes)
        self.sigmoid = nn.Sigmoid()

        # self.__init_weight()

    def __init_weight(self):
        # ? Init LSTM cell
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ? Init fully connected layer
                std = math.sqrt(4.0 / m.weight.size(0))
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch, brush_type = None, is_left_handed = 0):
        
        self.encoder.flatten_parameters() # reduce GPU usage
        batch, hidden = self.encoder(batch)

        # batch_final = batch[:, -1, :]
        batch_final = self.dp(batch[:, -1, :])
        # batch_final = self.dp(batch)

        batch_final = batch_final.view(batch_final.shape[0], -1)
        
        batch_final = self.fc1(batch_final)

        return batch_final #self.sigmoid(batch_final)



class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes, network_params=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        if network_params is None:
            self.dropout_prob = 0.1
            self.num_kernels1 = 16
            self.num_kernels1 = 16
            self.kernel1_size = 5
            self.stride1_size = 5
            self.kernel2_size = 3
            self.stride2_size = 1
        else:
            self.dropout_prob = network_params['dropout_prob']
            self.num_kernels1 = network_params['num_kernels1']
            self.num_kernels2 = network_params['num_kernels2']
            self.kernel1_size = network_params['kernel1_size']
            self.stride1_size = network_params['stride1_size']
            self.kernel2_size = network_params['kernel2_size']
            self.stride2_size = network_params['stride2_size']


        self.cnn1 = nn.Conv1d(self.input_size, self.num_kernels1, kernel_size=self.kernel1_size, stride=self.stride1_size)
        self.cnn2 = nn.Conv1d(self.num_kernels1, self.num_kernels2, kernel_size=self.kernel2_size, stride=self.stride2_size)
        self.cnn3 = nn.Conv1d(self.num_kernels2, self.num_kernels2//2, kernel_size=self.kernel2_size, stride=self.stride2_size)
        # self.bn1 = nn.BatchNorm2d(first_CNN_numKern)
        self.activation = nn.ReLU()
        # self.activation = nn.ELU()
        #self.fcnorm0 = nn.LayerNorm(hidden_size)
        # self.mp = nn.MaxPool2d(kernel_size=self.maxPoolSize, stride=self.strideSize)

        self.dp = nn.Dropout(p=self.dropout_prob)

        # self.fc1 = nn.Linear(self.numHiddenUnits*(1+self.encoder.bidirectional), num_classes)
        self.fc1 = nn.Linear(16, self.num_classes)

                
        # self.sigmoid = nn.Sigmoid()

        # self.__init_weight()

    def __init_weight(self):
        # ? Init LSTM cell
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ? Init fully connected layer
                std = math.sqrt(4.0 / m.weight.size(0))
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch, brush_type = None, is_left_handed = 0):
    
        batch_proc = copy.copy(batch)
            
        batch_proc = self.cnn1(batch_proc.transpose(1,2))
        batch_proc = self.activation(batch_proc)

        batch_proc = self.cnn2(batch_proc)
        batch_proc = self.activation(batch_proc)
        
        batch_proc = self.cnn3(batch_proc)
        batch_proc = self.activation(batch_proc)

        # batch_proc = self.bn1(batch_proc)
        # batch_proc = self.mp(batch_proc)

        batch_proc = batch_proc.view(batch.shape[0], -1)

        batch_final = self.fc1(batch_proc)

        # batch_final = self.dp(self.activation(self.fc1(batch_final)))
        # batch_final = self.fc2(batch_final)

        # return self.sigmoid(batch_final)
        return batch_final



class CNN1D_LSTM(nn.Module):
    def __init__(self, input_size, num_classes, network_params=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        if network_params is None:
            self.num_hidden_units = 32
            self.dropout_prob = 0.1
            self.num_kernels = 16
            self.kernel_size = 5
            self.stride_size = 1
        else:
            self.num_hidden_units = network_params['num_hidden_units']
            self.dropout_prob = network_params['dropout_prob']
            self.num_kernels = network_params['num_kernels']
            self.kernel_size = network_params['kernel_size']
            self.stride_size = network_params['stride_size']

        self.cnn = nn.Conv1d(self.input_size, self.num_kernels, kernel_size=self.kernel_size, stride=self.stride_size)
        # self.bn1 = nn.BatchNorm2d(first_CNN_numKern)
        self.activation = nn.ReLU()
        # self.activation = nn.ELU()
        #self.fcnorm0 = nn.LayerNorm(hidden_size)
        # self.mp = nn.MaxPool2d(kernel_size=self.maxPoolSize, stride=self.strideSize)
        
        self.encoder = nn.LSTM(
            input_size= self.num_kernels, #180,
            hidden_size=self.num_hidden_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dp = nn.Dropout(p=self.dropout_prob)

        # self.fc1 = nn.Linear(self.numHiddenUnits*(1+self.encoder.bidirectional), num_classes)
        self.fc1 = nn.Linear(self.num_hidden_units*(1+self.encoder.bidirectional), self.num_hidden_units)
        self.fc2 = nn.Linear(self.num_hidden_units, self.num_classes)

                
        self.sigmoid = nn.Sigmoid()

        # self.__init_weight()

    def __init_weight(self):
        # ? Init LSTM cell
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ? Init fully connected layer
                std = math.sqrt(4.0 / m.weight.size(0))
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch, brush_type = None, is_left_handed = 0):
    
        batch_proc = copy.copy(batch)
        # batch_proc = batch_proc.view(batch.shape[0]*batch.shape[1], *batch.shape[2:])
        
        # batch_proc = self.cnn(batch_proc)
        batch_proc = self.cnn(batch_proc.transpose(1,2))


        # batch_proc = self.bn1(batch_proc)
        batch_proc = self.activation(batch_proc)

        # batch_proc = self.dp(self.cnn2(batch_proc))
        # batch_proc = self.mp(batch_proc)

        # batch_proc = batch_proc.view(*batch.shape[0:2], -1)
        batch_proc = batch_proc.transpose(1,2)
        
        batch_proc, hidden = self.encoder(batch_proc)
       
        # batch = batch[:, -1, :]

        batch_final = self.dp(batch_proc[:,-1,:])
        # batch_final = batch_proc[:,-1,:]

        # batch = self.activation(self.fcnorm1(self.fc1(batch)))
        # batch_final = self.fc1(batch_final)

        batch_final = self.dp(self.activation(self.fc1(batch_final)))
        batch_final = self.fc2(batch_final)
        
        # batch_final = self.fc2(self.activation(batch_final))


        # return self.sigmoid(batch_final)
        return batch_final


class MLP(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        input_size = 50
        hidden_size = 128
        self.fc0 = nn.Linear(6, input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_size)

        self.activation = nn.ReLU()
        # self.__init_weight()

        self.fcnorm3 = nn.BatchNorm1d(64)
        self.fcnorm4 = nn.BatchNorm1d(64)
        self.fcnorm5 = nn.BatchNorm1d(32)

        self.dp = nn.Dropout(p=0.1)

    def forward(self, batch):
        batch_size = len(batch)
        #batch = torch.median(batch, 1)
        #print(batch.shape)
        batch = batch.view(batch_size, -1)
        batch = self.activation(self.fc0(batch))
        batch = self.activation(self.fc1(batch))
        batch = self.activation(self.fc2(batch))
        batch = self.activation(self.fc3(batch))
        batch = self.activation(self.fc5(batch))
        batch = self.fc6(batch)

        return batch


class LayerDrop(nn.ModuleList):
    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super().__init__(modules=modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_probs = torch.empty(len(self)).uniform_()

        for i, m in enumerate(super().__iter__()):
            if not self.training or dropout_probs[i] > self.p:
                yield m


class BERTModelCostum(nn.Module):
    """
    ? Note: We are trying to diversify the model to see which one will
    ? work best
    """

    def __init__(self, transformer_config, eval_config, segment_config):
        super().__init__()
        # ? Remove all unrelated fields like regionLabels, etc.
        #data_config["keep_fields"] = resize_keep_field(data_config["keep_fields"])
        #config = self.__read_config(config_file, data_config)

        self.__init_embed(transformer_config, segment_config)
        self.__init_upsample(transformer_config, eval_config)
        self.__init_encoder(transformer_config, segment_config)
        self.__init_weight()
        self.fc = nn.Linear(transformer_config["hidden_size"], eval_config['num_dental_regs'])

    def __init_embed(self, transformer_config, segment_config) -> None:
        self.embed_features = EmbeddedFeatures(transformer_config['hidden_size'], segment_config['segment_length'], transformer_config['position_embedding_dim'])

    def __init_upsample(self, transformer_config, eval_config) -> None:
        # TODO: Should we add layernorm?
        # TODO: Check either summation or concat
        upsample_choice = transformer_config["upsample_unit"]
        if upsample_choice.lower() == "cnn":
            # ! This is equivalent to fully connected network
            self.upsample = nn.Conv1d(
                len(eval_config["data_fields"]),
                transformer_config["hidden_size"],
                kernel_size=transformer_config['kernel_size'],
                bias=True
            )
        else:
            self.upsample = nn.Identity()

    def __init_encoder(self, transformer_config: Dict[str, Any], segment_config) -> None:
        """
        If we run normal attention mechanism, we can just run from
        Huggingface so that we can sure about the result
        """
        bert_config = BertConfig(
            hidden_size=transformer_config["hidden_size"], #+transformer_config["position_embedding_dim"],
            intermediate_size=transformer_config["intermediate_size"],
            num_attention_heads=transformer_config["num_attention_heads"],
            hidden_dropout_prob=transformer_config["hidden_dropout_prob"],
            max_position_embeddings=segment_config["segment_length"],
            attention_probs_dropout_prob=transformer_config["attention_probs_dropout_prob"],
            hidden_act=transformer_config["hidden_activation"],
            num_hidden_layers=transformer_config["num_hidden_layers"],
        )
        self.encoder = BertModel(bert_config)

        if transformer_config["attention_option"] == "custom":
            if transformer_config["layer_drop_prob"] > 0:
                layers = LayerDrop(p=transformer_config["layerdrop"])
            else:
                layers = nn.ModuleList([])

            for layer in self.encoder.encoder.layer:
                layers.append(layer)

            transformer_config["max_position_embeddings"] = bert_config.max_position_embeddings
            for layer in layers:
                layer.attention.self = CustomBertSelfAttention(transformer_config)

    def __init_weight(self) -> None:
        nn.init.xavier_uniform_(
            self.upsample.weight, gain=nn.init.calculate_gain("relu")
        )


    def forward(
        self,
        inputs: torch.Tensor,
        patient_ids: torch.Tensor = None,
        session_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        # ! Upsample the inputs before feeding into the transformers
        """
        TODO: how to get context vector
        """

        embedded_input = self.embed_features(self.upsample(inputs.permute(0, 2, 1)).permute(0, 2, 1))

        outputs = self.encoder(inputs_embeds=embedded_input)
        batch = self.fc(outputs["pooler_output"])
        
        return batch
