from turtle import position
import torch
import torch.nn as nn

from typing import Any, Dict


class EmbeddedFeatures(nn.Module):
    def __init__(self, segment_length, input_embedding_dim, brush_embedding_flag, computing_device='cpu'): 
        # position_embedding_dim, brush_embedding_dim, left_handedness_embedding_dim

        super().__init__()
        self.segment_length = segment_length + 1
        self.input_embedding_dim = input_embedding_dim
        # self.position_embedding_dim = position_embedding_dim
        # self.brush_embedding_dim = brush_embedding_dim
        self.brush_embedding_flag = brush_embedding_flag
        self.computing_device = computing_device
        
        self.hidden_size_transformer = self.input_embedding_dim #+ self.position_embedding_dim
        # if self.brush_embedding_flag == True:
        #     self.hidden_size_transformer += self.brush_embedding_dim

        # ! Follow Li et al. paper
        self.__init_embeddings()
        self.__init_cls_token()

        
        self.layer_norm_input = nn.LayerNorm(self.input_embedding_dim)
        self.layer_norm_embedding = nn.LayerNorm(self.hidden_size_transformer)

        # self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        # self.__init_weight()

    def __init_embeddings(self):
        self.position_embeddings = nn.Embedding(self.segment_length, self.input_embedding_dim, max_norm=1) #self.position_embedding_dim
        self.brush_type_embeddings = nn.Embedding(2, self.input_embedding_dim, max_norm=1) #8 #self.brush_embedding_dim
        self.left_handedness_embeddings = nn.Embedding(2, self.input_embedding_dim, max_norm=1) #self.left_handedness_embedding_dim

    def __init_cls_token(self) -> None:

        self.cls_token = nn.Parameter(torch.rand(self.input_embedding_dim))


    def __init_weight(self) -> None:
        
        nn.init.xavier_normal_(self.position_embeddings.weight)
        nn.init.xavier_normal_(self.brush_type_embeddings.weight)
        nn.init.xavier_normal_(self.left_handedness_embeddings.weight)
        nn.init.normal_(self.cls_token, mean=0, std=0.002)  
        # should check that this std is consistent with other ones above with xavier initialization

 

    def forward(
        self,
        input_segment: torch.Tensor,
        brush_type = None,
        is_left_handed = False
    ) -> torch.Tensor: 
        
        embeddings = self.layer_norm_input(input_segment)

        batch_size, _, _ = input_segment.shape

        batch_cls_token = self.cls_token.repeat(batch_size, 1, 1)
        input_segment = (
            torch.cat((batch_cls_token, input_segment), dim=1)
        )

        embeddings = self.__concat_position_embedding(input_segment)

        if self.brush_embedding_flag == True:
            embeddings = self.__concat_brush_type_embedding(embeddings, brush_type)

        embeddings = self.__concat_left_handed_embedding(embeddings, is_left_handed)

        # embeddings = self.layer_norm_embedding(embeddings)

        # embeddings = self.dropout(embeddings) 

        
        return embeddings

    def __concat_position_embedding(self,input_segment):

        batch_size, _, _ = input_segment.shape
        
        position_embeddings_curr = self.position_embeddings(torch.arange(0,self.segment_length).to(self.computing_device))[None,:,:].repeat(batch_size,1,1)
        
        input_embeddings = input_segment + position_embeddings_curr
        
        # input_embeddings = torch.cat((input_segment, position_embeddings_curr), dim=2)

        return input_embeddings


    def __concat_brush_type_embedding(self, input_segment, brush_type):
        
        brush_type_embeddings_curr = self.brush_type_embeddings(brush_type)[:,None,:].repeat(1,self.segment_length,1)
        
        # input_embeddings = torch.cat((input_segment, brush_type_embeddings_curr), dim=2)

        input_embeddings = input_segment + brush_type_embeddings_curr

        return input_embeddings


    def __concat_left_handed_embedding(self,input_segment, is_left_handed):

                
        left_handedness_embeddings = self.left_handedness_embeddings(is_left_handed)[:,None,:].repeat(1,self.segment_length,1)
        
        # input_embeddings = torch.cat((input_segment, left_handedness_embeddings), dim=2)
        
        input_embeddings = input_segment + left_handedness_embeddings


        return input_embeddings
