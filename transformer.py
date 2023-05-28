import tensorflow as tf
import math as m
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
import numpy as np
import pandas as pd

class NE(tf.keras.Model):
    def __init__(
        self,
        feature_names: list,
        X: np.array,
        y: np.array = None,
        task: str = None,
        emb_dim: int = 32,
        n_bins: int = 10,
        sigma: float = 1,
        tree_params = {},
    ):
        super(NE, self).__init__()
        
        self.num_features = len(feature_names)
        self.features = feature_names
        self.emb_dim = emb_dim
        
        w_init = tf.random_normal_initializer()
        self.linear_w = tf.Variable(
                initial_value=w_init(
                    shape=(self.num_features, 1, self.emb_dim), dtype='float32' 
                ), trainable=True)
        self.linear_b = tf.Variable(
                w_init(
                    shape=(self.num_features, 1), dtype='float32'
                ), trainable=True)
    
    
    def embed_column(self, f, data):
        emb = self.linear_layers[f](self.embedding_layers[f](data))
        return emb
   
    def call(self, x):
        embs = tf.einsum('f n e, b f -> bfe', self.linear_w, x)
        embs = tf.nn.relu(embs + self.linear_b)
        return embs

class CE(tf.keras.Model):
    def __init__(
        self,
        feature_names: list,
        X: np.array,
        emb_dim: int = 32,
    ):
        super(CE, self).__init__()
        self.features = feature_names
        self.emb_dim = emb_dim
        
        self.category_prep_layers = {}
        self.emb_layers = {}
        for i, c in enumerate(self.features):
            lookup = tf.keras.layers.StringLookup(vocabulary=list(np.unique(X[:, i])))
            emb = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=self.emb_dim)

            self.category_prep_layers[c] = lookup
            self.emb_layers[c] = emb
    
    def embed_column(self, f, data):
        return self.emb_layers[f](self.category_prep_layers[f](data))

    def call(self, x):
        emb_columns = []
        for i, f in enumerate(self.features):
            emb_columns.append(self.embed_column(f, x[:, i]))
        
        embs = tf.stack(emb_columns, axis=1)
        return embs


class TransformerBlock(Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        att_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable: bool = False
    ):
        super(TransformerBlock, self).__init__()
        self.explainable = explainable
        self.att = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=att_dropout
        )
        self.skip1 = Add()
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation=gelu),
                Dropout(ff_dropout),
                Dense(embed_dim),
            ]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.skip2 = Add()

    def call(self, inputs, mask):
        inputs = self.layernorm1(inputs)
        if self.explainable:
            attention_output, att_weights = self.att( inputs, inputs, return_attention_scores=True, attention_mask=mask)
        else:
            attention_output = self.att(inputs, inputs,attention_mask=mask)
        attention_output = self.skip1([inputs, attention_output])   
        feedforward_output = self.ffn(attention_output) 
        transformer_output = self.skip2([feedforward_output, attention_output])
        transformer_output = self.layernorm2(transformer_output)
        
        if self.explainable:
            return transformer_output, att_weights
        else:
            return transformer_output

class TransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        numerical_data: np.array,
        categorical_data: np.array,
        y: np.array = None,
        task: str = None,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_bins: int = None,
        ple_tree_params: dict = {},
        explainable=False,
    ):
        super(TransformerEncoder, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.embedding_dim = embedding_dim
        self.explainable = explainable
        self.depth = depth
        self.heads = heads
            
        if len(self.numerical) > 0:
            self.numerical_embeddings = NE(
                feature_names=self.numerical, 
                X=numerical_data, 
                y=y,
                task=task,
                emb_dim=embedding_dim, 
                n_bins=numerical_bins,
                tree_params=ple_tree_params
            )
        if len(self.categorical) > 0:
            self.categorical_embeddings = CE(
                feature_names=self.categorical,
                X=categorical_data,
                emb_dim =embedding_dim
            )

        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=self.explainable,
                )
            )
        self.flatten_transformer_output = Flatten()

        w_init = tf.random_normal_initializer()
        self.cls_weights = tf.Variable(
            initial_value=w_init(shape=(1, embedding_dim), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        mask = inputs==0
        cls_tokens = tf.repeat(self.cls_weights, repeats=tf.shape(inputs[self.numerical[0]])[0], axis=0)
        cls_tokens = tf.expand_dims(cls_tokens, axis=1)
        transformer_inputs = [cls_tokens]
    
        if len(self.categorical) > 0:
            cat_input = []
            for c in self.categorical:
                cat_input.append(inputs[c])
            
            cat_input = tf.stack(cat_input, axis=1)[:, :, 0]
            cat_embs = self.categorical_embeddings(cat_input)
            transformer_inputs += [cat_embs]
        
        if len(self.numerical) > 0:
            num_input = []
            for n in self.numerical:
                num_input.append(inputs[n])
            num_input = tf.stack(num_input, axis=1)[:, :, 0]
            num_embs = self.numerical_embeddings(num_input)
            transformer_inputs += [num_embs]
        
        transformer_inputs = tf.concat(transformer_inputs, axis=1)
        importances = []
        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, att_weights = transformer(transformer_inputs,mask)
                importances.append(tf.reduce_sum(att_weights[:, :, 0, :], axis=1))
            else:
                transformer_inputs = transformer(transformer_inputs,mask)

        if self.explainable:
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs


class Transformer(tf.keras.Model):
    def __init__(
        self,
        out_dim: int,
        out_activation: str,
        categorical_features: list = None,
        numerical_features: list = None,
        categorical_lookup: dict = None,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embeddings: dict = None,
        explainable=False,
        encoder=None,
    ):
        super(Transformer, self).__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = TransformerEncoder(
                categorical_features = categorical_features,
                numerical_features = numerical_features,
                categorical_lookup = categorical_lookup,
                embedding_dim = embedding_dim,
                depth = depth,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                numerical_embeddings = numerical_embeddings,
                explainable = explainable,
            )

        self.ln = tf.keras.layers.LayerNormalization()
        self.final_ff = Dense(embedding_dim//2, activation='relu')
        self.output_layer = Dense(out_dim, activation=out_activation)
    
    def call(self, inputs):
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)
        else:
            x = self.encoder(inputs)

        layer_norm_cls = self.ln(x[:, 0, :])
        layer_norm_cls = self.final_ff(layer_norm_cls)
        output = self.output_layer(layer_norm_cls)

        if self.encoder.explainable:
            return {"output": output, "importances": expl}
        else:
            return output