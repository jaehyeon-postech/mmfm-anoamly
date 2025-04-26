import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, q_embed_dim, kv_embed_dim, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # 定义线性变换来获取 Q, K, V
        self.query_proj = nn.Linear(q_embed_dim, embed_dim)
        self.key_proj = nn.Linear(kv_embed_dim, embed_dim)
        
    def forward(self, queries, keys, values):
        """
        queries: [batch_size, q_seq_len, embed_dim]
        keys: [batch_size, kv_seq_len, embed_dim]
        values: [batch_size, kv_seq_len, embed_dim]
        """
        # 线性变换获得 Q, K, V
        Q = self.query_proj(queries)  # [batch_size, q_seq_len, embed_dim]
        K = self.key_proj(keys)       # [batch_size, kv_seq_len, embed_dim]
        V = values                  # [batch_size, kv_seq_len, kv_embed_dim]
        
        # 计算注意力得分: QK^T / sqrt(d_k)
        d_k = K.shape[-1]  # embed_dim
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 使用 softmax 对注意力得分进行归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, q_seq_len, kv_seq_len]
        
        # 加权求和得到最终的注意力输出
        attention_output = torch.matmul(attention_weights, V)  # [batch_size, q_seq_len, kv_embed_dim]
        
        return attention_output, attention_weights


class AnomalyOV(nn.Module):
    def __init__(self, embedding_dim=512, class_dim=512, kernal_size=3, attention_dim=512, num_pooling_tokens=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.class_dim = class_dim
        self.kernal_size = kernal_size
        self.attention_dim = attention_dim
        self.num_pooling_tokens = num_pooling_tokens

        self.num_patches = 729
        self.ov_token_hidden_size = 1152

        self._build_extra()
        self._build_q_former()
     

    def _build_extra(self):
        # add two learnable parameters with shape [1, 1024, 1, 1]
        self.positive_embedding = nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
        self.negative_embedding = nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)

        # define a mlp layer to convert (N, num_patches, self.ov_token_hidden_size) to (N, 1, self.ov_token_hidden_size)
        self.pooling_layer = nn.Sequential(
                                nn.Linear(self.num_patches, 1),
                            )
        
        # define a linear layer to convert (N, self.ov_token_hidden_size) to (N, self.class_dim)
        self.class_predictor = nn.Sequential(
                                        nn.Linear(self.ov_token_hidden_size, self.class_dim),
                                        nn.GELU(),
                                        nn.Linear(self.class_dim, self.class_dim),
                                    )
        
        # linear or mlp ????
        self.siglip_adaptors = nn.ModuleList([
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
           nn.Linear(self.ov_token_hidden_size, self.embedding_dim),
        ])

        self.prompt_adaptors = nn.ModuleList([
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim + self.class_dim, self.embedding_dim),
        ])

        self.final_predictor = nn.Sequential(
                                    nn.Linear(self.ov_token_hidden_size, 512),
                                    nn.GELU(),
                                    nn.Linear(512, 1),
                                )
        
        self.sigmoid = nn.Sigmoid()
        
        self.global_pooling = nn.AdaptiveAvgPool2d((self.num_pooling_tokens, self.num_pooling_tokens))


    def _build_q_former(self):
        self.q_former = SingleHeadAttention(self.ov_token_hidden_size, self.ov_token_hidden_size, self.attention_dim)

    def freeze_layers(self):
        # freeze all the parameters
        for param in self.parameters():
            param.requires_grad = False

    def load_zero_shot_weights(self, path='./pretrained_expert_7b.pth'):
        checkpoint = torch.load(path)
        self_state_dict = self.state_dict()
        matched_names = []
        for key in self_state_dict.keys():
            if key in checkpoint.keys():
                matched_names.append(key)

        print(f"Matched names: {matched_names}")
        print('Loaded number of keys:', len(matched_names))
        self.load_state_dict(checkpoint, strict=False)

    def get_anomaly_map(self, sig_multi_level_features, ov_base_features): 
        # sig_multi_level_features is a list of tensors, each tensor is [batch_size * num_patches, 729, self.siglip_hidden_dim]
        # ov_base_features is a tensor with size of [batch_size * num_patches, 729, self.ov_token_hidden_size]

        total_size = ov_base_features.shape[0]
        # Get the embeddings of the image
        positive_embedding = self.positive_embedding.repeat(total_size, 1) # [batch_size * num_patches, self.embedding_dim]
        negative_embedding = self.negative_embedding.repeat(total_size, 1) # [batch_size * num_patches, self.embedding_dim]

        # get the class prediction
        ov_base_features = self.pooling_layer(ov_base_features.transpose(1, 2)).squeeze(-1) # [batch_size * num_patches, self.ov_token_hidden_size]
        class_prediction = self.class_predictor(ov_base_features) # [batch_size * num_patches, self.class_dim]

        # get the embeddings for the prompt
        prompt_embeddings_list = []
        for i in range(4):
            positive_prompt_embedding = self.prompt_adaptors[i](torch.cat([positive_embedding, class_prediction], dim=-1)) # [batch_size * num_patches, self.embedding_dim]
            positive_prompt_embedding = positive_prompt_embedding / positive_prompt_embedding.norm(dim=-1, keepdim=True) # [batch_size * num_patches, self.embedding_dim]
            negative_prompt_embedding = self.prompt_adaptors[i](torch.cat([negative_embedding, class_prediction], dim=-1)) # [batch_size * num_patches, self.embedding_dim]
            negative_prompt_embedding = negative_prompt_embedding / negative_prompt_embedding.norm(dim=-1, keepdim=True) # [batch_size * num_patches, self.embedding_dim]
            prompt_embedding = torch.cat([positive_prompt_embedding.unsqueeze(-1), negative_prompt_embedding.unsqueeze(-1)], dim=-1) # [batch_size * num_patches, self.embedding_dim, 2]
            prompt_embeddings_list.append(prompt_embedding)

        # get the embeddings for the sig features
        sig_embeddings_list = []
        for i in range(4):
            sig_embedding = self.siglip_adaptors[i](sig_multi_level_features[i]) # [batch_size * num_patches, 729, self.embedding_dim]
            sig_embedding = sig_embedding / sig_embedding.norm(dim=-1, keepdim=True)
            sig_embeddings_list.append(sig_embedding)

        # get the anomaly maps
        anomaly_maps = []
        patch_significances = []
        for i in range(4):
            anomaly_map = torch.matmul(100.0 * sig_embeddings_list[i], prompt_embeddings_list[i]) # [batch_size * num_patches, 729, 2]
            anomaly_map = anomaly_map.permute(0, 2, 1).view(total_size, 2, 27, 27) # [batch_size * num_patches, 2, 27, 27]
            anomaly_map = torch.softmax(anomaly_map, dim=1) # [batch_size * num_patches, 2, 27, 27]
            anomaly_maps.append(anomaly_map[:, 1:, :, :]) # [batch_size * num_patches, 1, 27, 27]
            patch_significance = anomaly_map[:, 1:, :, :].mean(dim=(1, 2, 3)) # [batch_size * num_patches,]
            patch_significances.append(patch_significance)

        anomaly_map = sum(anomaly_maps) / 4 # [batch_size * num_patches, 1, 27, 27]
        patch_significances = sum(patch_significances) / 4 # [batch_size * num_patches,]

        return anomaly_map, patch_significances
    
    def forward(self, ov_image_features, sig_multi_level_features, split_sizes):
        # ov_image_features is a tensor with size of [batch_size * num_patches, 729, self.ov_token_hidden_size]
        # sig_multi_level_features is a list of tensors, each tensor is [batch_size * num_patches, 729, self.siglip_hidden_dim]
        # split_sizes is a list of integers, each integer is the number of patches in the corresponding image

        batch_size = len(split_sizes)
        ov_image_features_split = torch.split(ov_image_features, split_sizes) # list of tensors, each tensor is [1+num_patches, 729, self.ov_token_hidden_size]
        ov_base_image_feature_list = []
        for i, image_feature in enumerate(ov_image_features_split):
            base_image_feature = image_feature[0].unsqueeze(0) # [1, 729, self.ov_token_hidden_size]
            ov_base_image_feature_list.append(base_image_feature.repeat(split_sizes[i], 1, 1)) # [num_patches, 729, self.ov_token_hidden_size]
        
        ov_base_image_features = torch.cat(ov_base_image_feature_list, dim=0) # [batch_size * num_patches, 729, self.ov_token_hidden_size]
 
        # get the anomaly map
        anomaly_map, out_patch_significance = self.get_anomaly_map(sig_multi_level_features, ov_base_image_features) # [batch_size * num_patches, 1, 27, 27]

        # get the attention output
        ov_image_features_reshaped = ov_image_features.permute(0, 2, 1).view(ov_image_features.shape[0], self.ov_token_hidden_size, 27, 27) # [batch_size * num_patches, self.ov_token_hidden_size, 27, 27]
        scaled_final_features = ov_image_features_reshaped * anomaly_map  # [batch_size * num_patches, self.ov_token_hidden_size, 27, 27]
        in_patch_significance = self.global_pooling(anomaly_map) # [batch_size * num_patches, 1, 2, 2]
        sig_tokens = self.global_pooling(scaled_final_features) / in_patch_significance # [batch_size * num_patches, self.ov_token_hidden_size, 2, 2]
        in_patch_significance = in_patch_significance.permute(0, 2, 3, 1).view(scaled_final_features.shape[0],  self.num_pooling_tokens * self.num_pooling_tokens, 1) # [batch_size * num_patches, 4, 1]

        # get the global significance
        global_significance = in_patch_significance * out_patch_significance.unsqueeze(-1).unsqueeze(-1) # [batch_size * num_patches, 4, 1]

        # convert to [batch_size * num_patches, 4, self.siglip_hidden_dim]
        sig_tokens = sig_tokens.permute(0, 2, 3, 1).view(scaled_final_features.shape[0], self.num_pooling_tokens * self.num_pooling_tokens, self.ov_token_hidden_size) # [batch_size * num_patches, 4, self.siglip_hidden_dim]
        # get the attention output
        major_attention_output, _ = self.q_former(sig_tokens, ov_image_features, ov_image_features) # [batch_size * num_patches, 4, self.ov_token_hidden_size]
        
        # split the attention output
        global_significance_split = torch.split(global_significance, split_sizes) # list of tensors, each tensor is [1+num_patches, 4, 1]
        attention_output_split = torch.split(major_attention_output, split_sizes) # list of tensors, each tensor is [1+num_patches, 4, self.ov_token_hidden_size]
        attention_output_list = []
        for i in range(batch_size):
            attention_output = (attention_output_split[i] * global_significance_split[i]).mean(dim=(0, 1)) / global_significance_split[i].mean(dim=(0, 1)) # [self.ov_token_hidden_size,]
            attention_output_list.append(attention_output.unsqueeze(0)) # [1, self.ov_token_hidden_size]

        final_attention_output = torch.cat(attention_output_list, dim=0).unsqueeze(1) # [batch_size, 1, self.ov_token_hidden_size]
        # get the final prediction
        final_prediction = self.final_predictor(final_attention_output.squeeze(1)) # [batch_size, 1]
        final_prediction = self.sigmoid(final_prediction) # [batch_size, 1]

        return major_attention_output, final_attention_output, final_prediction
    
   