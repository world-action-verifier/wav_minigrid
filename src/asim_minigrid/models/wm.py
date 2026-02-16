import torch
import torch.nn as nn
import torch.nn.functional as F

def init(module, weight_init, bias_init, gain=1):
    if hasattr(module, 'weight') and module.weight is not None:
        weight_init(module.weight.data, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices.squeeze(1)
        
            
class WorldModel(nn.Module):
    """
    Uses supervised VQ with action supervision, spatial preserving, and object consistency.
    """
    def __init__(self, observation_shape, num_actions=7, latent_dim=32):
        super().__init__()
        self.H, self.W, self.C = observation_shape
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.feature_dim = 64
        
        self.NUM_OBJ_CLASSES = 20
        self.NUM_COL_CLASSES = 10
        self.NUM_STATE_CLASSES = 10

        init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        init_final = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.emb_dim = 8
        self.obj_embedding = nn.Embedding(self.NUM_OBJ_CLASSES, self.emb_dim)
        self.col_embedding = nn.Embedding(self.NUM_COL_CLASSES, self.emb_dim)
        self.state_embedding = nn.Embedding(self.NUM_STATE_CLASSES, self.emb_dim)
        
        cnn_in_channels = (3 * self.emb_dim) + 2 
        
        self.encoder_cnn = nn.Sequential(
            init_relu(nn.Conv2d(cnn_in_channels, 32, kernel_size=3, stride=1, padding=1)), 
            nn.ReLU(),
            init_relu(nn.Conv2d(32, self.feature_dim, kernel_size=3, stride=1, padding=1)), 
            nn.ReLU()
        )

        self.scalar_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.fusion_conv = init_relu(nn.Conv2d(self.feature_dim + 16, self.feature_dim, kernel_size=1))

        self.vq_encoder_net = nn.Sequential(
            init_relu(nn.Conv2d(self.feature_dim * 2, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), 
            Flatten(),
            nn.Linear(64, self.latent_dim)
        )
        
        self.vq_layer = VectorQuantizer(num_embeddings=num_actions, embedding_dim=latent_dim, commitment_cost=0.25)

        self.prior_net = nn.Sequential(
            init_relu(nn.Conv2d(self.feature_dim, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            init_relu(nn.Linear(64, 64)),
            nn.ReLU(),
            init_final(nn.Linear(64, num_actions))
        )

        self.film_gen = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim * 2)
        )
        
        with torch.no_grad():
            self.film_gen[-1].weight.fill_(0)
            self.film_gen[-1].bias.fill_(0)
        
        self.dyn_conv = nn.Sequential(
            init_relu(nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init_final(nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1))
        )

        self.decoder_shared = nn.Sequential(
            init_relu(nn.Conv2d(self.feature_dim, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Dropout2d(p=0.1) 
        )
        self.head_obj_cls = init_final(nn.Conv2d(32, self.NUM_OBJ_CLASSES, kernel_size=1))
        self.head_col_cls = init_final(nn.Conv2d(32, self.NUM_COL_CLASSES, kernel_size=1))
        self.head_state_cls = init_final(nn.Conv2d(32, self.NUM_STATE_CLASSES, kernel_size=1))
        self.carried_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            init_final(nn.Linear(self.feature_dim, 2))
        )

        self.finetune_action_linear = nn.Linear(self.num_actions, self.latent_dim)
        self.finetune_adapter = nn.Sequential(
            nn.Linear(64+32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )
        init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.finetune_adapter.apply(init_relu)

    def _add_coord_channels(self, x):
        B, _, H, W = x.shape
        device = x.device
        xx_channel = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W).float() / (W - 1)
        yy_channel = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W).float() / (H - 1)
        return torch.cat([x, xx_channel, yy_channel], dim=1)

    def _extract_spatial_features(self, obs_inputs):
        """Extract spatial features preserving spatial structure [B, C, H, W]"""
        frames = obs_inputs['frame']
        if frames.dim() == 5: frames = frames[0]
        B, H, W, _ = frames.shape
        
        obj_idx = frames[..., 0].long().clamp(0, self.NUM_OBJ_CLASSES-1)
        col_idx = frames[..., 1].long().clamp(0, self.NUM_COL_CLASSES-1)
        state_idx = frames[..., 2].long().clamp(0, self.NUM_STATE_CLASSES-1)
        
        emb_feat = torch.cat([
            self.obj_embedding(obj_idx),
            self.col_embedding(col_idx),
            self.state_embedding(state_idx)
        ], dim=-1).permute(0, 3, 1, 2).contiguous()
        
        emb_feat = self._add_coord_channels(emb_feat)
        img_feat = self.encoder_cnn(emb_feat)
        
        c_col = obs_inputs['carried_col']
        c_obj = obs_inputs['carried_obj']
        if c_col.dim() == 3: c_col = c_col[0]
        if c_obj.dim() == 3: c_obj = c_obj[0]
        c_col = c_col.squeeze(-1)
        c_obj = c_obj.squeeze(-1)
        
        scalar_input = torch.stack([c_col, c_obj], dim=-1).float()
        scalar_feat = self.scalar_embed(scalar_input)
        
        scalar_map = scalar_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        h = self.fusion_conv(torch.cat([img_feat, scalar_map], dim=1))
        return h

    def forward(self, obs_inputs, next_obs_inputs=None, mode='posterior', gt_actions=None):
        h_t_map = self._extract_spatial_features(obs_inputs)
        outputs = {}
        z = None

        if mode == 'posterior' and next_obs_inputs is not None:
            h_next_map = self._extract_spatial_features(next_obs_inputs)
            
            cat_map = torch.cat([h_t_map, h_next_map], dim=1)
            z_e = self.vq_encoder_net(cat_map)
            if gt_actions is not None:
                indices = gt_actions.long()
                z_q = self.vq_layer._embedding(indices)
                vq_loss = torch.tensor(0.0)
            else:
                vq_loss, z_q, indices = self.vq_layer(z_e)
            
            prior_logits = self.prior_net(h_t_map)
            
            outputs.update({
                'vq_loss': vq_loss, 
                'prior_logits': prior_logits, 
                'target_indices': indices,
                'gt_actions': gt_actions
            })
            z = z_q

        elif mode == 'inference':
            prior_logits = self.prior_net(h_t_map)
            
            temperature = 0.5
            prior_logits = prior_logits / temperature
            probs = F.softmax(prior_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            indices = dist.sample() 
            
            z = self.vq_layer._embedding(indices)
            outputs['prior_logits'] = prior_logits

        elif mode == 'predict_with_action':
            if gt_actions is None:
                raise ValueError("gt_actions is required for predict_with_action mode")
            h_t_global = F.adaptive_avg_pool2d(h_t_map, 1).flatten(1)
            one_hot = F.one_hot(gt_actions.long(), num_classes=self.num_actions).float()
            a_emb = self.finetune_action_linear(one_hot)
            cat_feat = torch.cat([h_t_global, a_emb], dim=-1)
            z = self.finetune_adapter(cat_feat)
            outputs['z'] = z
            
        film_params = self.film_gen(z)
        gamma, beta = torch.split(film_params, self.feature_dim, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        h_mid = self.dyn_conv(h_t_map)
        h_delta = (1 + gamma) * h_mid + beta
        h_next_pred = h_t_map + h_delta

        x_shared = self.decoder_shared(h_next_pred)
        outputs.update({
            'logits_obj': self.head_obj_cls(x_shared),
            'logits_col': self.head_col_cls(x_shared),
            'logits_state': self.head_state_cls(x_shared),
            'carried_col': self.carried_head(h_next_pred)[:, 0:1],
            'carried_obj': self.carried_head(h_next_pred)[:, 1:2]
        })
        return outputs