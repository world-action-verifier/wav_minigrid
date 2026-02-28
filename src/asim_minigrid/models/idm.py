import torch
import torch.nn as nn
import torch.nn.functional as F

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

class DenseIDM(nn.Module):
    """IDM
    Dense Inverse Dynamics Model with full-frame CNN encoder.
    Uses full-frame CNN encoder with explicit geometry features (direction and position deltas).
    """
    def __init__(self, observation_shape=(7, 7), num_actions=7, embedding_dim=64):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        
        c, h, w = 3, observation_shape[0], observation_shape[1]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, h, w)
            cnn_out_dim = self.cnn(dummy_input).view(1, -1).shape[1]
            
        self.state_projector = nn.Linear(cnn_out_dim + 2, embedding_dim)

        input_dim = (embedding_dim * 2) + 2 + 2 
        hidden_dim = 128
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        
        self.head = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(0.2), 
            init_(nn.Linear(hidden_dim, num_actions))
        )

    def _encode_single_frame(self, frame, carried_col, carried_obj):
        """
        Encode single frame with carried object info.
        Args:
            frame: [B, H, W, 3]
            carried_col: [B, 1]
            carried_obj: [B, 1]
        Returns:
            [B, embedding_dim]
        """
        x = frame.permute(0, 3, 1, 2).float()
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        carried = torch.cat([carried_col, carried_obj], dim=-1).float()
        
        combined = torch.cat([features, carried], dim=1)
        embedding = F.relu(self.state_projector(combined))
        
        return embedding

    def _extract_direction(self, frame):
        """Extract agent direction (0-3)"""
        B, H, W, C = frame.shape
        directions = []
        agent_mask = (frame[..., 0] == 10)
        
        for i in range(B):
            idx = agent_mask[i].nonzero()
            if idx.shape[0] > 0:
                y, x = int(idx[0][0]), int(idx[0][1])
                d = int(frame[i, y, x, 2].item()) % 4
                directions.append(d)
            else:
                directions.append(0)
        return directions

    def _extract_position(self, frame):
        """Extract normalized agent coordinates (y, x)"""
        B, H, W, C = frame.shape
        coords = torch.zeros(B, 2, device=frame.device)
        agent_mask = (frame[..., 0] == 10)
        
        for i in range(B):
            idx = agent_mask[i].nonzero()
            if idx.shape[0] > 0:
                y, x = float(idx[0][0]), float(idx[0][1])
                coords[i, 0] = y / (H - 1)
                coords[i, 1] = x / (W - 1)
            else:
                coords[i] = 0.5
        return coords

    def _encode_direction_delta(self, curr_dirs, next_dirs, device):
        """Encode direction delta using sin/cos encoding"""
        curr = torch.tensor(curr_dirs, device=device, dtype=torch.float32)
        next_ = torch.tensor(next_dirs, device=device, dtype=torch.float32)
        
        delta = (next_ - curr + 4) % 4
        angle = delta * (2 * 3.14159 / 4)
        
        sin_ = torch.sin(angle).unsqueeze(1)
        cos_ = torch.cos(angle).unsqueeze(1)
        return torch.cat([sin_, cos_], dim=1)

    def forward(self, obs_inputs):
        """
        Args:
            obs_inputs: {
                'frame': [T, B, H, W, 3],
                'carried_col': [T, B, 1],
                'carried_obj': [T, B, 1]
            }
            T must be >= 2
        """
        frames = obs_inputs['frame']
        carried_col = obs_inputs['carried_col']
        carried_obj = obs_inputs['carried_obj']
        
        curr_frame, next_frame = frames[0], frames[1]
        
        curr_emb = self._encode_single_frame(curr_frame, carried_col[0], carried_obj[0])
        next_emb = self._encode_single_frame(next_frame, carried_col[1], carried_obj[1])
        
        curr_dir = self._extract_direction(curr_frame)
        next_dir = self._extract_direction(next_frame)
        
        curr_pos = self._extract_position(curr_frame)
        next_pos = self._extract_position(next_frame)
        
        dir_delta = self._encode_direction_delta(curr_dir, next_dir, curr_emb.device)
        pos_delta = next_pos - curr_pos
        
        combined = torch.cat([curr_emb, next_emb, dir_delta, pos_delta], dim=-1)
        return self.head(combined)

class MaskGenerator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1) 
        )
        
        with torch.no_grad():
            self.net[-1].bias[1] = 2.0
            self.net[-1].bias[0] = 0.0

    def forward(self, x, tau=1.0):
        logits = self.net(x)
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)
        return probs[:, 1:2, :, :], logits

class SparseIDM(nn.Module):
    def __init__(self, grid_h, grid_w, num_actions=8, embedding_dim=64):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        
        # 1. 轻量化 CNN (Bottleneck 结构)
        # 将输入编码为 32 维的语义特征
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # 加入 BN 稳定少样本训练
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1), # 1x1 卷积增加非线性
            nn.ReLU()
        )
        
        # 2. 改进的 Mask Generator (增加初始化偏置)
        self.mask_gen = MaskGenerator(in_channels=6)
        
        # 3. 状态投影层：现在的输入维度只是通道数 (32) + 携带物信息 (2)
        # 维度大幅缩小：从 64*H*W 降到了 32！
        self.state_projector = nn.Linear(32 + 2, embedding_dim)
        
        # 4. 动作预测 Head
        input_dim = (embedding_dim * 2) + 2 + 2 + 2 + 2
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def _encode_with_mask(self, frame, carried_col, carried_obj, mask):
        x = frame.permute(0, 3, 1, 2).float() # [B, 3, H, W]
        
        # A. 提取每个格点的特征
        features = self.feature_extractor(x)  # [B, 32, H, W]
        
        # B. 施加 Mask 并通过全局池化聚合
        # mask shape: [B, 1, H, W]
        # 我们只保留被 mask 选中的格子的特征，并把它们加起来
        # 这就是一种“硬注意力”机制
        masked_features = features * mask 
        
        # 关键改动：全局空间求和，把 [B, 32, H, W] 变成 [B, 32]
        # 这样模型就不用去管物体在哪，只需要管“选中的物体是什么”
        sum_features = torch.sum(masked_features, dim=(2, 3)) 
        
        # C. 拼接状态
        carried = torch.cat([carried_col, carried_obj], dim=-1).float()
        combined = torch.cat([sum_features, carried], dim=1)
        
        return F.relu(self.state_projector(combined))

    def _extract_direction(self, frame):
        """提取 agent 朝向 (0-3)"""
        B, H, W, C = frame.shape
        directions = []
        agent_mask = (frame[..., 0] == 10)
        
        for i in range(B):
            idx = agent_mask[i].nonzero()
            if idx.shape[0] > 0:
                y, x = int(idx[0][0]), int(idx[0][1])
                d = int(frame[i, y, x, 2].item()) % 4
                directions.append(d)
            else:
                directions.append(0)
        return directions

    def _extract_position(self, frame):
        """提取 agent 的归一化坐标 (y, x)"""
        B, H, W, C = frame.shape
        coords = torch.zeros(B, 2, device=frame.device)
        agent_mask = (frame[..., 0] == 10)
        
        for i in range(B):
            idx = agent_mask[i].nonzero()
            if idx.shape[0] > 0:
                y, x = float(idx[0][0]), float(idx[0][1])
                coords[i, 0] = y / (H - 1)
                coords[i, 1] = x / (W - 1)
            else:
                coords[i] = 0.5
        return coords

    def _encode_direction_delta(self, curr_dirs, next_dirs, device):
        """将方向差编码为 sin/cos"""
        curr = torch.tensor(curr_dirs, device=device, dtype=torch.float32)
        next_ = torch.tensor(next_dirs, device=device, dtype=torch.float32)
        
        delta = (next_ - curr + 4) % 4
        angle = delta * (2 * 3.14159 / 4)
        
        sin_ = torch.sin(angle).unsqueeze(1)
        cos_ = torch.cos(angle).unsqueeze(1)
        return torch.cat([sin_, cos_], dim=1)
    

    def forward(self, obs_inputs, tau=1.0):
        frames = obs_inputs['frame'] 
        curr_frame = frames[0]
        next_frame = frames[1]
        
        # 1. 准备 Mask Generator 的输入 (两帧拼接以捕捉 Toggle 变化)
        curr_p = curr_frame.permute(0, 3, 1, 2).float()
        next_p = next_frame.permute(0, 3, 1, 2).float()
        combined_for_mask = torch.cat([curr_p, next_p], dim=1) # [B, 6, H, W]
        
        # 2. 生成全局唯一的空间 Mask
        mask, mask_logits = self.mask_gen(combined_for_mask, tau=tau)
        
        # 3. 特征空间编码
        carried_col = obs_inputs['carried_col']
        carried_obj = obs_inputs['carried_obj']
        curr_carried = torch.cat([carried_col[0], carried_obj[0]], dim=-1).float()
        next_carried = torch.cat([carried_col[1], carried_obj[1]], dim=-1).float()
        curr_emb = self._encode_with_mask(curr_frame, carried_col[0], carried_obj[0], mask)
        next_emb = self._encode_with_mask(next_frame, carried_col[1], carried_obj[1], mask)
        
        # 4. 显式几何特征提取 (保持不变)
        curr_dir = self._extract_direction(curr_frame)
        next_dir = self._extract_direction(next_frame)
        curr_pos = self._extract_position(curr_frame)
        next_pos = self._extract_position(next_frame)
        
        dir_delta = self._encode_direction_delta(curr_dir, next_dir, curr_emb.device)
        pos_delta = next_pos - curr_pos
        
        # 5. Head
        combined = torch.cat(
            [curr_emb, next_emb, curr_carried, next_carried, dir_delta, pos_delta], 
            dim=-1
        )
        return self.head(combined), mask, mask_logits

class OracleSparseIDM(nn.Module):
    """
    Sparse Inverse Dynamics Model.
    Focuses on two cells: agent center and front position.
    Uses explicit direction encoding to help distinguish Turn Left/Right actions.
    """
    def __init__(self, num_actions=8):
        super().__init__()
        self.num_actions = num_actions
        
        input_dim = 16 + 2
        hidden_dim = 64

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.DIR_TO_VEC = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ]
        
    def _extract_two_cells(self, frames):
        """
        Extract features from agent center and front cells.
        Returns: features [B, 6], coords_info, directions [B]
        """
        B, H, W, C = frames.shape
        device = frames.device
        
        agent_mask = (frames[..., 0] == 10)
        
        frames_perm = frames.permute(0, 3, 1, 2).float()
        pad = 1
        frames_padded = F.pad(frames_perm, (pad, pad, pad, pad), mode='constant', value=0)
        frames_padded = frames_padded.permute(0, 2, 3, 1)
        
        extracted_features = []
        coords_info = []
        directions = []
        
        for i in range(B):
            idx = agent_mask[i].nonzero()
            if idx.shape[0] > 0:
                y, x = int(idx[0][0]), int(idx[0][1])
            else:
                y, x = H // 2, W // 2
            
            direction = int(frames[i, y, x, 2].item())
            directions.append(direction)
            
            dy, dx = self.DIR_TO_VEC[direction % 4]
            
            py, px = y + 1, x + 1
            fy, fx = py + dy, px + dx
            
            center_feat = frames_padded[i, py, px]
            front_feat = frames_padded[i, fy, fx]
            
            feat = torch.cat([center_feat, front_feat], dim=0)
            extracted_features.append(feat)
            
            coords_info.append((py, px, fy, fx))
            
        return torch.stack(extracted_features), coords_info, directions

    def _extract_from_coords(self, frames, coords_info):
        """
        Extract features from frames using given coordinates.
        Args:
            frames: [B, H, W, 3]
            coords_info: list of (py, px, fy, fx)
        Returns:
            features [B, 6], directions [B]
        """
        frames_perm = frames.permute(0, 3, 1, 2).float()
        pad = 1
        frames_padded = F.pad(frames_perm, (pad, pad, pad, pad), mode='constant', value=0)
        frames_padded = frames_padded.permute(0, 2, 3, 1)
        
        features = []
        directions = []
        
        H, W = frames.shape[1], frames.shape[2]
        
        for i, (py, px, fy, fx) in enumerate(coords_info):
            center_feat = frames_padded[i, py, px]
            front_feat = frames_padded[i, fy, fx]
            features.append(torch.cat([center_feat, front_feat], dim=0))
            
            y, x = py - 1, px - 1
            if 0 <= y < H and 0 <= x < W:
                direction = int(frames[i, y, x, 2].item())
            else:
                direction = 0
            directions.append(direction)
            
        return torch.stack(features), directions

    def forward(self, obs_inputs):
        frames = obs_inputs['frame']
        carried_col = obs_inputs['carried_col']
        carried_obj = obs_inputs['carried_obj']
        
        curr_frame = frames[0]
        next_frame = frames[1]
        
        curr_feats, coords_info, curr_directions = self._extract_two_cells(curr_frame)
        next_feats, next_directions = self._extract_from_coords(next_frame, coords_info)
        
        curr_carried = torch.cat([carried_col[0], carried_obj[0]], dim=-1).float()
        next_carried = torch.cat([carried_col[1], carried_obj[1]], dim=-1).float()
        
        curr_dir_tensor = torch.tensor(curr_directions, device=curr_feats.device, dtype=torch.float32)
        next_dir_tensor = torch.tensor(next_directions, device=curr_feats.device, dtype=torch.float32)
        
        dir_delta = (next_dir_tensor - curr_dir_tensor + 4) % 4
        dir_angle = dir_delta * (2 * 3.14159 / 4)
        dir_delta_sin = torch.sin(dir_angle).unsqueeze(1)
        dir_delta_cos = torch.cos(dir_angle).unsqueeze(1)
        dir_delta_encoded = torch.cat([dir_delta_sin, dir_delta_cos], dim=1)
        
        combined = torch.cat([curr_feats, next_feats, curr_carried, next_carried, dir_delta_encoded], dim=-1)
        
        return self.net(combined)