from torch.utils.data import Dataset, ConcatDataset
import torch
import numpy as np

class MiniGridDynamicsDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.states = torch.tensor(data['states']).squeeze(1).squeeze(1).float()
        self.next_states = torch.tensor(data['next_states']).squeeze(1).squeeze(1).float()
        self.actions = torch.tensor(data['actions']).long()
        self.carried = torch.tensor(data['carried']).long()
        self.next_carried = torch.tensor(data['next_carried']).long()
        
    def __len__(self): 
        return len(self.states)
    
    def __getitem__(self, idx):
        frames_seq = torch.stack([self.states[idx], self.next_states[idx]], dim=0)
        carried_col_seq = torch.stack([self.carried[idx, 0], self.next_carried[idx, 0]], dim=0).unsqueeze(1)
        carried_obj_seq = torch.stack([self.carried[idx, 1], self.next_carried[idx, 1]], dim=0).unsqueeze(1)
        
        return {
            'frame': frames_seq,            
            'carried_col': carried_col_seq, 
            'carried_obj': carried_obj_seq, 
            'action': self.actions[idx]     
        }

class MemoryDynamicsDataset(Dataset):
    """
    Wrapper for newly collected (s, a, s') data to match MiniGridDynamicsDataset format.
    """
    def __init__(self, data_list):
        """
        Args:
            data_list: list of dict, each dict contains 'state', 'action', 'next_state', 'carried', 'next_carried'
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        s_t = torch.tensor(item['state']).float()
        s_next = torch.tensor(item['next_state']).float()
        frames_seq = torch.stack([s_t, s_next], dim=0)
        
        c_t = item['carried']
        c_next = item['next_carried']
        
        if isinstance(c_t, np.ndarray):
            c_t_col = int(c_t[0])
            c_t_obj = int(c_t[1])
        else:
            c_t_col = int(c_t[0])
            c_t_obj = int(c_t[1])
            
        if isinstance(c_next, np.ndarray):
            c_next_col = int(c_next[0])
            c_next_obj = int(c_next[1])
        else:
            c_next_col = int(c_next[0])
            c_next_obj = int(c_next[1])
        
        carried_col_seq = torch.tensor([c_t_col, c_next_col]).long().unsqueeze(1)
        carried_obj_seq = torch.tensor([c_t_obj, c_next_obj]).long().unsqueeze(1)
        
        return {
            'frame': frames_seq,
            'carried_col': carried_col_seq,
            'carried_obj': carried_obj_seq,
            'action': torch.tensor(item['action']).long()
        }

class IndexedSubset(Dataset):
    """
    Lightweight wrapper for Subset that returns original indices for pseudo-labeling.
    """
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        sample = self.base_dataset[real_idx]
        sample = {k: v for k, v in sample.items()}
        sample['__index__'] = real_idx
        return sample

class PseudoLabeledSubset(Dataset):
    """
    Wrapper that uses pseudo action labels for specified indices while keeping original data.
    """
    def __init__(self, base_dataset, indices, pseudo_actions):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pseudo_actions = pseudo_actions or {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        sample = self.base_dataset[real_idx]
        sample = {k: v for k, v in sample.items()}
        if real_idx in self.pseudo_actions:
            pseudo = self.pseudo_actions[real_idx]
            sample['action'] = torch.tensor(pseudo, dtype=torch.long)
        return sample

class NormalizedDataset(Dataset):
    """
    Wrapper to normalize data format, ensuring carried_col and carried_obj have consistent shape [2, 1].
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if 'carried_col' in sample:
            carried_col = sample['carried_col']
            while len(carried_col.shape) > 2:
                carried_col = carried_col.squeeze(-1)
            if len(carried_col.shape) == 1:
                carried_col = carried_col.unsqueeze(1)
            sample['carried_col'] = carried_col
        
        if 'carried_obj' in sample:
            carried_obj = sample['carried_obj']
            while len(carried_obj.shape) > 2:
                carried_obj = carried_obj.squeeze(-1)
            if len(carried_obj.shape) == 1:
                carried_obj = carried_obj.unsqueeze(1)
            sample['carried_obj'] = carried_obj
        
        return sample