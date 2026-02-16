"""
MiniGrid physics oracle for evaluating model predictions.
"""

import numpy as np


class MiniGridPhysicsOracle:
    """Lightweight physics simulator for MiniGrid environment."""
    
    def __init__(self):
        self.OBJECT_TO_IDX = {
            'empty': 1, 'wall': 2, 'floor': 3, 'door': 4, 'key': 5, 
            'ball': 6, 'box': 7, 'goal': 8, 'lava': 9, 'agent': 10
        }
        self.COLOR_TO_IDX = {
            'red': 0, 'green': 1, 'blue': 2, 'yellow': 3, 'purple': 4, 'orange': 5
        }
        self.CARRIED_EMPTY_COL = 5
        self.GRID_EMPTY_COL = 0
        self.AGENT_IDX = 10
        self.DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def get_agent_pos_dir(self, grid):
        """Find agent position and direction."""
        rows, cols = np.where(grid[:, :, 0] == self.AGENT_IDX)
        if len(rows) == 0:
            return None, None
        r, c = rows[0], cols[0]
        direction = int(grid[r, c, 2])
        return (r, c), direction

    def get_fwd_pos(self, pos, direction, H, W):
        """Get forward position based on direction."""
        r, c = pos
        DIR_VEC_NP = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dr, dc = DIR_VEC_NP[direction]
        nr = r + dr
        nc = c + dc
        if 0 <= nr < H and 0 <= nc < W:
            return nr, nc
        return r, c

    def step(self, frame, carried_col, carried_obj, action):
        """
        Simulate one step in MiniGrid environment.
        
        Args:
            frame: Current frame [H, W, 3]
            carried_col: Carried object color
            carried_obj: Carried object type
            action: Action to take (0-6)
        
        Returns:
            next_frame: Next frame after action
            next_c_col: Next carried color
            next_c_obj: Next carried object
        """
        carried_col = int(carried_col)
        carried_obj = int(carried_obj)
        action = int(action)
        next_frame = frame.copy()
        next_c_col = carried_col
        next_c_obj = carried_obj
        
        H, W, _ = frame.shape
        pos, direction = self.get_agent_pos_dir(frame)
        
        if pos is None:
            return frame, carried_col, carried_obj
        
        r, c = pos
        f_r, f_c = self.get_fwd_pos(pos, direction, H, W)
        front_cell = next_frame[f_r, f_c]
        front_obj, front_col, front_state = front_cell
        
        # Action 0: Turn Left
        if action == 0:
            next_frame[r, c, 2] = (direction - 1) % 4
            
        # Action 1: Turn Right
        elif action == 1:
            next_frame[r, c, 2] = (direction + 1) % 4
            
        # Action 2: Move Forward
        elif action == 2:
            CAN_MOVE = (front_obj == self.OBJECT_TO_IDX['empty']) or \
                       (front_obj == self.OBJECT_TO_IDX['door'] and front_state == 0)
            if CAN_MOVE:
                next_frame[f_r, f_c] = next_frame[r, c].copy()
                next_frame[r, c] = np.array([self.OBJECT_TO_IDX['empty'], 0, 0])
        
        # Action 3: Pickup
        elif action == 3:
            if next_c_obj == self.OBJECT_TO_IDX['empty'] and front_obj not in [1, 2, 4]:
                next_c_obj = front_obj
                next_c_col = front_col
                next_frame[f_r, f_c] = np.array([self.OBJECT_TO_IDX['empty'], 0, 0])
                
        # Action 4: Drop
        elif action == 4:
            if next_c_obj != self.OBJECT_TO_IDX['empty'] and front_obj == self.OBJECT_TO_IDX['empty']:
                next_frame[f_r, f_c] = np.array([next_c_obj, next_c_col, 0])
                next_c_obj = self.OBJECT_TO_IDX['empty']
                next_c_col = 5
                
        # Action 5: Toggle
        elif action == 5:
            if front_obj == self.OBJECT_TO_IDX['box']:
                next_c_obj = self.OBJECT_TO_IDX['empty']
                next_c_col = 5
            elif front_obj in [self.OBJECT_TO_IDX['key'], self.OBJECT_TO_IDX['ball']]:
                if next_frame[f_r, f_c, 1] == self.COLOR_TO_IDX['red']:
                    next_frame[f_r, f_c, 1] = self.COLOR_TO_IDX['blue']
                elif next_frame[f_r, f_c, 1] == self.COLOR_TO_IDX['blue']:
                    next_frame[f_r, f_c, 1] = self.COLOR_TO_IDX['red']
                
        # Action 6: Done (Swap)
        elif action == 6:
            front_pickable = front_obj not in [
                self.OBJECT_TO_IDX['empty'], 
                self.OBJECT_TO_IDX['wall'], 
                self.OBJECT_TO_IDX['door']
            ]
            has_carry = next_c_obj != self.OBJECT_TO_IDX['empty']
            
            if front_pickable and has_carry:
                obj_to_grid = next_c_obj
                col_to_grid = next_c_col
                
                if obj_to_grid == self.OBJECT_TO_IDX['empty']:
                    col_to_grid = self.GRID_EMPTY_COL
                
                obj_to_hand = front_obj
                col_to_hand = front_col
                
                if obj_to_hand == self.OBJECT_TO_IDX['empty']:
                    col_to_hand = self.CARRIED_EMPTY_COL
                
                next_frame[f_r, f_c] = np.array([obj_to_grid, col_to_grid, 0])
                next_c_obj = obj_to_hand
                next_c_col = col_to_hand

        return next_frame, next_c_col, next_c_obj

