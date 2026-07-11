import numpy as np
from gym_minigrid.minigrid import (
    MiniGridEnv,
    WorldObj,
    COLORS,
    fill_coords,
    point_in_rect,
    point_in_circle,
    Grid,
    TILE_PIXELS,
)
from gym_minigrid.rendering import highlight_img

# Off-white grey for outer walls in InteractiveMiniGridEnv render
WALL_COLOR = np.array([215, 215, 215], dtype=np.uint8)

# COLOR_CYCLE = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
COLOR_CYCLE = ['red', 'blue']
COLOR_NOISE = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

class IKey(WorldObj):
    def __init__(self, color='blue'):
        super(IKey, self).__init__('key', color)

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        """
        Color change logic: cycle through colors in COLOR_CYCLE order
        """
        if self.color in COLOR_CYCLE:
            current_idx = COLOR_CYCLE.index(self.color)
            next_idx = (current_idx + 1) % len(COLOR_CYCLE)
            self.color = COLOR_CYCLE[next_idx]
        else:
            self.color = COLOR_CYCLE[0]
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class IBall(WorldObj):
    def __init__(self, color='blue'):
        super(IBall, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        """
        Color change logic: cycle through colors in COLOR_CYCLE order
        """
        if self.color in COLOR_CYCLE:
            current_idx = COLOR_CYCLE.index(self.color)
            next_idx = (current_idx + 1) % len(COLOR_CYCLE)
            self.color = COLOR_CYCLE[next_idx]
        else:
            self.color = COLOR_CYCLE[0]
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class IBox(WorldObj):
    def __init__(self, color, contains=None):
        super(IBox, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        """
        Implement swap logic:
        1. Temporarily store what the Agent is holding (env.carrying)
        2. Replace what the Agent is holding with what's in the Box (self.contains)
        3. Replace what's in the Box with what was temporarily stored
        """
        obj_in_hand = env.carrying
        obj_in_box = self.contains

        env.carrying = obj_in_box
        self.contains = obj_in_hand

        if env.carrying is not None:
            env.carrying.cur_pos = np.array([-1, -1])

        return True
    
    # Color Change (Use this toggle logic when collecting random playing data)
    # def toggle(self, env, pos):
    #     """
    #     Color change logic: cycle through colors in COLOR_CYCLE order
    #     """
    #     if self.color in COLOR_CYCLE:
    #         current_idx = COLOR_CYCLE.index(self.color)
    #         next_idx = (current_idx + 1) % len(COLOR_CYCLE)
    #         self.color = COLOR_CYCLE[next_idx]
    #     else:
    #         self.color = COLOR_CYCLE[0]
    #     return True


class NoiseFloor(WorldObj):
    """Background noise object: agent can overlap, not pickup-able, renders as colored tile."""
    def __init__(self, color='blue'):
        # Use 'floor' type so MiniGrid encoding does not raise
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c)


class InteractiveMiniGridEnv(MiniGridEnv):
    """
    This is an enhanced MiniGrid environment base class.
    It redefines the originally unused 'done' action as 'switch' (swap objects in hand and in front).
    """

    @staticmethod
    def _is_wall_cell(cell):
        return cell is not None and cell.type == 'wall'

    def _get_highlight_mask(self, highlight=True):
        if not highlight:
            return np.zeros((self.width, self.height), dtype=np.bool)

        _, vis_mask = self.gen_obs_grid()
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        highlight_mask = np.zeros((self.width, self.height), dtype=np.bool)
        for vis_j in range(self.agent_view_size):
            for vis_i in range(self.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if 0 <= abs_i < self.width and 0 <= abs_j < self.height:
                    highlight_mask[abs_i, abs_j] = True
        return highlight_mask

    @staticmethod
    def _draw_grid_lines(tile):
        fill_coords(tile, point_in_rect(0, 0.031, 0, 1), WALL_COLOR)
        fill_coords(tile, point_in_rect(0, 1, 0, 0.031), WALL_COLOR)

    def _render_tile_white_floor(self, cell, agent_dir=None, highlight=False, tile_size=TILE_PIXELS):
        """Render interior tiles on a white background with WALL_COLOR grid lines."""
        tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        self._draw_grid_lines(tile)

        fg = Grid.render_tile(
            cell,
            agent_dir=agent_dir,
            highlight=False,
            tile_size=tile_size,
        )
        is_bright = np.max(fg, axis=2) > 20
        is_default_grid = np.all(np.abs(fg.astype(np.int16) - 100) <= 8, axis=-1)
        content = is_bright & ~is_default_grid
        tile[content] = fg[content]

        if highlight:
            highlight_img(tile)
        return tile

    def _render_tile_wall(self, highlight=False, tile_size=TILE_PIXELS):
        """Render outer wall tiles in off-white grey."""
        tile = np.full((tile_size, tile_size, 3), WALL_COLOR, dtype=np.uint8)
        if highlight:
            highlight_img(tile)
        return tile

    def _render_interactive_grid(self, tile_size, agent_pos, agent_dir, highlight_mask):
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        for j in range(self.height):
            for i in range(self.width):
                cell = self.grid.get(i, j)
                agent_here = np.array_equal(agent_pos, (i, j))
                if self._is_wall_cell(cell):
                    tile_img = self._render_tile_wall(
                        highlight=highlight_mask[i, j],
                        tile_size=tile_size,
                    )
                else:
                    tile_img = self._render_tile_white_floor(
                        cell,
                        agent_dir=agent_dir if agent_here else None,
                        highlight=highlight_mask[i, j],
                        tile_size=tile_size,
                    )

                ymin, ymax = j * tile_size, (j + 1) * tile_size
                xmin, xmax = i * tile_size, (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """Render full grid with white interior and grey wall border."""
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        highlight_mask = self._get_highlight_mask(highlight)
        img = self._render_interactive_grid(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask,
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    def step(self, action):
        # Let the parent class run standard logic first (handles step counter and standard actions)
        # If action is done, the parent class does nothing, just consumes one step of time
        obs, reward, done, info = super().step(action)

        # Add custom Switch logic
        if action == self.Actions.done:
            # Get the position and object in front (parent class doesn't pass these variables out)
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if self.carrying is not None and fwd_cell is not None:
                # Ensure the object in front can be picked up (avoid swapping things into walls or doors)
                if fwd_cell.can_pickup():
                    temp_new_obj = fwd_cell
                    self.grid.set(*fwd_pos, self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = temp_new_obj
                    self.carrying.cur_pos = np.array([-1, -1])
                    
                    # Critical: regenerate observation since we changed the environment state
                    # Otherwise the Agent will only see the changes in the next step
                    obs = self.gen_obs()

        return obs, reward, done, info
