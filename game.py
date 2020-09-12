import pygame
import functools
import numpy as np
# import cupy as np

np.random.seed(0)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Player:
    def __init__(self, area_width, area_height, speed, turn_speed):
        self.area_width = area_width
        self.area_height = area_height

        self.x = np.random.uniform(high=area_width, size=1).astype(np.float32)
        self.y = np.random.uniform(high=area_height, size=1).astype(np.float32)
        self.angle = np.random.uniform(high=360, size=1).astype(np.float32)
        self.color = np.random.randint(255, size=3)

        self.speed = speed
        self.turn_speed = turn_speed

        self.alive = True

    def move(self, turn):
        if not self.alive:
            return

        self.angle += turn * self.turn_speed

        self.x += np.cos(np.radians(self.angle)) * self.speed
        self.y += np.sin(np.radians(self.angle)) * self.speed

        # Wrap movement around
        self.x %= self.area_width
        self.y %= self.area_height

    def get_position(self):
        return self.x.astype(int)[0], self.y.astype(int)[0]

    def get_angle(self):
        return self.angle

    def get_speed(self):
        return self.speed

    def get_color(self):
        return self.color

    def kill(self):
        self.alive = False

    def is_alive(self):
        return self.alive

    def draw(self, screen):
        pygame.draw.circle(screen, RED, self.get_position(), 5)


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.grid = np.zeros((width, height)).astype(np.int32)

    def get_location(self, x, y):
        return self.grid[x][y]

    def get_full_grid(self):
        return self.grid.copy()

    def update(self, players, update_radius):
        player_is_alive = list(map(lambda p: p.is_alive(), players))

        for i in range(len(players)):
            if player_is_alive[i]:
                in_range_mask = self.get_in_range_mask(players[i].get_position(), update_radius).astype(int)
                updatable_mask = self.get_updatable_mask()
                self.grid += in_range_mask * updatable_mask * (i + 1)  # Color 0 is not of a player, its empty/black

    def get_in_range_mask(self, location, radius):
        location_matrix = self.get_location_matrix()
        distance_matrix = self.calculate_euclidean_distance_matrix(location_matrix, location)
        in_range_mask = np.less(distance_matrix, radius)
        return in_range_mask

    def get_location_matrix(self):
        return np.moveaxis(np.indices((self.width, self.height)).astype(np.int32), 0, -1)

    @staticmethod
    def calculate_euclidean_distance_matrix(location_matrix, location):
        return np.sqrt(np.sum(np.square((location_matrix - np.asarray(location)).astype(np.int32)), axis=2))\
            .astype(np.float32)

    def get_updatable_mask(self):
        return np.equal(self.grid, 0)

    def draw(self, player_colors, screen):
        grey_matrix = (np.dstack((self.grid, self.grid, self.grid)) * 255) // len(player_colors)  # .get()
        grid_image = pygame.surfarray.make_surface(grey_matrix)
        screen.blit(grid_image, (0, 0))


class TrAIlCrossers:
    AREA_WIDTH = 256
    AREA_HEIGHT = 256
    GRID_UPDATE_RADIUS = 2

    SPEED = 3
    TURN_SPEED = 6

    DEATH_PENALTY = -1
    WIN_REWARD = 1

    # Positive will result in dragging out game for as long as possible,
    # Negative in either suicide or very competitive play, 0 is neutral
    SURVIVE_REWARD = 0

    SHOW_FPS_EVERY = 30

    ALLOWED_INPUTS = 3
    PLAYER_STATE_COUNT = 5

    def __init__(self, n_players):
        pygame.init()
        self.screen = pygame.display.set_mode((TrAIlCrossers.AREA_WIDTH, TrAIlCrossers.AREA_HEIGHT))
        self.screen.set_alpha(None)
        pygame.display.set_caption('TrAIlCrossers')

        self.clock, self.current_step, self.players, self.grid = (None, None, None, None)
        self.n_players = n_players
        self.reset()

        self.toggled_fps = 0
        self.is_drawing = True

    def reset(self):
        self.clock = pygame.time.Clock()
        self.current_step = 0

        self.players = []
        for i in range(self.n_players):
            self.players.append(Player(TrAIlCrossers.AREA_WIDTH, TrAIlCrossers.AREA_HEIGHT,
                                       TrAIlCrossers.SPEED, TrAIlCrossers.TURN_SPEED))

        self.grid = Grid(TrAIlCrossers.AREA_WIDTH, TrAIlCrossers.AREA_HEIGHT)

        return self.grid.get_full_grid(), self.build_player_states()

    def step(self, player_actions):  # Player actions should be [-1, 0, 1]
        pygame.event.get()  # Prevent OS from thinking game not responding
        self.clock.tick()
        self.current_step += 1

        self.handle_input()

        player_got_killed = []
        for i in range(len(self.players)):
            self.players[i].move(player_actions[i])
            if self.players[i].is_alive() and \
                    (self.players[i].get_position()[0] < 0 or self.players[i].get_position()[0] >= self.AREA_WIDTH or
                     self.players[i].get_position()[1] < 0 or self.players[i].get_position()[1] >= self.AREA_HEIGHT):
                self.players[i].kill()
                player_got_killed.append(True)
            elif self.players[i].is_alive() and self.is_colliding(self.players[i].get_position()):
                self.players[i].kill()
                player_got_killed.append(True)
            else:
                player_got_killed.append(False)

        self.grid.update(self.players, TrAIlCrossers.GRID_UPDATE_RADIUS)
        rewards, finished = self.calculate_rewards(player_got_killed)

        if self.is_drawing:
            self.draw()
        else:
            pygame.display.update(self.draw_fps())

        return self.grid.get_full_grid(), self.build_player_states(), rewards, finished

    def is_colliding(self, position):
        if self.grid.get_location(position[0], position[1]) != 0:
            return True
        return False

    def calculate_rewards(self, player_got_killed):
        alive_count = functools.reduce(lambda p1, p2: p1 + p2, list(map(lambda p: p.is_alive(), self.players)))
        rewards = []

        for i in range(len(self.players)):
            if player_got_killed[i]:
                rewards.append(TrAIlCrossers.DEATH_PENALTY)
            elif alive_count == 1 and self.players[i].is_alive():
                rewards.append(TrAIlCrossers.WIN_REWARD)
            else:
                rewards.append(TrAIlCrossers.SURVIVE_REWARD)

        return rewards, alive_count < 2

    def build_player_states(self):
        player_states = []
        for i in range(len(self.players)):
            position = self.players[i].get_position()
            state = [self.players[i].is_alive(), position[0].item(), position[1].item(),
                     self.players[i].get_angle().item(), self.players[i].get_speed()]
            player_states.append(state)
        return np.array(player_states)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_f] and self.toggled_fps == 0:
            self.is_drawing = not self.is_drawing
            self.toggled_fps += 1
        elif self.toggled_fps != 0:
            self.toggled_fps = (self.toggled_fps + 1) % 100  # Wont toggle twice within a 100 frames

    def draw(self):
        self.screen.fill(BLACK)

        self.grid.draw(self.get_grid_colors(), self.screen)
        for player in self.players:
            player.draw(self.screen)

        self.draw_fps()
        pygame.display.update()

    def get_grid_colors(self):
        return [np.array([0, 0, 0])] + list(map(lambda x: x.get_color(), self.players))

    def draw_fps(self):
        if not self.current_step % TrAIlCrossers.SHOW_FPS_EVERY or self.is_drawing:
            rect = self.screen.blit(pygame.font.SysFont('Comic Sans MS', 15).render(
                f"Rendering with {1000 / self.clock.get_time()}fps", False, WHITE), (10, 10))
            return rect
        else:
            return None

    @staticmethod
    def stop():
        pygame.quit()
