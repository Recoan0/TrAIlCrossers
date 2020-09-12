import numpy as np
# import cupy as np
import time

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, MaxPool2D
from tensorflow.keras.optimizers import Adam

from ReinforcementLearning import AgentTrainer


class A2CAgent:
    GAMMA = 0.99
    LEARNING_RATE = 0.001

    def __init__(self, model_name, board_shape, state_shape, output_count):
        self.model_name = model_name
        self.board_shape = board_shape
        self.state_shape = state_shape
        self.output_count = output_count

        self.actor = self.create_actor(board_shape, state_shape, output_count)
        self.critic = self.create_critic(board_shape, state_shape)

    def create_actor(self, board_shape, state_shape, output_count):
        state_input = Input(state_shape)
        board_input, flat = self.create_convolution_layers(board_shape)

        concat = Concatenate()([flat, state_input])

        d = Dense(64, activation='relu')(concat)
        d = Dense(64, activation='relu')(d)
        d = Dense(32, activation='relu')(d)
        output = Dense(output_count, activation='softmax')(d)

        actor = Model([board_input, state_input], output)
        actor.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='categorical_crossentropy')
        print(actor.summary())
        return actor

    def create_critic(self, board_shape, state_shape):
        state_input = Input(state_shape)
        board_input, flat = self.create_convolution_layers(board_shape)

        concat = Concatenate()([flat, state_input])

        d = Dense(64, activation='relu')(concat)
        d = Dense(64, activation='relu')(d)
        d = Dense(32, activation='relu')(d)
        output = Dense(1, activation='linear')(d)

        critic = Model([board_input, state_input], output)
        critic.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')
        return critic

    def create_convolution_layers(self, input_shape):
        network_input = Input(input_shape)

        c = Conv2D(2, (3, 3), padding='same', activation='relu')(network_input)
        c = MaxPool2D((2, 2))(c)
        c = Conv2D(4, (3, 3), padding='same', activation='relu')(c)
        c = MaxPool2D((2, 2))(c)
        c = Conv2D(8, (3, 3), padding='same', activation='relu')(c)
        c = MaxPool2D((2, 2))(c)
        c = Conv2D(16, (3, 3), padding='same', activation='relu')(c)
        c = MaxPool2D((2, 2))(c)
        c = Conv2D(16, (3, 3), padding='same', activation='relu')(c)
        c = MaxPool2D((2, 2))(c)
        c = Conv2D(32, (3, 3), padding='same', activation='relu')(c)
        c = MaxPool2D((2, 2))(c)

        flattened = Flatten()(c)

        return network_input, flattened

    def get_action(self, board, player_states):
        policy = self.actor([np.expand_dims(np.array(board), axis=0),  # .get(),
                             np.array(player_states).reshape(-1, *self.state_shape)  # .get()
                             ], training=False).numpy().ravel()
        return np.random.choice(np.arange(self.output_count), 1, p=policy)[0]

    def train(self, boards, player_states, actions, rewards, next_boards, next_states, dones):
        targets = np.zeros((len(player_states), 1))
        advantages = np.zeros((len(player_states), self.output_count))

        expanded_boards = np.expand_dims(np.array(boards), axis=3)

        for i in range(len(player_states)):
            expanded_board = np.expand_dims(np.array(boards[i]), axis=0)
            expanded_next_board = np.expand_dims(np.array(next_boards[i]), axis=0)
            reshaped_state = np.array(player_states[i]).reshape(-1, *self.state_shape)
            reshaped_next_state = np.array(next_states[i]).reshape(-1, *self.state_shape)
            value = self.critic([expanded_board, reshaped_state], training=False).numpy()
            next_value = self.critic([expanded_next_board, reshaped_next_state], training=False).numpy()

            if dones[i]:
                advantages[i][actions[i]] = rewards[i] - value
                targets[i][0] = rewards[i]
            else:
                advantages[i][actions[i]] = rewards[i] + self.GAMMA * next_value - value
                targets[i][0] = rewards[i] + self.GAMMA * next_value

        self.actor.fit([expanded_boards, np.array(player_states).reshape(len(actions), -1)], advantages, verbose=0)
        self.critic.fit([expanded_boards, np.array(player_states).reshape(len(actions), -1)], targets, verbose=0)

    def save(self, average_reward, min_reward, max_reward):
        self.actor.save(f'models/{self.model_name}_Actor__{max_reward:>7.2f}max_'
                        f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        self.critic.save(f'models/{self.model_name}_Critic__{max_reward:>7.2f}max_'
                         f'{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


class A2CTrainer(AgentTrainer):
    def __init__(self, env, agent_count, model_name, training_steps):
        AgentTrainer.__init__(self, env)

        self.training_steps = training_steps
        self.agent_count = agent_count
        self.agents = []
        for i in range(agent_count):
            self.agents.append(A2CAgent(model_name + str(i), (env.AREA_WIDTH, env.AREA_HEIGHT, 1),
                                        (env.PLAYER_STATE_COUNT * agent_count,), env.ALLOWED_INPUTS))

    def run(self, episodes):
        game_number = 1

        actions, boards = [[] for _ in range(self.agent_count)], [[] for _ in range(self.agent_count)]
        states, next_states = [[] for _ in range(self.agent_count)], [[] for _ in range(self.agent_count)]
        rewards, dones = [[] for _ in range(self.agent_count)], [[] for _ in range(self.agent_count)]
        next_boards = [[] for _ in range(self.agent_count)]

        while game_number <= episodes:
            done = False
            board, player_states = self.env.reset()
            print(f'Game number: {game_number}')

            while not done:  # Game loop
                player_actions = []
                for i in range(self.agent_count):
                    player_actions.append(self.agents[i].get_action(board, player_states.ravel()))

                new_board, new_player_states, action_rewards, done = \
                    self.env.step(self.adjust_actions_to_game(player_actions))

                for i in range(self.agent_count):
                    is_alive = new_player_states[i][0]

                    if is_alive == 1 or len(dones[i]) == 0 or (not dones[i][-1]):  # If the agent is still alive or wasnt dead last step
                        boards[i].append(board)
                        next_boards[i].append(new_board)
                        states[i].append(player_states)
                        next_states[i].append(new_player_states)

                        actions[i].append(player_actions[i])
                        rewards[i].append(action_rewards[i])
                        dones[i].append(is_alive == 0)

                        if not len(states[i]) % self.training_steps:
                            self.agents[i].train(boards[i], states[i], actions[i], rewards[i],
                                                 next_boards[i], next_states[i], dones[i])
                            boards[i], next_boards[i], states[i], next_states[i] = [], [], [], []
                            actions[i], rewards[i], dones[i] = [], [], []

                board = new_board
                player_states = new_player_states

            game_number += 1

        self.env.stop()

    @staticmethod
    def adjust_actions_to_game(actions):
        return list(map(lambda x: x - 1, actions))
