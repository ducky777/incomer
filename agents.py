import os
import logging

import matplotlib.pyplot as plt

import random

from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate
import tensorflow.keras.losses as kls
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

# logging.basicConfig(filename='logs/a2c.log', level=logging.INFO)

def res_block(x, filters, size, stride, downsample=False):
    y = Conv1D(filters, size, (1 if not downsample else 2), padding='same')(x)
    y = Activation('relu')(y)
    y = Conv1D(filters, size, 1, padding='same')(y)

    if downsample:
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   strides=2,
                   padding='same')(x)

    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    clipping_val = 0.2
    entropy_beta = 0.001
    critic_discount = 0.5
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

def ppo_loss2(y_true, y_pred, oldpolicy_probs, advantages, rewards, values):
    clipping_val = 0.2
    entropy_beta = 0.001
    critic_discount = 0.5
    newpolicy_probs = y_pred
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    critic_loss = K.mean(K.square(rewards - values))
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
        -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))

    return total_loss

class PPOAgent:
    def __init__(self, lookbacks, model_type='cnn', lr=7e-3,
                 clipping_val=0.2, critic_discount=0.5, entropy_beta=0.001,
                 gamma=0.99, lmbda=0.95):
        self.model_type = model_type
        self.lookbacks = lookbacks
        self.clipping_val = clipping_val
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.lmbda = lmbda

        # create actor model
        self.model_actor = self._create_cnn_actor()

        # create critic model
        self.model_critic = self._create_cnn_critic()

    def custom_train(self, model, x, y, epochs=10):
        # print("Start of epoch %d" % (epochs,))

        optimizer=Adam(lr=1e-4)

        dummy_n = np.zeros((1, 1, 3))
        dummy_1 = np.zeros((1, 1, 1))

        # Iterate over the batches of the dataset.
        for step in range(len(x[0])):
            with tf.GradientTape() as tape:
                y_pred = model([x[0], x[1], dummy_n, dummy_1, dummy_1, dummy_1])
                # Compute reconstruction loss
                loss = ppo_loss2(y, y_pred, x[2], x[3], x[4], x[5])
                # loss += sum(vae.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # loss_metric(loss)
        return model

    def _create_cnn_critic(self):
        account_state = Input(shape=(4,))
        start_model = Input(shape=(self.lookbacks - 1, 3))

        input_model = Conv1D(kernel_size=24,
                            strides=1,
                            filters=64,
                            padding='same')(start_model)
        input_model = res_block(start_model, 128, 12, 3, downsample=True)
        input_model = res_block(input_model, 256, 6, 3, downsample=True)
        input_model = res_block(input_model, 512, 3, 3, downsample=True)
        input_model = res_block(input_model, 1024, 2, 3, downsample=True)
        input_model = MaxPooling1D()(input_model)
        input_model = Flatten()(input_model)
        input_model = Concatenate(axis=-1)([input_model, account_state])

        # actor layers
        actor = Dense(64)(input_model)
        actor = LeakyReLU()(actor)
        actor = Dense(1, activation='linear')(actor)

        model = Model([start_model, account_state], [actor])

        model.compile(optimizer=Adam(lr=1e-4), loss='mse')

        return model

    def _create_cnn_actor(self):
        account_state = Input(shape=(4,), name='account_state')
        start_model = Input(shape=(self.lookbacks - 1, 3), name='market_state')
        oldpolicy_probs = Input(shape=(1, 3, ), name='oldpolicy_probs')
        advantages = Input(shape=(1, 1, ), name='advantages')
        rewards = Input(shape=(1, 1, ), name='rewards')
        values = Input(shape=(1, 1, ), name='values')

        input_model = Conv1D(kernel_size=24,
                            strides=1,
                            filters=64,
                            padding='same')(start_model)
        input_model = res_block(start_model, 128, 12, 3, downsample=True)
        input_model = res_block(input_model, 256, 6, 3, downsample=True)
        input_model = res_block(input_model, 512, 3, 3, downsample=True)
        input_model = res_block(input_model, 1024, 2, 3, downsample=True)
        input_model = MaxPooling1D()(input_model)
        input_model = Flatten()(input_model)
        input_model = Concatenate(axis=-1)([input_model, account_state])

        # actor layers
        actor = Dense(64)(input_model)
        actor = LeakyReLU()(actor)
        actor = Dense(3, activation='softmax')(actor)

        model = Model([start_model, account_state, oldpolicy_probs,
                       advantages, rewards, values], [actor])

        # model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        #     oldpolicy_probs=oldpolicy_probs,
        #     advantages=advantages,
        #     rewards=rewards,
        #     values=values)])

        # model.compile(optimizer=Adam(lr=1e-4), loss='mse')

        return model

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def test_reward(self, env):
        state = env.reset()

        dummy_n = np.zeros((1, 1, 3))
        dummy_1 = np.zeros((1, 1, 1))
        done = False
        total_reward = 0
        # print('Testing...')
        limit = 0
        while not done:
            state_input = state
            action_probs = self.model_actor([state_input[0], state_input[1],
                                             dummy_n, dummy_1, dummy_1, dummy_1])
            action = np.argmax(action_probs)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            limit += 1
            if limit > 9000:
                break
        return total_reward

    def train(self, env, num_steps, ppo_steps=128):
        ep_reward_dir = './logs/tensorboard/ppo'
        ep_reward_writer = tf.summary.create_file_writer(ep_reward_dir)

        best_reward = 0
        iters = 0

        store_rewards = deque(maxlen=10)

        state = env.reset()
        for cur_step in range(num_steps):
            states = []
            actions = []
            values = []
            masks = []
            rewards = []
            actions_probs = []
            actions_onehot = []
            state_input = None

            dummy_n = np.zeros((1, 1, 3))
            dummy_1 = np.zeros((1, 1, 1))

            for itr in range(ppo_steps):
                state_input = state
                action_dist = self.model_actor([state_input[0], state_input[1], dummy_n,
                                                   dummy_1, dummy_1, dummy_1])
                q_value = self.model_critic(state_input)
                action = np.random.choice(3, p=np.squeeze(action_dist))
                action_onehot = np.zeros(3)
                action_onehot[action] = 1

                observation, reward, done, _ = env.step(action)
                # print('step: [%i/%i] || itr: [%i/%i] || action=%i || reward=%.5f || q_value=%.5f'
                #       % (cur_step, num_steps, itr, ppo_steps, action, reward, q_value))
                mask = not done

                states.append(state)
                actions.append(action)
                actions_onehot.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                state = observation
                if done:
                    state = env.reset()

            q_value = self.model_critic(state_input)
            values.append(q_value)
            returns, advantages = self.get_advantages(values, masks, rewards)

            market_states = np.array([i[0] for i in states]).squeeze(1)
            account_states = np.array([i[1] for i in states]).squeeze(1)

            actions_probs = np.array([i.numpy() for i in actions_probs], dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32).reshape((-1, 1, 1))
            actions_onehot = np.array(actions_onehot, dtype=np.float32).reshape((-1, 3))
            values = np.array([float(i[0]) for i in values], dtype=np.float32).reshape(-1, 1, 1)

            # actor_loss = self.model_actor.fit(
            #     [market_states, account_states, actions_probs, advantages, rewards, values[:-1]],
            #     [actions_onehot], verbose=True, shuffle=True, epochs=8,
            #     callbacks=[tensor_board], batch_size=256)

            y = np.array([float(i) for i in returns])

            self.model_actor = self.custom_train(self.model_actor,
                                                 [market_states, account_states, actions_probs,
                                                  advantages, rewards, values[:-1]],
                                                 [actions_onehot], epochs=10)
            critic_loss = self.model_critic.fit([market_states, account_states], [y],
                                                shuffle=True, epochs=10, verbose=0, batch_size=256)

            avg_reward = np.mean([self.test_reward(env) for _ in range(1)])
            # print('total test reward=' + str(avg_reward))

            store_rewards.append(avg_reward)

            mean_rewards = np.mean(store_rewards)

            print("Avg reward @ step %i: %.5f" % (cur_step, mean_rewards))

            if mean_rewards > best_reward:
                print('best reward = %.5f! Saving model...' % mean_rewards)
                self.model_actor.save('model_actor_%i_%.5f.hdf5' % (iters, mean_rewards))
                self.model_critic.save('model_critic_%i_%.5f.hdf5' % (iters, mean_rewards))
                best_reward = mean_rewards
            iters += 1
            state = env.reset()

            with ep_reward_writer.as_default():
                tf.summary.scalar("Episode Reward", mean_rewards, step=cur_step)


class A2CAgent:
    def __init__(self, model_type='lstm', lr=7e-3, gamma=0.99,
                 value_c=0.5, entropy_c=1e-4, save_suffix=None,
                 lookbacks=241):

        if model_type not in ['cnn', 'lstm']:
            raise ValueError('Model type %s not supported' % model_type)

        logging.info("Initializing A2C agent")

        self.model_type = model_type
        self.value_c = value_c
        self.entropy_c = entropy_c
        self.gamma = gamma
        self.save_suffix = save_suffix
        self.lookbacks = lookbacks
        self.model_filename = 'a2c_%s_%s.h5' % (self.model_type, self.save_suffix)

        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.np_eps = np.finfo(np.float32).eps.item()

        # if os.path.isfile(self.model_filename):
        #     logging.info("File exists. Loading current model...")
        #     print(("File exists. Loading current model..."))
        #     self.model = tf.keras.models.load_model(self.model_filename,
        #                                             custom_objects={'_logits_loss': self._logits_loss,
        #                                                             '_value_loss': self._value_loss})
        #     # self.model = self._create_lstm()
        #     # self.model.set_weights(temp_model.get_weights())
        # else:
        logging.info("No saved model. Instantiating a new one...")
        print("No saved model. Instantiating a new one...")
        if model_type == 'cnn':
            self.model_actor = self._create_cnn(3)
            self.model_critic = self._create_cnn(1)
            # self.model.load_weights('a2c_cnn_EURUSD602_weights.h5')
        elif model_type == 'lstm':
            self.model = self._create_lstm()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss=[self._logits_loss, self._value_loss])


        print("Model instantiated!")

    def _create_cnn(self, num_final_layer):
        account_state = Input(shape=(4,))
        start_model = Input(shape=(self.lookbacks - 1, 3))
        input_model = Conv1D(kernel_size=24,
                            strides=1,
                            filters=32,
                            padding='same')(start_model)
        input_model = res_block(start_model, 64, 12, 3, downsample=True)
        input_model = res_block(input_model, 128, 6, 3, downsample=True)
        input_model = res_block(input_model, 256, 3, 3, downsample=True)
        input_model = res_block(input_model, 512, 2, 3, downsample=True)
        input_model = MaxPooling1D()(input_model)
        input_model = Flatten()(input_model)
        input_model = Concatenate(axis=-1)([input_model, account_state])

        # actor layers
        actor = Dense(64)(input_model)
        actor = LeakyReLU()(actor)
        actor = Dense(num_final_layer, activation='softmax')(actor)

        # critic layers
        # critic = Dense(128)(input_model)
        # critic = LeakyReLU()(critic)
        # critic = Dense(1, activation='linear')(critic)

        model = Model([start_model, account_state], [actor])
        return model

    def _create_lstm(self):
        account_state = Input(shape=(4,))
        start_model = Input(shape=(self.lookbacks - 1, 3))
        # input_model = BatchNormalization()(start_model)
        input_model = Bidirectional(LSTM(256, return_sequences=True))(start_model)
        input_model = Bidirectional(LSTM(128, return_sequences=True))(input_model)
        input_model = LSTM(64)(input_model)
        input_model = Concatenate(axis=-1)([input_model, account_state])
        input_model = Dense(128)(input_model)
        input_model = LeakyReLU()(input_model)
        # input_model = BatchNormalization()(input_model)

        # actor
        actor = Dense(64)(input_model)
        actor = LeakyReLU()(actor)
        # actor = BatchNormalization()(actor)
        actor = Dense(128)(actor)
        actor = LeakyReLU()(actor)
        # actor = BatchNormalization()(actor)
        actor = Dense(3, activation='softmax')(actor)

        # critic layers
        critic = Dense(32)(input_model)
        critic = LeakyReLU()(critic)
        # critic = BatchNormalization()(critic)
        critic = Dense(64)(critic)
        critic = LeakyReLU()(critic)
        # critic = BatchNormalization()(critic)
        critic = Dense(1, activation='linear')(critic)

        model = Model([start_model, account_state], [actor, critic])
        return model

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    def run_episode(self, env, gamma=0.99):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        state = env.reset()
        done = False

        counter = 0
        while not done:
            # Convert state into a batched tensor (batch size = 1)
            # state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(counter, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(counter, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done, _ = env.step(int(action))
            # state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(counter, reward)
            counter += 1

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        returns = self.get_expected_return(rewards, gamma)

        return action_probs, values, rewards, returns

    def get_expected_return(
            self,
            rewards: tf.Tensor,
            gamma: float,
            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                    (tf.math.reduce_std(returns) + self.np_eps))

        return returns

    def compute_loss(
            self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    # @tf.function
    def train_step(
        self,
        env,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards, returns = self.run_episode(env, gamma)

            # Calculate expected returns
            # returns = self.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward
        # return sum(rewards)

    def train(self, env, num_episodes, gamma=0.99):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        ep_reward_dir = './logs/tensorboard/rewards'
        ep_reward_writer = tf.summary.create_file_writer(ep_reward_dir)

        running_reward = 0
        best_running_reward = 0

        for ep in range(num_episodes):
            episode_reward = self.train_step(env, optimizer, gamma)
            running_reward = episode_reward*0.1 + running_reward*.99

            if running_reward > best_running_reward:
                self.model.save('a2c_%s_%s.h5' % \
                    (self.model_type, self.save_suffix))
                best_running_reward = running_reward

            with ep_reward_writer.as_default():
                tf.summary.scalar("Episode Reward", running_reward, step=ep)

            print(("[%d/%d] Episode Reward: %.5f" % (ep + 1, num_episodes, running_reward)))


    def train1(self, env, batch_size=256, updates=1000):
        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_reward_dir = './logs/tensorboard/rewards'
        ep_reward_writer = tf.summary.create_file_writer(ep_reward_dir)

        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            market_state, account_state = [], []
            actions, rewards, dones, values = [], [], [], []
            for step in range(batch_size):
                market_state.append(next_obs[0])
                account_state.append(next_obs[1])
                pr = self.model(next_obs)
                action = np.random.choice(3, p=np.squeeze(pr[0]))
                actions.append(action)
                values.append(float(pr[1]))
                next_obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (
                        len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model(next_obs)

            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            # acts_and_advs = np.concatenate((actions, advs), axis=0)
            acts_and_advs = np.column_stack((actions, advs))
            # print(len(observations))

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            # x = np.concatenate(observations, axis=0)
            x1 = np.reshape(market_state, (batch_size, self.lookbacks - 1, 3))
            x2 = np.reshape(account_state, (batch_size, 4))
            losses = self.model.train_on_batch([x1, x2], [acts_and_advs, returns])

            self.model.save('a2c_%s_%s.h5' % (self.model_type, self.save_suffix))

            # logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
            # print(("[%d/%d] Losses: %s" % (update + 1, updates, losses)))
            print(("[%d/%d] Episode Reward: %.5f" % (update + 1, updates, sum(rewards))))

            with ep_reward_writer.as_default():
                tf.summary.scalar("Episode Reward", sum(rewards), step=update)
                tf.summary.scalar("Episode Loss", sum(losses), step=update)

        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        # print(rewards)
        # print(next_value)
        # returns = [rewards]
        # returns.append(next_value)
        # returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        returns = rewards[:]
        returns.append(float(next_value))
        returns = np.array(returns)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # print(values)
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        return ep_reward