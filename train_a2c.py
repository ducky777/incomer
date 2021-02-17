import logging

from stock_env import FXEnv, LoadData
from agents import A2CAgent, PPOAgent

logging.basicConfig(filename='logs/a2c.log', level=logging.INFO)
# logging.root.setLevel(logging.NOTSET)

if __name__ == '__main__':
    symbol = "EURUSD"
    timeframe = 60
    filename = "%s%s.csv" % (symbol, timeframe)
    lookbacks = 241
    total_bars = 60
    save_suffix = '%s%s2' % (symbol, timeframe)

    logging.info("Starting app training...")
    print("Starting app training...")

    x, y, body, data = LoadData.load_fx(filename, lookbacks=lookbacks,
                                    total_bars=total_bars)
    env = FXEnv(x, body, data.Open, data.Close, max_trades=50,
                periods_per_episode=50, spread=0.0001)
    # agent = A2CAgent(model_type='cnn', lr=7e-3, gamma=0.99,
    #                 value_c=0.5, entropy_c=1e-4, save_suffix=save_suffix,
    #                 lookbacks=lookbacks)
    agent = PPOAgent(lookbacks)
    # rewards = agent.train(env, batch_size=256, updates=10000)
    agent.train(env, num_steps=500000, ppo_steps=128)
