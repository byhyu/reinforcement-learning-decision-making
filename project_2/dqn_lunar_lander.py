import time

class DqnAgent:

    def evaluate(self, env, render=True):
        state = env.reset()
        done, ep_reward = False, 0
        while not done:
            action = self.act(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward


