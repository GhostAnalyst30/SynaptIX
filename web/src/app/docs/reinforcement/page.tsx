import { Gamepad2 } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { MethodCard } from "@/components/MethodCard";
import { CodeBlock } from "@/components/CodeBlock";

export default function ReinforcementDocs() {
  return (
    <>
      <DocHeader
        title="Reinforcement Learning"
        description="Q-Learning, SARSA and DQN agents over the GridWorld environment: tabular RL works out of the box, deep RL uses Keras via the dl extra."
        icon={<Gamepad2 className="w-6 h-6" />}
        version="0.2.0"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="qlearning.py"
          code={`import synaptix as sx

env = sx.reinforcement.GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])
agent = sx.reinforcement.QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,
    gamma=0.95,
    seed=0,
)

rewards = agent.train(env, episodes=500)   # list with the reward of each episode
agent.plot_rewards(window=20)              # moving-average learning curve
env.render(agent.policy())                 # grid with the greedy action per cell
print(agent.q_table.shape)                 # (n_states, n_actions)`}
        />
        <p className="text-sm text-muted leading-relaxed mt-4">
          Need a custom loop? Use <code className="code-inline">agent.act(state, greedy=False)</code> with{" "}
          <code className="code-inline">env.reset()</code> and{" "}
          <code className="code-inline">env.step(action) -&gt; (state, reward, done)</code>.
        </p>
      </section>

      <section className="mb-12">
        <h2 className="section-title">Train / Test Episodes</h2>
        <p className="text-sm text-muted leading-relaxed mb-4">
          In RL, <code className="code-inline">test_size</code> is the share of episodes held out to
          evaluate the learned policy. <code className="code-inline">fit_evaluate()</code> trains with
          the exploratory (epsilon-greedy) policy on the rest, then plays the held-out episodes greedily
          without learning. <code className="code-inline">evaluate()</code> runs only the greedy part.
        </p>
        <CodeBlock
          title="rl_train_test.py"
          code={`env = sx.reinforcement.GridWorld(rows=5, cols=5, obstacles=[(1, 1), (2, 3)])
agent = sx.reinforcement.QLearningAgent(env.n_states, env.n_actions, seed=0)

result = agent.fit_evaluate(env, episodes=500, test_size=0.2)   # 400 train / 100 test
print(result.to_frame())
# recompensa_media, recompensa_std, recompensa_min, recompensa_max,
# pasos_medios, tasa_exito (when the env implements is_success())

agent.evaluate(env, episodes=50, metrics=["recompensa_media", "tasa_exito"])`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">API Reference</h2>
        <div className="method-list">
          <MethodCard
            name="GridWorld"
            signature="GridWorld(rows=4, cols=4, start=(0, 0), goal=None, obstacles=None, max_steps=200)"
            description="2D grid environment. Actions: 0=up, 1=right, 2=down, 3=left. Rewards: +10 at the goal, -1 per step, -5 when hitting an obstacle or the border. Attributes <code className='code-inline'>n_states</code>, <code className='code-inline'>n_actions</code>. Methods: <code className='code-inline'>reset()</code>, <code className='code-inline'>step(action)</code>, <code className='code-inline'>render(policy=None)</code>, <code className='code-inline'>state_id(position)</code>, <code className='code-inline'>position_of(state)</code>, <code className='code-inline'>is_success()</code>."
            parameters={[
              { name: "goal", type: "tuple | None", description: "Goal cell; defaults to the bottom-right corner.", default: "None" },
              { name: "obstacles", type: "list[tuple] | None", description: "Blocked cells.", default: "None" },
              { name: "max_steps", type: "int", description: "Maximum steps per episode.", default: "200" },
            ]}
          />

          <MethodCard
            name="QLearningAgent / SARSAAgent"
            signature="QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, seed=None)"
            description="Tabular agents: Q-Learning is off-policy, SARSA on-policy (same signature). Methods: <code className='code-inline'>act(state, greedy=False)</code>, <code className='code-inline'>train(env, episodes=500, verbose=False)</code>, <code className='code-inline'>policy()</code>, <code className='code-inline'>plot_rewards(window=20)</code>, <code className='code-inline'>evaluate(env, episodes=100, metrics=None, max_steps=1000, verbose=True)</code>, <code className='code-inline'>fit_evaluate(env, episodes=500, test_size=0.2, metrics=None, verbose=True)</code>. Attributes: <code className='code-inline'>q_table</code>, <code className='code-inline'>rewards_history</code>, <code className='code-inline'>steps_history</code>, <code className='code-inline'>success_history</code>."
            example={`sarsa = sx.reinforcement.SARSAAgent(env.n_states, env.n_actions, epsilon_decay=0.99)
sarsa.train(env, episodes=300)
sarsa.evaluate(env, episodes=50)`}
          />

          <MethodCard
            name="DQNAgent"
            signature="DQNAgent(state_dim, n_actions, hidden_layers=(64, 64), gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-3, buffer_size=10000, batch_size=64, target_update_every=10)"
            description="Deep Q-Network over Keras (requires synaptix[dl]) for environments with vector states. Methods: <code className='code-inline'>act(state, greedy=False)</code>, <code className='code-inline'>remember(state, action, reward, next_state, done)</code>, <code className='code-inline'>replay()</code>, <code className='code-inline'>train(env, episodes=200, verbose=True)</code>, <code className='code-inline'>plot_rewards(window=10)</code>, <code className='code-inline'>evaluate</code>, <code className='code-inline'>fit_evaluate</code>."
            example={`agent = sx.reinforcement.DQNAgent(state_dim=4, n_actions=2)
agent.train(my_env, episodes=200)   # env.reset() -> vector, env.step(a) -> (vector, reward, done)`}
          />
        </div>
      </section>
    </>
  );
}
