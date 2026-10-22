import { Gamepad2 } from "lucide-react";
import { DocHeader } from "@/components/DocHeader";
import { CodeBlock } from "@/components/CodeBlock";

export default function ReinforcementDocs() {
  return (
    <>
      <DocHeader
        title="Reinforcement Learning"
        description="Q-Learning, SARSA and DQN agents over the GridWorld environment: tabular RL works out of the box, deep RL uses Keras via the dl extra."
        icon={<Gamepad2 className="w-6 h-6" />}
        version="0.1.7"
      />

      <section className="mb-12">
        <h2 className="section-title">Quick Start</h2>
        <CodeBlock
          title="qlearning.py"
          code={`import synaptix as sx

env = sx.reinforcement.GridWorld(size=5)      # 5x5 grid, stochastic slips
agent = sx.reinforcement.QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1,
)

# Train
for episode in range(500):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state

print("Learned policy sample:", agent.policy[:5])`}
        />
      </section>

      <section className="mb-12">
        <h2 className="section-title">Agents &amp; Environment</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3 pr-4">Component</th>
                <th className="text-left font-syne text-xs font-semibold text-white uppercase tracking-wider pb-3">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                ["GridWorld", "2D grid environment with walls, slippery cells, rewards and terminal states."],
                ["QLearningAgent", "Off-policy tabular agent — learns the optimal action-value function."],
                ["SARSAAgent", "On-policy tabular agent — updates with the action actually taken."],
                ["DQNAgent", "Deep Q-Network over Keras (requires synaptix[dl]) for large state spaces."],
              ].map(([c, d]) => (
                <tr key={c} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 pr-4 font-mono text-xs text-accent">{c}</td>
                  <td className="py-2.5 text-muted">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="section-title">SARSA Variant</h2>
        <CodeBlock
          title="sarsa.py"
          code={`agent = sx.reinforcement.SARSAAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1,
)

for episode in range(300):
    state = env.reset()
    action = agent.act(state)
    done = False
    while not done:
        next_state, reward, done = env.step(action)
        next_action = agent.act(next_state)
        agent.learn(state, action, reward, next_state, next_action)
        state, action = next_state, next_action`}
        />
      </section>
    </>
  );
}
