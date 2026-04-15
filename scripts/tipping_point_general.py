

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# 1. Define the vocabulary and fixed 2D embeddings
embeddings = {
    "A": np.array([0.4, -0.3, 0]),
    "B": np.array([0.8, 0.0, 0]),
    "C": np.array([-0.2, -0.2, 0.0]),
    "D": np.array([0.9, 0.5, 0])
}
vocab = list(embeddings.keys())

# 2. Utility: softmax function
def softmax(score_list):
    """Computes softmax for a list of scores."""
    exps = np.exp(score_list)
    return exps / np.sum(exps)

# 3. Initialize sequence and parameters
initial_prompt = ["A", "C", "C", "A"]
print(f"Initial prompt: {initial_prompt}")
max_length = 30

# 4. Histories for later plotting/animation
context_vec_history = []
seq_snapshots = []

# 5. Main simulation loop
current_sequence = list(initial_prompt)
while len(current_sequence) < max_length:
    n = len(current_sequence)
    sequence_embeddings_matrix = np.array([embeddings[token] for token in current_sequence])
    scores = np.dot(sequence_embeddings_matrix, sequence_embeddings_matrix.T)
    weights = np.apply_along_axis(softmax, 1, scores)
    context_vectors = np.dot(weights, sequence_embeddings_matrix)
    last_context = context_vectors[-1]
    context_vec_history.append(last_context)
    dot_products = {token: np.dot(last_context, vec) for token, vec in embeddings.items()}
    next_token = max(dot_products, key=dot_products.get)
    current_sequence.append(next_token)
    seq_snapshots.append(list(current_sequence))

print(f"\n--- Final Sequence ---\n{''.join(current_sequence)}")

# 6. Refactored plotting function for reuse
def draw_frame(ax, frame_index, title=""):
    """Draws a single frame of the simulation on a given axis."""
    ax.clear()

    # Plot fixed embeddings arrows
    ax.arrow(0, 0, embeddings['A'][0], embeddings['A'][1], head_width=0.04, head_length=0.04, fc='#006400', ec='#006400')
    ax.arrow(0, 0, embeddings['B'][0], embeddings['B'][1], head_width=0.04, head_length=0.04, fc='blue', ec='blue')
    ax.arrow(0, 0, embeddings['C'][0], embeddings['C'][1], head_width=0.04, head_length=0.04, fc='orange', ec='orange')
    ax.arrow(0, 0, embeddings['D'][0], embeddings['D'][1], head_width=0.04, head_length=0.04, fc='red', ec='red')

    # Add labels for each embedding vector
    ax.text(embeddings['A'][0] + 0.02, embeddings['A'][1] - 0.08, 'A', color='#006400', fontweight='bold', fontsize=16)
    ax.text(embeddings['B'][0] + 0.03, embeddings['B'][1] + 0.02, 'B', color='blue', fontweight='bold', fontsize=16)
    ax.text(embeddings['C'][0] - 0.05, embeddings['C'][1] + 0.03, 'C', color='orange', fontweight='bold', fontsize=16)
    ax.text(embeddings['D'][0], embeddings['D'][1] + 0.03, 'D', color='red', fontweight='bold', fontsize=16)

    # Plot trajectory
    trajectory = np.array([vec[:2] for vec in context_vec_history[:frame_index+1]])
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='purple', linewidth=2, zorder=1)
    ax.scatter(trajectory[:, 0], trajectory[:, 1], color='purple', s=50, zorder=2)

    # Plot current context vector
    current_context = context_vec_history[frame_index]
    ax.arrow(0, 0, current_context[0], current_context[1], head_width=0.04, head_length=0.04, fc='purple', ec='purple', linewidth=2.5, zorder=3)

    # Set plot limits and labels
    ax.set_xlim(-0.3, 1.0)
    ax.set_ylim(-0.4, 0.6)
    ax.set_xlabel("x-dimension", fontsize=12)
    ax.set_ylabel("y-dimension", fontsize=12)

    # Use the provided title or generate a default one
    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(f"Iteration {frame_index + 1}: Next Token -> '{seq_snapshots[frame_index][-1]}'", fontsize=16)

    ax.grid(False) # Turn off the grid
    ax.set_aspect('equal', adjustable='box')


# 7. Create and display the animation
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
def update_animation(frame):
    draw_frame(ax_anim, frame)

ani = FuncAnimation(fig_anim, update_animation, frames=len(context_vec_history), blit=False, repeat=False)
display(HTML(ani.to_jshtml()))
plt.close(fig_anim) # Close the animation plot object

# 8. Generate and display the static plots

# Find the frame right before the first 'D' is generated
tip_to_d_frame_index = -1
for i, seq in enumerate(seq_snapshots):
    if seq[-1] == 'D':
        tip_to_d_frame_index = i
        break

if tip_to_d_frame_index != -1:
    # Plot 1: State just before the tip to 'D'
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    title1 = f"State Just Before Tip to Harmful ('D')\n(Iteration {tip_to_d_frame_index + 1})"
    draw_frame(ax1, tip_to_d_frame_index, title=title1)
    plt.show()

# Plot 2: Final state of the simulation
fig2, ax2 = plt.subplots(figsize=(8, 8))
final_frame_index = len(context_vec_history) - 1
title2 = f"Final State (Iteration {final_frame_index + 1})"
draw_frame(ax2, final_frame_index, title=title2)
plt.show()