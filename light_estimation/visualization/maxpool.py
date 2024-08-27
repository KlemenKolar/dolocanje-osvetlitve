import matplotlib.pyplot as plt
import numpy as np

# Define the input matrix
input_matrix = np.array([
    [2, 0, 1, 2],
    [1, 3, 0, 1],
    [0, 1, 4, 1],
    [1, 1, 2, 0]
])

# Define the output matrix
output_matrix = np.array([
    [3, 2],
    [1, 4]
])

# Colors
input_colors = [
    ['#FFA07A', '#FA8072', '#E9967A', '#F08080'],
    ['#CD5C5C', '#DC143C', '#B22222', '#8B0000'],
    ['#FF4500', '#FF6347', '#FF7F50', '#FF8C00'],
    ['#FFA500', '#FFD700', '#FFFF00', '#FFFFE0']
]

output_colors = [
    ['#98FB98', '#00FF7F'],
    ['#00FA9A', '#7CFC00']
]

# Plot the input matrix
fig, ax = plt.subplots()
for i in range(input_matrix.shape[0]):
    for j in range(input_matrix.shape[1]):
        ax.add_patch(plt.Rectangle((j, input_matrix.shape[0] - i - 1), 1, 1, fill=True, color=input_colors[i][j]))
        ax.text(j + 0.5, input_matrix.shape[0] - i - 1 + 0.5, input_matrix[i, j], ha='center', va='center', fontsize=12)
        
ax.set_xlim(0, input_matrix.shape[1])
ax.set_ylim(0, input_matrix.shape[0])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

# Plot the output matrix
for i in range(output_matrix.shape[0]):
    for j in range(output_matrix.shape[1]):
        ax.add_patch(plt.Rectangle((j + 5, output_matrix.shape[0] - i - 1), 1, 1, fill=True, color=output_colors[i][j]))
        ax.text(j + 5 + 0.5, output_matrix.shape[0] - i - 1 + 0.5, output_matrix[i, j], ha='center', va='center', fontsize=12)

ax.arrow(4, 2, 1, 0, head_width=0.5, head_length=0.5, fc='k', ec='k')

ax.text(2, -1, '4x4 Input', ha='center', va='center', fontsize=14)
ax.text(7, -1, '2x2 Output', ha='center', va='center', fontsize=14)
ax.text(4.5, 1, 'Max pool\n(2x2 Filters, Stride 2)', ha='center', va='center', fontsize=12)

plt.axis('off')
plt.show()
