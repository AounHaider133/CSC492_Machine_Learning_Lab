import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
labels = ['A', 'B', 'C', 'D', 'E']
marker_shapes = ['o', 's', 'v', '*', 'x']  # Different marker shapes for each data point

# Create scatter plot with different marker shapes
for i in range(len(x)):
    plt.scatter(x[i], y[i], marker=marker_shapes[i])

# Add labels to the data points
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Set x-axis and y-axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Set title
plt.title('Scatter Plot with Different Marker Shapes')

# Display the plot
plt.show()
