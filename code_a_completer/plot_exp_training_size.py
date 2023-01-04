import numpy as np
import matplotlib.pyplot as plt

# Charger les résultats de l'expérience depuis le fichier .npz
results = np.load('/home/baba/ProjetPythonM2/code_a_completer/ErrorAndTime.npz',allow_pickle=True)

results.keys()
valid_errors = results['valid_error']
train_errors = results['train_error']
train_sizes = results['N']
learning_time = results['learning_time']
methode = list(results['methode'])



# Set up the plot
fig, ax = plt.subplots()
# Plot the validation errors for each method
for i, method in enumerate(methode):
    ax.plot(train_sizes, valid_errors[:,i], label=method)

# Plot the training errors for each method in dashed lines
for i, method in enumerate(methode):
    ax.plot(train_sizes, train_errors[:,i], '--', label=method)

# Add a title, axis labels, and a legend
# Add a title, axis labels, and a legend
ax.set_title('Prediction Performance as a Function of Training Set Size')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Error')
ax.legend()

# Save the figure to a .png file
#La premiere figure est enregistree dans performance.png
plt.savefig('performance.png')





# Plot the computation times for each method
for i, method in enumerate(methode):
    plt.plot(train_sizes, learning_time[:,i], '--', label=method)

# Add a legend, title, and labels to the axes
plt.legend()
# Set the x-axis scale to logarithmic
plt.xscale('log')

# Set the y-axis scale to logarithmic
plt.yscale('log')

# Add a title
plt.title('Computation times as a function of training set size')

# Add x-axis and y-axis labels
plt.xlabel('Training set size')
plt.ylabel('Computation time (seconds)')

# Save the figure to a .png file
plt.savefig('computation_times.png')

# Show the plot
plt.show()
