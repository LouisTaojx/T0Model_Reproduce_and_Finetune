import matplotlib.pyplot as plt

# Dataset names
datasets = ["COPA", "WSC.fixed", "Winogrande-XL"]

# Accuracy scores for T5-small (fine-tuned) and T0_3B (zero-shot)
t5_small = [0.6047, 0.6154, 0.5785]
t0_3b = [0.7, 0.6731, 0.5249]

# Bar settings
x = range(len(datasets))
width = 0.35

# Plotting
fig, ax = plt.subplots()
bars1 = ax.bar([i - width/2 for i in x], t5_small, width, label='T5-small' )
bars2 = ax.bar([i + width/2 for i in x], t0_3b, width, label='T0_3B', color='orange', alpha=0.75)

# Labels and formatting
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison: T5-small vs. T0_3B')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 0.8)
ax.legend()

plt.tight_layout()
plt.savefig("accuracy_comparison_t5_t0.png")
plt.show()
