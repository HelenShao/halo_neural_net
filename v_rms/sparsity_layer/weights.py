# Print Weights of First Layer
model_layers = list(model.children())
print(model_layers[0].weight)

################################# Bar Graph for Weights ###############################
weights = model_layers[0].weight.detach().numpy()

fig, ax = plt.subplots(figsize=(11,5))

# Save the chart so we can loop through the bars below.
bars = ax.bar(height= weights, x=properties, width=0.5)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#868687')
ax.xaxis.grid(False)

plt.title("J: Weights of First Layer", fontsize="12")
plt.ylabel("Weight Values", fontsize="14")
plt.xlabel("Features", fontsize="14")

fig.tight_layout()
plt.savefig("J_Weights")
