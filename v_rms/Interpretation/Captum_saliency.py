# Get the average saliency using Captum
import captum
from captum.attr import Saliency

# Store the saliency values for all inputs in test loader
saliency_vrms = np.zeros((367, 9), dtype=np.float32)

i = -1
saliency = Saliency(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = saliency.attribute(input)
    saliency_vrms[i] = attribution.cpu().numpy()
    
# take abs value of each column and take average saliency for each property
saliency_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    saliency_vrms[:,i] = np.abs(saliency_vrms[:,i])
    saliency_avg[:,i]   = np.mean(saliency_vrms[:,i])
    
print(saliency_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, saliency_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title("NN: V_RMS (Captum)")
plt.savefig("Saliency_v_rms_captum")
plt.show()
