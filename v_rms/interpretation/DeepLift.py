# Captum: DeepLift
import captum
from captum.attr import DeepLift

method_name = "DeepLift"
# Store the gradient values for all inputs in test loader
deep_lift = np.zeros((3674, 9), dtype=np.float32)

i = -1
method = DeepLift(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = method.attribute(input)
    deep_lift[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    deep_lift[:,i] = np.abs(deep_lift[:,i])
    method_avg[:,i]   = np.mean(deep_lift[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_" + method_name)
plt.show()
