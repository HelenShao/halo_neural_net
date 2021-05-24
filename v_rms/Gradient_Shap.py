# Captum: GradientShap
import captum
from captum.attr import GradientShap

method_name = "GradientShap"
# Store the gradient values for all inputs in test loader
grad_shap = np.zeros((3674, 9), dtype=np.float32)
baselines = torch.randn([1,9])

i = -1
method = GradientShap(model)
for input, output in test_loader:
    input = input.to(device)
    i += 1
    attribution = method.attribute(input, baselines)
    grad_shap[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    grad_shap[:,i] = np.abs(grad_shap[:,i])
    method_avg[:,i]   = np.mean(grad_shap[:,i])
    
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
plt.savefig("VRMS_"+method_name)
plt.show()
