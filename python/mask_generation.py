import itertools
import matplotlib
import matplotlib.pyplot as plt
from libs.util import random_mask

# Plot the results
_, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = list(itertools.chain.from_iterable(axes))

for i in range(len(axes)):
    # Generate image
    img = random_mask(500, 500)

    # Plot image on axis
    axes[i].imshow(img * 255)


plt.show()

