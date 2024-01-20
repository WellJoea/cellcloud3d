import matplotlib.pyplot as plt

from cellcloud3d.utilis import take_data

def comp_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa, cmap='viridis', axis=0):
    
    # Create a figure with two subplots and the specified size.

    plt.subplots(1, 2, figsize=(10, 8))
    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(take_data(fixed_npa, fixed_image_z, axis), cmap=cmap)
    plt.title("fixed image")
    plt.axis("off")

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(take_data(moving_npa, moving_image_z, axis), cmap=cmap)
    plt.title("moving image")
    plt.axis("off")

    plt.show()

def imshows(image_z, images, axis=0, cmap=None, **kargs):
    img=take_data(images, image_z, axis)
    plt.imshow(img, cmap=cmap, **kargs)
    plt.axis("off")
    plt.show()

def comp_images_alpha(image_z, alpha, fixed, moving, cmap=plt.cm.Greys_r, axis=0):
    ifix=take_data(fixed, image_z, axis)
    imov=take_data(moving, image_z, axis)
    img = (1.0 - alpha) * ifix + alpha * imov
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()
    
# interact(
#     comp_images_alpha,
#     image_z=(0, transformed.shape[2] - 1),
#     alpha=(0.0, 1.0, 0.05),
#     axis=2,
#     fixed=fixed(static),
#     moving=fixed(transformed),
# );

# interact(
#     comp_images,
#     fixed_image_z=(0, static.shape[2] - 1),
#     moving_image_z=(0, transformed.shape[2] - 1),
#     axis=(0, 2),
#     fixed_npa=fixed(static),
#     moving_npa=fixed(transformed),

# );