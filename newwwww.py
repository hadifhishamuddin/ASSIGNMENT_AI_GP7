# from rectpack import newPacker

# rectangles = [(100, 30), (40, 60), (30, 30),(70, 70), (100, 50), (30, 30)]
# bins = [(300, 450), (80, 40), (200, 150)]

# packer = newPacker()

# # Add the rectangles to packing queue
# for r in rectangles:
# 	packer.add_rect(*r)

# # Add the bins where the rectangles will be placed
# for b in bins:
# 	packer.add_bin(*b)

# # Start packing
# packer.pack()

# # Obtain number of bins used for packing
# nbins = len(packer)

# # Index first bin
# abin = packer[0]

# # Bin dimmensions (bins can be reordered during packing)
# width, height = abin.width, abin.height

# # Number of rectangles packed into first bin
# nrect = len(packer[0])

# # Second bin first rectangle
# rect = packer[1][0]

# # rect is a Rectangle object
# x = rect.x # rectangle bottom-left x coordinate
# y = rect.y # rectangle bottom-left y coordinate
# w = rect.width
# h = rect.height

# for abin in packer:
#   print(abin.bid) # Bin id if it has one
#   for rect in abin:
#     print(rect)

# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt

# creating initial data values
# of x and y
x = np.linspace(0, 10, 100)
y = np.sin(x)

# to run GUI event loop
plt.ion()

# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)

# setting title
plt.title("Geeks For Geeks", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Loop
for _ in range(50):
	# creating new Y values
	new_y = np.sin(x-0.5*_)

	# updating data values
	line1.set_xdata(x)
	line1.set_ydata(new_y)

	# drawing updated values
	figure.canvas.draw()

	# This will run the GUI event
	# loop until all UI events
	# currently waiting have been processed
	figure.canvas.flush_events()

	time.sleep(0.1)
