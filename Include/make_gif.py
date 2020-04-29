from PIL import Image
import glob

# Create the frames
frames = []
imgs = sorted(glob.glob("./self-organizing networks/Plots/*"))
# imgs = sorted(glob.glob("./self-organizing networks/Plots_one_winner/*"))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('nauka.gif', format='GIF', save_all=True, duration=1, loop=0, append_images=frames[1:])
