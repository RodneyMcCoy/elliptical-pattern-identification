import matplotlib.pyplot as plt
import numpy as np
hw=.05
lw=1
axcolor = 'black'
ellipcolor = 'black'
rotcolor = 'grey'
u_par = 1
offset = .05
fntsze = 12
# ellipse parameters
xc = 0
yc = 0
a = .75
b = a/2
theta = np.pi/5
param = np.linspace(0, 2 * np.pi)
x = a * np.cos(param) * np.cos(theta) - b * np.sin(param) * np.sin(theta) + xc
y = a * np.cos(param) * np.sin(theta) + b * np.sin(param) * np.cos(theta) + yc

fig,ax = plt.subplots(dpi=300)
fig.set_size_inches(3.54,3.54)
ax.set_aspect('equal')
ax.axis('off')
# ellipse
ax.plot(x,y, color=ellipcolor)

# u / v axis
ax.arrow(0,0,1,0)
ax.arrow(0,0,1,0, length_includes_head=True, head_width=hw, lw=lw, color=axcolor)
ax.arrow(0,0,0,1, length_includes_head=True, head_width=hw, lw=lw, color=axcolor)

# u_parallel / v_perp axis
ax.arrow(0,0,u_par*np.cos(theta), u_par * np.sin(theta), head_width=hw*.75, lw=lw*.75, color=rotcolor)
ax.arrow(0,0,u_par*np.cos(theta+np.pi/2), u_par * np.sin(theta+np.pi/2), head_width=hw*.75, lw=lw*.75, color=rotcolor)

# labels
plt.text(1,0,'$+u$', fontsize=fntsze)
plt.text(0,1, '$+v$', fontsize=fntsze)
plt.text(u_par*np.cos(theta)+offset, u_par * np.sin(theta)+offset, '$u_\parallel$')
plt.text(u_par*np.cos(theta+np.pi/2), u_par * np.sin(theta+np.pi/2), '$v_\perp$')

plt.xlim([-.65, 1.1])
plt.ylim([-.65, 1])
#remove ticks
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.0,
wspace=0.0)
plt.tight_layout(pad=0.05)

plt.savefig('rotatedCoordinates.eps', format='eps')
plt.show()