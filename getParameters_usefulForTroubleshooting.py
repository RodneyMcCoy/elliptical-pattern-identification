# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 08:23:44 2021

@author: Malachi
"""

#This code is troubleshooting from getParameters
"""
        ########################################PLOTTING FOR TROUBLESHOOTING##################################################
        #plots all micro-hodographs for a single flight
        uvPlot = plt.figure("Troubleshooting", figsize=(10, 5))
        
        ax = uvPlot.add_subplot(1,2,1, aspect='equal')
        ax.plot(self.u, self.v, 'red') 

        #plot parametric best fit ellipse
        param = np.linspace(0, 2 * np.pi)
        x = self.a * np.cos(param) * np.cos(self.phi) - self.b * np.sin(param) * np.sin(self.phi) + self.c_x
        y = self.a * np.cos(param) * np.sin(self.phi) + self.b * np.sin(param) * np.cos(self.phi) + self.c_y
        ax.plot(x, y)
        ax.set_xlabel("(m/s)")
        ax.set_ylabel("(m/s)")
        ax.set_aspect('equal')
        ax.set_title("UV with fit ellipse")
        ax.grid()
        #plot uvRot
        ax.plot(urot, uvrot[1,:])
        

        #plot u, t, vs alt
        color = 'tab:red'
        ax = uvPlot.add_subplot(1,2,2, )
        ax.set_xlabel('urot (m/s)', color=color)
        ax.set_ylabel('alt (m)')
        ax.plot(urot, self.alt, label='urot', color=color)
        ax.plot(self.temp, self.alt, label='temp', color = 'green')
        ax.tick_params(axis='x', labelcolor=color)

        ax2 = ax.twiny()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_xlabel('dtdz', color=color)  # we already handled the y-label with ax1
        ax2.plot(dTdz, self.alt[0:-1], color=color, label='dTdz')
        ax2.tick_params(axis='x', labelcolor=color)

        ax.legend()  
            
        plt.show() 


        #########################################END PLOTTING FOR TROUBLESHOOTING#############################################