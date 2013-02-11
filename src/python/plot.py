
"""
A library for plotting pretty images of all kinds.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt

                                                    
class InteractiveImshow(object):
    """
    A brief extension to matplotlib's imshow that puts a colorbar next to 
    the image that you can click on to scale the maximum numeric value
    displayed.
    
    Based on code from pyana_misc by Ingrid Ofte.
    
    Parameters
    ----------
    inarr : np.ndarray
        The array to imshow()
        
    filename : {str, None}
        The filename to call the file if it is saved. If `None`, disable saving
        ability.
    """
    
    def __init__(self, inarr, filename=None):
        """
        Parameters
        ----------
        inarr : np.ndarray
            The array to imshow()

        filename : {str, None}
            The filename to call the file if it is saved. If `None`, disable saving
            ability.
        """
        self.inarr = inarr
        self.filename = filename
        self.cmax = self.inarr.max()
        self.cmin = self.inarr.min()
        self._draw_img()
        

    def _on_keypress(self, event):
        if event.key == 's':
            logger.info("Saving image: %s" % self.filename)
            plt.savefig(self.filename)
        if event.key == 'r':
            colmin, colmax = self.orglims
            plt.clim(colmin, colmax)
            plt.draw()
            

    def _on_click(self, event):
        if event.inaxes:
            lims = self.axes.get_clim()
            colmin = lims[0]
            colmax = lims[1]
            rng = colmax - colmin
            value = colmin + event.ydata * rng
            if event.button is 1 :
                if value > colmin and value < colmax :
                    colmin = value
            elif event.button is 2 :
                colmin, colmax = self.orglims
            elif event.button is 3 :
                if value > colmin and value < colmax:
                    colmax = value
            plt.clim(colmin, colmax)
            plt.draw()
            

    def _draw_img(self):
        fig = plt.figure()
        cid1 = fig.canvas.mpl_connect('key_press_event', self._on_keypress)
        cid2 = fig.canvas.mpl_connect('button_press_event', self._on_click)
        canvas = fig.add_subplot(111)
        #canvas.set_title(self.filename)
        self.axes = plt.imshow(self.inarr.T, vmax = self.cmax, origin='lower')
        self.colbar = plt.colorbar(self.axes, pad=0.01)
        self.orglims = self.axes.get_clim()
        plt.show()