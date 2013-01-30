

                                                    
class img_class (object):
    """
    Imaging class copied from Ingrid Ofte's pyana_misc code                                                     
    """
    def __init__(self, inarr, filename):
        self.inarr = inarr*(inarr>0)
        for i in range(len(inarr)):
            self.inarr[i] = self.inarr[i][::-1]
        self.filename = filename
        self.cmax = self.inarr.max()
        self.cmin = self.inarr.min()

    def on_keypress(self,event):
        if event.key == 'p':
            if not os.path.exists(write_dir + runtag):
                os.mkdir(write_dir + runtag)
            pngtag = write_dir + runtag + "/%s.png" % (self.filename)
            print "saving image as " + pngtag
            P.savefig(pngtag)
        if event.key == 'r':
            colmin, colmax = self.orglims
            P.clim(colmin, colmax)
            P.draw()

    def on_click(self, event):
        if event.inaxes:
            lims = self.axes.get_clim()
            colmin = lims[0]
            colmax = lims[1]
            range = colmax - colmin
            value = colmin + event.ydata * range
            if event.button is 1 :
                if value > colmin and value < colmax :
                    colmin = value
            elif event.button is 2 :
                colmin, colmax = self.orglims
            elif event.button is 3 :
                if value > colmin and value < colmax:
                    colmax = value
            P.clim(colmin, colmax)
            P.draw()

    def draw_img(self):
        fig = P.figure()
        cid1 = fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
        canvas = fig.add_subplot(111)
        canvas.set_title(self.filename)
        self.axes = P.imshow(self.inarr, vmax = self.cmax, origin='lower')
        self.colbar = P.colorbar(self.axes, pad=0.01)
        self.orglims = self.axes.get_clim()
        P.show()