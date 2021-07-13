# -*- coding: utf-8 -*-
# pylint: disable=E0012, fixme, invalid-name, no-member, R0913, W0613, W0622, E0611, W0603, R0902, R0903, C0301
"""  Term 2, Assignment 5

Assignment Tasks: 8

Aim:
    GUI to visualise electric field lines for a collection of point charges.

    To remove a charge:
        Left-click on it and release.

    To add a charge:
        Left-click and hold, move mouse left or right reading charge in status
        message and release. Adds the charge of displayed magnitude at the initial
        mouse press position.

    Drag a charge:
        Left-click on a charge and drag it

Restrictions:
    Do not change anything outside TODO blocks.
    Do not use import.
    Do not add pylint directives.

Guidance:
    Please read the introduction page to this assignment on Canvas.

Author of the template:
    Wolfgang Theis
    School of Physics and Astronomy
    University of Birmingham
"""

import sys
import numpy as np
from scipy.integrate import odeint as odeint
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class Charges:
    ''' A collection of point charges in the xy-plane

    distances, charges, electric fields, and potentials are treated as unitless

    The electric field due to a charge q at distance d is taken to be
    E = q/d**2

    The potential V is given by dV/dr = -E

    '''

    def __init__(self):
        # private attributes
        # self._q is a 1d array holding the value of the charge for the different charges
        # charege n has a charge of self._q[n]
        self._q = np.zeros((0, ))
        # self._pos is a 2d array with xy coordinates of the charges
        # x position of charge n is self._pos[n, 0]
        # y position of charge n is self._pos[n, 1]
        self._pos = np.zeros((0, 2))

    def add_charge(self, q, xy):
        ''' add charge of magnitude q at locations x,y = xy '''
        self._q = np.hstack([self._q, q])
        self._pos = np.vstack([self._pos, np.array([xy])])

    def delete_charge(self, k):
        ''' delete charge k '''
        if k >= 0 and k < self._q.shape[0]:
            self._q = np.delete(self._q, k)
            self._pos = np.delete(self._pos, k, 0)

    def set_position(self, k, xy):
        ''' set position of charge k to xy '''
        if k >= 0 and k < self._q.shape[0]:
            self._pos[k, :] = xy

    def get_charges(self):
        ''' provide list of (charge, position) tuples '''
        p = self._pos
        return [(q, p[k, :]) for k, q in enumerate(self._q)]

    def get_closest(self, xy, limit=0.1):
        ''' determine the index of the closest charge

        Parameters
        ----------
        xy: array-like
            x,y pair of position to find the closest charge to.

        limit:
            maximum distance to find a charge in.

        Returns
        -------
        index: int
            index of the closest charge
            If no charge is within the limit then None is returned.
        '''
        # TODO: Assignment Task 1: write function body
        try:
            r = xy-self._pos #distance variable
            somma = np.sum(r**2, axis=1)
            min_value = np.min(somma)
            if min_value <= limit**2: #check limit condition and, if respected, write the index
                index = np.argmin(somma)
            else:
                index = None
        except ValueError: #avoid error when all the charges are gone and mouse is pressed
            index = None
        return index
        # End of Task 1; proceed to task 2.

    def scaled_electric_field(self, xy, _):
        ''' calculate suitably scaled electric field vector at position xy.

        The scaling is such that integrating this scaled field along field lines
        Es(lambda) dlambda from lamda=0 to lamda_max transverses roughly a
        modulus of the potential energy difference of lambda_max.

        To avoid numerical instabilities it should be ensured that the return value
        is not divergent by ensuring that the scaling factor remains finite.
        If you had a pure scaling factor of 1/x where x could be zero,
        you would use 1/(x+0.0001) instead]

        Parameters
        ----------
        xy: array-like
            x,y pair of position at which the scaled electric field is requested

        _:
            not used but required place holder for ode integrator

        Returns
        -------
        ef: array-like
            scaled electric field Es = (s*Ex, s*Ey) at position xy

        '''
        # TODO: Assignment Task 2: write function body
        r = xy-self._pos
        electricfield = np.sum((self._q*(np.sum(r**2, axis=1)**(-3/2)))[:, np.newaxis]*r, axis=0)
        ef = electricfield/(np.sum(electricfield**2)+0.00005) #scale ef to unit vector
        return ef
        # End of Task 2; proceed to task 3.

    def field_lines(self, nr_of_fieldlines=32, start_radius=0.2, lambda_max=10, points=801):
        ''' calculate the field lines which should include one at pi/4, rather than 0

            Parameters
            ----------

            nr_of_fieldlines: int
                number of field lines originating from each positive charge

            start_radius: float
                radius around each positive charge at which the field lines start

            lambda_max: float
                the maximum value of the parameter for the parametric representation
                of the fieldlines x(lambda), y(lambda), lambda =[0,..., lambda_max]

            points: int
                the number of points in each fieldline

            Returns
            -------

            fieldlines: list of 2-d numpy arrays
                each element of the list is an array of shape (points, 2) with
                fieldlines[k][:, 0] and fieldlines[k][:, 1] providing
                the x and y values, respectively of the k-th fieldline
        '''
        # TODO: Assignment Task 3: write function body
        lambda_1 = np.linspace(0, lambda_max, points)
        angles = np.linspace(np.pi/nr_of_fieldlines, 2*np.pi + np.pi/nr_of_fieldlines, nr_of_fieldlines, endpoint=False)
        fieldlines = []
        for q, posit in self.get_charges():
            if q > 0: #plot only for positive charges since fieldlines (lines from positive to negative)
                for angle in angles:
                    lines_list = odeint(self.scaled_electric_field, posit+(start_radius*np.cos(angle), start_radius*np.sin(angle)), lambda_1*q)
                    #scaling factor of q, depends on each positive charge
                    fieldlines.append(lines_list)
        return fieldlines
        # End of Task 3; proceed to task 4.


class MyMainWindow(QMainWindow):
    ''' the main window potentially with menus, statusbar, ... '''

    def __init__(self):
        super().__init__()
        self.resize(500, 500)
        self.move(400, 300)
        central_widget = MyCentralWidget(self)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('PyQt widget with matplotlib figure')
        self.statusBar().showMessage('Waiting for mouse move or mouse click')


class MyCentralWidget(QWidget):
    ''' everything in the main area of the main window '''

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        # define figure canvas
        self.mpl_widget = MyMplWidget(self.main_window)
        # place MplWidget into a vertical box layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.mpl_widget)  # add the figure
        # use the box layout to fill the window
        self.setLayout(vbox)


class MyMplWidget(FigureCanvas):
    ''' both a QWidget and a matplotlib figure '''

    def __init__(self, main_window, parent=None, figsize=(4, 4), dpi=100):
        self.main_window = main_window
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.mouse_pressed_pos = None
        self.closest_k = None
        self.dragging = False
        self.qadd = 0
        self.lines = []
        self.points = []
        self.field_lines_args = None            # used to save the parameters for use in drag_replt
        self.charges = Charges()
        # add some charges to start with an example
        self.charges.add_charge(1, (1, 0))
        self.charges.add_charge(1, (-1, 0))
        self.charges.add_charge(-1, (0, 1))
        self.charges.add_charge(-1, (0, -1))
        self.plot_fieldlines()
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def plot_fieldlines(self, nr_of_fieldlines=32, start_radius=0.2, lambda_max=10, points=801):
        ''' clear figure and plot fieldlines and charges '''
        self.fig.clf()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.lines = []         # list of matplotlib lines in the plot showing the fieldlines (for drag_replot)
        self.points = []        # list of matplotlib lines in the plot showing the charges (for drag_replot)
        self.field_lines_args = (nr_of_fieldlines, start_radius, lambda_max, points)
        # TODO: Assignment Task 4: calculate and plot field lines; plot charges; collect lines and points
        fieldlines = self.charges.field_lines(*self.field_lines_args) #get the fieldline list, then draw these and add them to a list
        for linefield in fieldlines:
            l, = self.ax.plot(linefield[:, 0], linefield[:, 1], 'k')
            self.lines.append(l)
        list_of_charges = self.charges.get_charges()
        for i, xy in list_of_charges:
            Area = 13*np.abs(i) #volume coeff for area of the charges. Arbitrary but effective
            if i > 0:
                c1, = self.ax.plot(xy[0], xy[1], 'bo', markersize=Area)
                self.points.append(c1)
            if i < 0:
                c2, = self.ax.plot(xy[0], xy[1], 'ro', markersize=Area)
                self.points.append(c2)
        # End of Task 4; proceed to task 5.
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.draw()

    def drag_replot(self):
        ''' replot elements in self.lines and self.point
            after adjusting their xdata and ydata arrays
            to reflect the new position of the dragged charge.
        '''
        self.ax.draw_artist(self.ax.patch)                              # <-- redraw the plotting area of the axis
        # TODO: Assignment Task 5: redraw updated field lines and charges
        data = self.charges.field_lines(*self.field_lines_args)#import fieldlines with arguments, then rewrite lists
        for i, line in enumerate(self.lines):
            line.set_xdata(data[i][:, 0])
            line.set_ydata(data[i][:, 1])
        charges = self.charges.get_charges()
        for i, point in enumerate(self.points):
            point.set_xdata(charges[i][1][0])
            point.set_ydata(charges[i][1][1])
        self.draw()
        # End of Task 5; proceed to task 6.
        self.fig.canvas.update()                                        # <-- update the figure
        self.fig.canvas.flush_events()                                  # <-- ensure all draw requests are sent out

    def on_mouse_move(self, event):
        ''' add charge or drag

            self.dragging determines whether a charge is being dragged and is
            updated accordingly.

            If a charge is being added then self.qadd, the value of the charge
            is updated and the current value displayed in the statusbar.

            If a charge is being dragged then its position is updated in the
            self.charges object and the charges and fieldlines are displayed using
            drag_replot().

        '''
        # TODO: Assignment Task 6: write function body
        if event.xdata is not None and event.ydata is not None: #print mouse coordinates
            msg = f'x={event.xdata:.2f}, y={event.ydata:.2f}'
            self.main_window.statusBar().showMessage(msg)
        if self.mouse_pressed_pos is not None: #check if values acceptable, then correct coordinates
            if event.xdata is None:
                event.xdata = 0
            if event.ydata is None:
                event.ydata = 0
            self.qadd = event.xdata - np.asarray(self.mouse_pressed_pos)[0]
            if self.qadd != 0 and self.closest_k is None: #add charge
                msg = f'Charge={self.qadd:.2f}'
                self.main_window.statusBar().showMessage(msg)
            elif self.qadd != 0 and self.dragging is True: #drag charge
                self.charges.set_position(self.closest_k, [event.xdata, event.ydata])
                self.plot_fieldlines(8)
                self.drag_replot()
        # End of Task 5; proceed to task 6.

    def on_mouse_press(self, event):
        ''' set self.mouse_pressed_pos and self.closest_k, the
            xy pair for the position the mouse was clicked at
            and the index of of the charge closest to that position, respectively.
        '''
        # TODO: Assignment Task 7: write function body
        if event.xdata and event.ydata is not None:
            self.mouse_pressed_pos = (event.xdata, event.ydata)
            self.closest_k = self.charges.get_closest(self.mouse_pressed_pos)
            #set private flags
            if self.closest_k is not None:
                self.dragging = True
            else:
                self.qadd = 1
        # End of Task 7; proceed to task 7.

    def on_mouse_release(self, event):
        ''' perform required actions when the mouse button is released

        If a charge should be deleted, delete the charge.
        If a charge should be added, add the charge.

        In all cases, redraw the figure with 32 fieldlines per charge
        and reset attributes as appropriate.
        '''
        # TODO: Assignment Task 8: write function body
        if event.xdata and event.ydata is not None: #correct the values if not acceptable
            position = np.asarray(self.mouse_pressed_pos) #import initial mouse pos as array, then check if it is acceptable
            if self.mouse_pressed_pos is None:
                position = np.zeros(2)
            self.qadd = event.xdata - position[0]
            if self.qadd != 0 and self.closest_k is None:
                self.charges.add_charge(self.qadd, position)
                msg = f'Charge value={self.qadd:.2f}, created at x={position[0]:.2f}, y={position[1]:.2f}'
                self.main_window.statusBar().clearMessage()
                self.main_window.statusBar().showMessage(msg)
                print(msg) #console debug
            elif self.qadd == 0 and self.closest_k is not None:
                del_pos = self.charges.get_charges()[self.closest_k][1]
                self.charges.delete_charge(self.closest_k)
                msg = f'Charge deleted at x={del_pos[0]:.2f}, y={del_pos[1]:.2f}'
                self.main_window.statusBar().clearMessage()
                self.main_window.statusBar().showMessage(msg)
                print(msg)
        #reset all the flags
        self.dragging = False
        self.qadd = 0
        self.mouse_pressed_pos = None
        self.plot_fieldlines(32)
        # End of Task 8; no further tasks


app = None


# pylint: disable=C0111
def main():
    global app
    app = QApplication(sys.argv)
    w = MyMainWindow()
    w.show()
    app.exec()


if __name__ == '__main__':
    main()
