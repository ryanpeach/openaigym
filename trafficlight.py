from Tkinter import *

def addv(var, v):
    var.set(var.get()+v)

MANUAL, AUTO = 0., 1.
RED, YELLOW, GREEN = 0., 1., 2.
class FourWayStop:
    def __init__(self):
        # Initialize TK
        window = Tk()
        window.title("Traffic Light")
        self.frame = Frame(window)
        self.frame.pack()
        self.canvas = Canvas(window, width=500, height=300)
        self.canvas.pack()

        # Create T and it's label
        self.T = DoubleVar()
        self.Tstring = StringVar()
        self.Tlabel = Label(self.frame, textvariable=self.Tstring)
        self.Tlabel.grid(row = 4, column = 2)
        self.T.trace('w', self.updateT)
        self.T.set(0)

        # Create two colors
        self.color1 = DoubleVar()
        self.color2 = DoubleVar()
        self.color1.trace('w', self.color_change1)
        self.color2.trace('w', self.color_change2)

        # Create six indicators, 3 for each color
        self.red1 = self.canvas.create_oval(10, 10, 110, 110, fill="white")
        self.yellow1 = self.canvas.create_oval(120, 10, 220, 110, fill="white")
        self.green1 = self.canvas.create_oval(230, 10, 330, 110, fill="white")
        self.red2 = self.canvas.create_oval(10, 110, 110, 210, fill="white")
        self.yellow2 = self.canvas.create_oval(120, 110, 220, 210, fill="white")
        self.green2 = self.canvas.create_oval(230, 110, 330, 210, fill="white")

        # Create Modes Auto and Manual
        self.mode = DoubleVar()
        self.mode.trace('w', self.mode_change)

        # One button for each
        self.manual_b = Button(self.frame, text="Manual", command=self.set_manual)
        self.manual_b.grid(row=1, column=1)
        self.auto_b = Button(self.frame, text="Auto", command=self.set_auto)
        self.auto_b.grid(row=1, column=2)
        self.mode.set(MANUAL)

        # Create Manual buttons
        self.red_b1 = Button(self.frame, text="Red", command=self.manual_red1)
        self.red_b1.grid(row = 2, column = 1)
        self.yellow_b1 = Button(self.frame, text="Yellow", command=self.manual_yellow1)
        self.yellow_b1.grid(row = 2, column = 2)
        self.green_b1 = Button(self.frame, text="Green", command=self.manual_green1)
        self.green_b1.grid(row = 2, column = 3)

        self.red_b2 = Button(self.frame, text="Red", command=self.manual_red2)
        self.red_b2.grid(row = 3, column = 1)
        self.yellow_b2 = Button(self.frame, text="Yellow", command=self.manual_yellow2)
        self.yellow_b2.grid(row = 3, column = 2)
        self.green_b2 = Button(self.frame, text="Green", command=self.manual_green2)
        self.green_b2.grid(row = 3, column = 3)
        self.color1.set(RED)
        self.color2.set(GREEN)

        self.iterate_b = Button(self.frame, text="Next", command=self.next)
        self.iterate_b.grid(row = 4, column = 1)

        # Cars and pedestrians
        self.cars1 = DoubleVar()
        self.cars2 = DoubleVar()
        self.ped1 = DoubleVar()
        self.ped2 = DoubleVar()
        self.cars1_string = StringVar()
        self.cars2_string = StringVar()
        self.ped1_string = StringVar()
        self.ped2_string = StringVar()
        self.cars1_label = Label(self.frame, textvariable=self.cars1_string)
        self.cars1_label.grid(row = 5, column = 1)
        self.cars2_label = Label(self.frame, textvariable=self.cars2_string)
        self.cars2_label.grid(row = 5, column = 2)
        self.ped1_label = Label(self.frame, textvariable=self.ped1_string)
        self.ped1_label.grid(row = 5, column = 3)
        self.ped2_label = Label(self.frame, textvariable=self.ped2_string)
        self.ped2_label.grid(row = 5, column = 4)
        self.cars1.trace('w', self.cars1_update)
        self.cars2.trace('w', self.cars2_update)
        self.ped1.trace('w', self.ped1_update)
        self.ped2.trace('w', self.ped2_update)
        self.cars1.set(0)
        self.cars2.set(0)
        self.ped1.set(0)
        self.ped2.set(0)

        # Green Time and Yellow Time
        self.greenstart = 0
        self.yellowstart = -1
        self.GREENTIME1 = DoubleVar()
        self.GREENTIME1_string = StringVar()
        self.GREENTIME1.set(3)
        self.GREENTIME1_string.set("3")
        self.GREENTIME1_label = Label(self.frame, text="Green Time1:")
        self.GREENTIME1_label.grid(row = 1, column = 7)
        self.GREENTIME1_entry = Entry(self.frame, textvariable=self.GREENTIME1_string)
        self.GREENTIME1_entry.grid(row = 1, column = 8)
        self.GREENTIME1_string.trace('w', self.GREENTIME1_update)

        self.YELLOWTIME1 = DoubleVar()
        self.YELLOWTIME1_string = StringVar()
        self.YELLOWTIME1.set(1)
        self.YELLOWTIME1_string.set("1")
        self.YELLOWTIME1_label = Label(self.frame, text="Yellow Time1:")
        self.YELLOWTIME1_label.grid(row = 2, column = 7)
        self.YELLOWTIME1_entry = Entry(self.frame, textvariable=self.YELLOWTIME1_string)
        self.YELLOWTIME1_entry.grid(row = 2, column = 8)
        self.YELLOWTIME1_string.trace('w', self.YELLOWTIME1_update)

        self.GREENTIME2 = DoubleVar()
        self.GREENTIME2_string = StringVar()
        self.GREENTIME2.set(3)
        self.GREENTIME2_string.set("3")
        self.GREENTIME2_label = Label(self.frame, text="Green Time2:")
        self.GREENTIME2_label.grid(row = 3, column = 7)
        self.GREENTIME2_entry = Entry(self.frame, textvariable=self.GREENTIME2_string)
        self.GREENTIME2_entry.grid(row = 3, column = 8)
        self.GREENTIME2_string.trace('w', self.GREENTIME2_update)

        self.YELLOWTIME2 = DoubleVar()
        self.YELLOWTIME2_string = StringVar()
        self.YELLOWTIME2.set(1)
        self.YELLOWTIME2_string.set("1")
        self.YELLOWTIME2_label = Label(self.frame, text="Yellow Time2:")
        self.YELLOWTIME2_label.grid(row = 4, column = 7)
        self.YELLOWTIME2_entry = Entry(self.frame, textvariable=self.YELLOWTIME2_string)
        self.YELLOWTIME2_entry.grid(row = 4, column = 8)
        self.YELLOWTIME2_string.trace('w', self.YELLOWTIME2_update)

        #self.T_txt = Text(self.frame)
        #self.T_txt.grid(row = 4, column = 2)
        mainloop()

    # Auto and Manual Modes
    def mode_change(self, *args):
        if self.mode.get() == MANUAL:
            self.manual_b.config(relief=SUNKEN)
            self.auto_b.config(relief=RAISED)
        elif self.mode.get() == AUTO:
            self.auto_b.config(relief=SUNKEN)
            self.manual_b.config(relief=RAISED)

    def set_manual(self):
        self.nextT()
        self.mode.set(MANUAL)

    def set_auto(self):
        self.nextT()
        test1 = self.color1.get() != self.color2.get()
        test2 = self.color1.get() == RED or self.color2.get() == RED
        if test1 and test2:
            self.mode.set(AUTO)

    def _color_change(self, lr = True):
        if lr:
            color, red, yellow, green = self.color1.get(), self.red1, self.yellow1, self.green1
        else:
            color, red, yellow, green = self.color2.get(), self.red2, self.yellow2, self.green2
        if color == RED:
            self.canvas.itemconfig(red, fill="red")
            self.canvas.itemconfig(yellow, fill="white")
            self.canvas.itemconfig(green, fill="white")
        elif color == YELLOW:
            self.yellowstart = self.T.get()
            self.canvas.itemconfig(red, fill="white")
            self.canvas.itemconfig(yellow, fill="yellow")
            self.canvas.itemconfig(green, fill="white")
        elif color == GREEN:
            self.greenstart = self.T.get()
            self.canvas.itemconfig(red, fill="white")
            self.canvas.itemconfig(yellow, fill="white")
            self.canvas.itemconfig(green, fill="green")
        else:
            raise KeyError

    def color_change1(self, *args):
        self._color_change(True)
    def color_change2(self, *args):
        self._color_change(False)

    # Color changing in manual mode
    def _manual_color(self, lr, color):
        self.nextT()
        if self.mode.get() == MANUAL:
            if lr:
                self.color1.set(color)
            else:
                self.color2.set(color)
    def manual_red1(self):
        self._manual_color(True,RED)
    def manual_yellow1(self):
        self._manual_color(True,YELLOW)
    def manual_green1(self):
        self._manual_color(True,GREEN)
    def manual_red2(self):
        self._manual_color(False,RED)
    def manual_yellow2(self):
        self._manual_color(False,YELLOW)
    def manual_green2(self):
        self._manual_color(False,GREEN)
    def auto_color(self):
        T = int(self.T.get())
        if self.mode.get() == AUTO:
            if self.color1.get() == GREEN and (self.greenstart - T + self.GREENTIME1.get()) <= 0:
                self.color1.set(YELLOW)
            elif self.color1.get() == YELLOW and (self.yellowstart - T + self.YELLOWTIME1.get()) <= 0:
                self.color1.set(RED)
                self.color2.set(GREEN)
            elif self.color2.get() == GREEN and (self.greenstart - T + self.GREENTIME2.get()) <= 0:
                self.color2.set(YELLOW)
            elif self.color2.get() == YELLOW and (self.yellowstart - T + self.YELLOWTIME2.get()) <= 0:
                self.color2.set(RED)
                self.color1.set(GREEN)

    # Cars and pedestrians
    def cars1_update(self, *args):
        self.cars1_string.set("Cars1: {}".format(self.cars1.get()))
    def cars2_update(self, *args):
        self.cars2_string.set("Cars2: {}".format(self.cars2.get()))
    def ped1_update(self, *args):
        self.ped1_string.set("Ped1: {}".format(self.ped1.get()))
    def ped2_update(self, *args):
        self.ped2_string.set("Ped2: {}".format(self.ped2.get()))

    # Green and Yellow time
    def GREENTIME1_update(self, *args):
        if self.mode.get() == MANUAL:
            try:
                v = float(self.GREENTIME1_string.get())
            #except:
            #    pass
            finally:
                self.GREENTIME1.set(v)
    def YELLOWTIME1_update(self, *args):
        if self.mode.get() == MANUAL:
            try:
                v = float(self.YELLOWTIME1_string.get())
            #except:
            #    pass
            finally:
                self.YELLOWTIME1.set(v)
    def GREENTIME2_update(self, *args):
        if self.mode.get() == MANUAL:
            try:
                v = float(self.GREENTIME2_string.get())
            #except:
            #    pass
            finally:
                self.GREENTIME2.set(v)
    def YELLOWTIME2_update(self, *args):
        if self.mode.get() == MANUAL:
            try:
                v = float(self.YELLOWTIME2_string.get())
            #except:
            #    pass
            finally:
                self.YELLOWTIME2.set(v)

    # Time incrementing
    def next(self):
        self.nextT()
        self.auto_color()
        self.canvas.itemconfig(self.iterate_b, text="Next, T={}".format(self.T))

    def nextT(self):
        addv(self.T, 1)
        if self.color1.get() == RED:
            addv(self.cars1, 1)
            if self.ped2.get() > 0: addv(self.ped2, -1)
        if self.color2.get() == RED:
            addv(self.cars2, 1)
            if self.ped1.get() > 0: addv(self.ped1, -1)
        if self.color1.get() in [GREEN, YELLOW]:
            if self.cars1.get() > 0:addv(self.cars1, -1)
            addv(self.ped2, 1)
        if self.color2.get() in [GREEN, YELLOW]:
            if self.cars2.get() > 0:addv(self.cars2, -1)
            addv(self.ped1, 1)

    def updateT(self, *args):
        self.Tstring.set("T: {}".format(self.T.get()))

if __name__=="__main__":
    FourWayStop()
