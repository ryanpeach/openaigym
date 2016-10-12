from Tkinter import *

def addv(var, v):
    var.set(var.get()+v)

MANUAL, AUTO = 0., 1.
RED, YELLOW, GREEN = 0., 1., 2.
TRAINING, WATCHING, CONTROL = 1., 2., 3.
class FourWayStop:
    def __init__(self):
        # Initialize TK
        window = Tk()
        window.title("Traffic Light")
        self.frame = Frame(window)
        self.frame.pack()
        self.canvas = Canvas(window, width=500, height=300)
        self.canvas.pack()

        def createButton(loc, text, command = None):
            b = Button(self.frame, text=text, command=command)
            b.grid(row=loc[0], column=loc[1])
            return b
            
        def createDoubleVar(writetrace = None):
            v = DoubleVar()
            v.trace('w', writetrace)
            return v
            
        def createLabel(loc, text):
            if isinstance(text, str):
                l = Label(self.frame, text = text)
            else:
                l = Label(self.frame, textvariable=text)
            l.grid(row=loc[0], col=loc[1])
            return l
            
        def createNumericLabel(loc, labelstr, default=-1.):
            v = DoubleVar()
            s = StringVar()
            def numeric_update(self):
                s.set("{}: {}".format(labelstr, v.get()))
            v.trace('w', numeric_update)
            l = createLabel(loc, s)
            v.set(default)
            return v, s, l
        
        def createNumericField(loc, text, update_func = None):
            v = DoubleVar()
            v.set(1)
            l = createLabel(loc=loc, text=text+": ")
            e = Entry(self.frame, textvariable=s)
            e.grid(row = loc[0], column = loc[1]+1)
            def update(self, *args):
                try:
                    v = float(self.YELLOWTIME2_string.get())
                except:
                    pass
                finally:
                    if update_func:
                        ok = update_func(v)
                        if ok: v.set(v)
                    else:
                        
            createButton(loc = (loc[0],loc[1]+1), text = "Update", command = update)

        
        # Create T and it's label
        self.T = DoubleVar()
        self.Tstring = StringVar()
        self.Tlabel = Label(self.frame, textvariable=self.Tstring)
        self.Tlabel.grid(row = 4, column = 2)
        self.T.trace('w', self.updateT)
        self.T.set(0)

        # Create two colors
        self.color1 = createDoubleVar(self.color_change1)
        self.color2 = createDoubleVar(self.color_change2)

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
        self.manual_b = createButton(loc=(1,1), text = "Manual", command = self.set_manual)
        self.auto_b = createButton(loc=(1,2), text = "Auto", command = self.set_auto)
        self.mode.set(MANUAL)

        # Create Manual buttons
        self.red_b1 = createButton(loc=(2,1), text = "Red", command = self.manual_red1)
        self.yellow_b1 = createButton(loc=(2,2), text = "Yellow", command = self.manual_yellow1)
        self.green_b1 = createButton(loc=(2,3), text = "Green", command = self.manual_green1)
        self.red_b2 = createButton(loc=(3,1), text = "Red", command = self.manual_red2)
        self.yellow_b2 = createButton(loc=(3,2), text = "Yellow", command = self.manual_yellow2)
        self.green_b2 = createButton(loc=(3,3), text = "Green", command = self.manual_green2)
        self.color1.set(RED)
        self.color2.set(GREEN)

        self.iterate_b = createButton(loc=(4,1), text = "Next", command = self.next)
        
        # AI Buttons and Labels
        self.aimode = DoubleVar()
        self.start_training_b = createButton((5, 7), "Start Training", command = self.start_training)
        self.stop_training_b  = createButton((5, 8), "Stop Training", command = self.stop_training)
        self.activate_b       = createButton((5, 9), "Activate AI", command = self.activate_ai)
        self.deactivate_b    = createButton((5, 9), "Deactivate AI", command = self.deactivate_ai)
        self.aiconf, _, _ = createNumericLabel((5,11), "Confidence", default=-1.)
    
        # Cars and pedestrians
        self.cars1, _, _ = createNumericLabel((5,1), "Cars1", default=0.)
        self.cars2, _, _ = createNumericLabel((5,2), "Cars2", default=0.)
        self.ped1, _, _ = createNumericLabel((5,3), "Ped1", default=0.)
        self.ped2, _, _ = createNumericLabel((5,11), "Ped2", default=0.)

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

    # AI Buttons and modes
    def start_training(self):
        self.aimode.set(TRAINING)
    def stop_training(self):
        self.aimode.set(WATCHING)
    def activate_ai(self):
        self.aimode.set(CONTROL)
    def deactivate_ai(self):
        self.aimode.set(WATCHING)
    def aiconf_update(self):
        self.aiconf_string.set("Confidence: {}".format(self.aiconf.get()))

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
