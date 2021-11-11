import tkinter
from tkinter import *
from tkinter.font import Font


class Interface:
    def __init__(self, test):
        self.window = None

        self.window_height = 500
        self.window_width = 1000
        self.text_field_height = int(self.window_height / 20)
        self.text_field_width = int(self.window_width / 10)
        self.input_text_field_height = int(self.text_field_height / 10)
        self.input_text_field_width = self.text_field_width

        self.text_field = None
        self.input_text = None
        self.input = None
        self.test = test  # it's a class created to predict results.

    def start(self):
        self.gui_window()
        self.gui_text_field()
        self.gui_input_text_field()
        self.window.mainloop()

    def gui_window(self):
        self.window = tkinter.Tk()
        self.window.minsize(self.window_width, self.window_height)
        self.window.config(bg=self.rgb_hack((59, 59, 59)))
        self.window.title("AI-Chatbot")

    def gui_text_field(self):
        self.text_field = Text(self.window,
                               height=self.text_field_height,
                               width=self.text_field_width,
                               bg=self.rgb_hack((71, 67, 67)),
                               fg="white")
        self.text_field.pack()

    def gui_input_text_field(self):
        self.input = Text(self.window,
                          height=self.input_text_field_height,
                          width=self.input_text_field_width,
                          bg=self.rgb_hack((71, 67, 67)),
                          fg="white")
        self.input.pack()

        button = tkinter.Button(self.window,
                                height=2,
                                width=20,
                                text="Send",
                                command=lambda: self.take_input(self.input))

        button.place(x=self.window_width - 600, y=self.window_height - 40)

    def take_input(self, input):

        self.input_text = input.get("1.0", "end-1c")

        if self.input_text is not None:
            self.text_field.insert(END, "User: " + self.input_text + "\n")

            answer = self.test.predict(self.input_text)
            self.text_field.insert(END, "Bot: " + answer + "\n \n")

            input.delete("0.0", END)

    def delete_text(self):
        self.input.delete("1.0", "end")

    def rgb_hack(self, rgb):
        return "#%02x%02x%02x" % rgb
