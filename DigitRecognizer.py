import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from NeuralNetwork import NeuralNetwork

class DigitRecognizer:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Draw on 28x28 Window")

        self.model = model

        self.width, self.height = 280, 280
        self.scale = self.width // 28
        self.line_width = 1
        self.current_image = np.zeros((28, 28), dtype=np.uint8)

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 20))
        self.prediction_label.pack(pady=10)

        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear_screen)
        self.button_clear.pack()

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.draw_on_screen()

    def draw_pixel(self, pos):
        x, y = pos
        self.current_image[y - self.line_width: y + self.line_width, x - self.line_width:x + self.line_width] = 255

    def on_mouse_down(self, event):
        self.drawing = True

    def on_mouse_up(self, event):
        self.drawing = False

    def on_mouse_motion(self, event):
        if self.drawing:
            x = event.x // self.scale
            y = event.y // self.scale
            self.draw_pixel((x, y))
            self.draw_on_screen()

    def draw_on_screen(self):
        img = Image.fromarray(self.current_image, 'L')
        img = img.resize((self.width, self.height))
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk

    def predict(self):
        # Supongamos que tienes una instancia de NeuralNetwork llamada nn
        prediction = self.model.predict(self.current_image.reshape(-1, 1) / 255.)
        self.prediction_label.config(text="Prediction: " + str(prediction))

    def clear_screen(self):
        self.current_image = np.zeros((28, 28), dtype=np.uint8)
        self.draw_on_screen()
        self.prediction_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    digit_recognizer = DigitRecognizer(root)
    root.mainloop()
