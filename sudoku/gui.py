import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from sudoku.feature_extracter import FeatureExtracter
from sudoku.solve_sudoku import SolveSudoku
from PIL import Image, ImageTk


class GUI:

    def __init__(self, config) -> None:
        self.config = config
        self.title = config["GUI"]["name"]
        self.window_size = config["GUI"]["window_size"]
        self.canvas_size = config["GUI"]["canvas_size"]
        self.root = tk.Tk()
        
        self.setting_default()
        self.create_canvas()
        self.create_button()

    def setting_default(self):
        self.root.title(self.title)
        self.root.geometry(self.window_size)

    def create_canvas_grid(self, master):
        canvas = tk.Canvas(master, width=32, height=40)
        canvas.pack(side=tk.LEFT, padx=(10, 10))

        _canvas = tk.Canvas(canvas, width=32, height=32, bg='black')
        _canvas.pack()

        entry = ttk.Entry(canvas, width=5)
        entry.pack()
        return _canvas, entry

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.canvas_size*2, height=self.canvas_size)
        self.canvas.pack(pady=(10, 0))

        self.canvas_buttom = tk.Canvas(self.root, width=250, height=250)
        self.canvas_buttom.pack(pady=(50, 0))

        self.canvas1 = tk.Canvas(
            self.canvas, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas1.pack(side=tk.LEFT, padx=(0, 100))

        self.canvas2 = tk.Canvas(self.canvas, width=self.canvas_size, height=self.canvas_size)
        self.canvas2.pack(side=tk.LEFT)

        self.canvas_row = []
        for _ in range(9):
            canvas = tk.Canvas(self.canvas2, width=500, height=50)
            canvas.pack()
            self.canvas_row.append(canvas)

        self.canvas_image = []
        self.entry_pred = []
        for canvas_row in self.canvas_row:
            for _ in range(9):
                canvas_image, entry_pred = self.create_canvas_grid(canvas_row)
                self.canvas_image.append(canvas_image)
                self.entry_pred.append(entry_pred)

    def create_button(self):
        s = ttk.Style()
        s.configure(style="my.TButton", font=('Console', 20))
        self.btn_selectImage = ttk.Button(
            self.canvas_buttom, text='Select image', padding=(20, 10), style='my.TButton', command=self.selectImage)
        self.btn_selectImage.grid(column=0, row=0)

        self.btn_run = ttk.Button(
            self.canvas_buttom, text='Run', padding=(20, 10), style='my.TButton', command=self.feature_extracter)
        self.btn_run.grid(column=1, row=0)

        self.btn_test = ttk.Button(
            self.canvas_buttom, text='Test', padding=(20, 10), style='my.TButton', command=self.getNumber2Entry
        )
        self.btn_test.grid(column=2, row=0)

        self.btn_back = ttk.Button(
            self.canvas_buttom, text='Back', padding=(20, 10), style='my.TButton', command=self.back_canvas)
        self.btn_back.grid(column=0, row=1)

        self.btn_save = ttk.Button(
            self.canvas_buttom, text="Save", padding=(20, 10), style='my.TButton', command=self.save)
        self.btn_save.grid(column=1, row=1)

        self.btn_delete = ttk.Button(
            self.canvas_buttom, text="Reset", padding=(20, 10), style='my.TButton', command=self.reset)
        self.btn_delete.grid(column=2, row=1)

    def update(self):
        try:
            self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.photo2)
        except:
            pass
        try:
            for i, canvas in enumerate(self.canvas_image):
                canvas.create_image(0, 0, anchor=tk.NW,
                                    image=self.photos_num[i])
        except:
            pass

    def selectImage(self):
        filetype = (
            ('image files', '*.jpg'),
            ('All file', '*.*')
        )
        image = filedialog.askopenfilename(filetypes=filetype)
        im = cv2.imread(image)
        self.img = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, dsize=(510, 510))
        im = Image.fromarray(im)
        self.photo = ImageTk.PhotoImage(im)
        self.canvas1.after(1, self.update)

    def display_image(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, dsize=(510, 510))
        im = Image.fromarray(im)
        return im

    def delete_subcanvas(self):
        self.canvas2.destroy()
        self.canvas2 = tk.Canvas(
            self.canvas, width=510, height=510, bg='black')
        self.canvas2.pack(side=tk.LEFT)

    def back_canvas(self):
        self.delete_subcanvas()
        self.canvas_row = []
        for _ in range(9):
            canvas = tk.Canvas(self.canvas2, width=500, height=50)
            canvas.pack()
            self.canvas_row.append(canvas)

        self.canvas_image = []
        self.entry_pred = []
        for canvas_row in self.canvas_row:
            for _ in range(9):
                canvas_image, entry_pred = self.create_canvas_grid(canvas_row)
                self.canvas_image.append(canvas_image)
                self.entry_pred.append(entry_pred)

        self.photo = ImageTk.PhotoImage(self.display_image(self.img_pred))
        self.canvas1.after(1, self.update)
        self.getNumber()
        self.update()

    def getNumber2Entry(self):
        self.photo = ImageTk.PhotoImage(self.display_image(self.img))
        self.canvas1.after(1, self.update)
        pred_number = np.array(
            [entry.get() for entry in self.entry_pred], dtype=np.uint8).reshape(9, 9)
        self.pred = pred_number.flatten()
        self.delete_subcanvas()
        pred_copy = pred_number.copy()
        solve_sdk = SolveSudoku(pred_number)
        solve_sdk.solve()
        grid_show = np.where(pred_number == pred_copy, 0, pred_number)
        img = self.extractor.display_predict_number(grid_show)
        self.img_ = self.display_image(img)
        self.photo2 = ImageTk.PhotoImage(self.img_)
        self.canvas2.after(1, self.update)

    def getNumber(self):
        for i, entry in enumerate(self.entry_pred):
            entry.delete(0, 'end')
            entry.insert(0, self.pred[i])

    def deleteNumber(self):
        try:
            for entry in self.entry_pred:
                entry.delete(0, 'end')
        except:
            pass
        for canvas in self.canvas_image:
            canvas.destroy()

    def feature_extracter(self):
        self.extractor = FeatureExtracter(self.img, self.config)
        pred, imgs = self.extractor()
        self.photos_num = [ImageTk.PhotoImage(
            Image.fromarray(cv2.resize(img, (32, 32)))) for img in imgs]
        for canvas in self.canvas_image:
            canvas.after(0, self.update)

        self.img_pred = self.extractor.display_predict_number(
            pred, alpha=1, beta=0.8, color=(255, 0, 0))
        self.photo = ImageTk.PhotoImage(self.display_image(self.img_pred))
        self.canvas1.after(1, self.update)
        self.pred = pred.flatten()
        self.getNumber()

    def save(self):
        f = filedialog.asksaveasfilename(defaultextension=".jpg")
        if f:
            img = np.array(self.img_)
            cv2.imwrite(f, img)

    def reset(self):
        self.canvas.destroy()
        self.canvas_buttom.destroy()
        self.deleteNumber()
        self.create_canvas()
        self.create_button()
        self.photos_num = []

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.run()
