from tkinter import *
import os
from tkinter import filedialog
from tkinter import Menu
import tkinter as tk
import json, torch, librosa, sys
import torch.utils.data as data
import numpy as np
from models import RagaDetector

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
# from __future__ import unicode_literals
import youtube_dl


ragas = ['Todi'           ,
'Puriyā dhanaśrī' ,
'Rāgēśrī'         ,
'Mārvā'           ,
'Yaman kalyāṇ'    ,
'Bhūp'            ,
'Miyān malhār'    ,
'Madhukauns'      ,
'Madhuvanti'      ,
'Basant'          ,
'Khamāj'          ,
'Kēdār'           ,
'Lalit'           ,
'Alahaiyā bilāval',
'Bhairav'         ,
'Bihāg'           ,
'Jōg'             ,
'Bilāsakhānī tōḍī',
'Mārūbihāg'       ,
'Darbāri'         ,
'Dēś'             ,
'Ahira bhairav'   ,
'Śrī'             ,
'Gauḍ malhār'     ,
'Ābhōgī'          ,
'Śuddh sāraṅg'    ,
'Bairāgi'         ,
'Mālkauns'        ,
'Bāgēśrī'         ,
'Hamsadhvāni']

import matplotlib.pyplot as plt

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.top.attributes('-type', 'dialog')
        self.l=Label(top,text="Enter URL")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack()
        self.b=Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class Program:
    def solve(self, file_name):
        print('Loading song')

        if(os.path.isfile(file_name.replace('.mp3', '.npy'))):
            vsong = np.load(file_name.replace('.mp3', '.npy')) 
        else:
            y, sr = librosa.load(file_name)
            vsong = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)
            np.save(file_name.replace('.mp3', ''), vsong)

        print('Done loading song')

        song = torch.from_numpy(vsong).unsqueeze(0).unsqueeze(0).float()
        model = RagaDetector(0, 256)
        model.load_state_dict(torch.load('aug2_epoch_665_acc_0.7888888888888889.model', map_location='cpu')['net'])
        model.eval()

        out = None
        with torch.no_grad():
            song_segs = song.split(1500, dim=3)
            for i in range(len(song_segs)):
                chunk = song_segs[i] # get specific chunk
                if not chunk.shape[3] == 1500:  # hardcoded 1500 size
                    padding = torch.zeros(1, 1, chunk.size()[2], 1500 - chunk.size()[3])
                    chunk = torch.cat((chunk, padding), 3)
                if(out is None):
                   out = model.fc1(model(chunk.contiguous()))
                else:
                   out += model.fc1(model(chunk.contiguous()))

        print(out)
        confidences = torch.nn.functional.softmax(out, dim=1).numpy()[0]*100
        print(confidences)
        self.line1.remove()
        self.line1 = self.ax.bar(ragas, confidences)
        self.canvas.draw()

    
    def youtubeDialog(self):
        self.w = popupWindow(self.root)
        self.root.wait_window(self.w.top)
        url = self.w.value
        name = url.split('=')[1]

        if not os.path.exists(name + ".mp3"):
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': name + '.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }

            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        self.solve(name + ".mp3")

    def fileDialog(self):
        file_name = (filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("mp3 files","*.mp3"),("all files","*.*"))))

        self.solve(file_name)


    def __init__(self):
        self.root = Tk()
        self.root.attributes('-type', 'dialog')
        self.root.title("PhonoNet")
        self.root.geometry('1000x800')
        self.menu = Menu(self.root)
        
        self.new_item = Menu(self.menu)
        self.new_item.add_command(label='New Audio File', command=self.fileDialog)
        self.new_item.add_command(label='New Youtube Song', command=self.youtubeDialog)
        self.menu.add_cascade(label='File', menu=self.new_item)
        self.fig = Figure(figsize=(5,7,), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line1 = self.ax.bar(ragas, [0] * len(ragas))
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Prediction Confidence (%)', fontsize=14)
        self.ax.set_xlabel('\nRaga', fontsize=14)
        self.ax.set_title('Prediction Confidence vs. Raga', fontsize=14)
        for tick in self.ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(12)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.root.config(menu=self.menu)
        self.root.mainloop()

        print('Loading song')

        if(os.path.isfile(file_name.replace('.mp3', '.npy'))):
            vsong = np.load(file_name.replace('.mp3', '.npy')) 
        else:
            y, sr = librosa.load(file_name)
            vsong = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)
            np.save(file_name.replace('.mp3', ''), vsong)

        print('Done loading song')

        song = torch.from_numpy(vsong).unsqueeze(0).unsqueeze(0).float()
p = Program()
