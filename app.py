import streamlit as st
import librosa
import torch
import numpy as np
from models import SalientRagaDetector
from models import RagaDetector
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import youtube_dl
import os

st.title("PhonoNet Raga Analyzer")

MODEL_PATH = 'aug2_epoch_665_acc_0.7888888888888889.model'
ragas = ['Todi',
         'Puriyā dhanaśrī',
         'Rāgēśrī',
         'Mārvā',
         'Yaman kalyāṇ',
         'Bhūp',
         'Miyān malhār',
         'Madhukauns',
         'Madhuvanti',
         'Basant',
         'Khamāj',
         'Kēdār',
         'Lalit',
         'Alahaiyā bilāval',
         'Bhairav',
         'Bihāg',
         'Jōg',
         'Bilāsakhānī tōḍī',
         'Mārūbihāg',
         'Darbāri',
         'Dēś',
         'Ahira bhairav',
         'Śrī',
         'Gauḍ malhār',
         'Ābhōgī',
         'Śuddh sāraṅg',
         'Bairāgi',
         'Mālkauns',
         'Bāgēśrī',
         'Hamsadhvāni']


@st.cache
def load_phononet_model():
    model = RagaDetector(0, 256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['net'])
    model.eval()
    vis_model = SalientRagaDetector(model)
    vis_model.eval()
    return vis_model, model


vis_model, model = load_phononet_model()

mode = st.radio("How do you want to upload your file?", ('Youtube Link', 'WAV/OGG File Upload'))

if mode == 'Youtube Link':
    youtube_link = st.text_input('Insert Youtube Link')
    uploaded_file = None
    if youtube_link is not None and len(youtube_link) > 2:
        @st.cache
        def download_youtube():
            name = youtube_link.split('=')[1]
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
                    ydl.download([youtube_link])

            return name + '.mp3'
        uploaded_file = download_youtube()

else:
    st.info('mp3 File Upload is Not Currently Supported')
    uploaded_file = st.file_uploader("Choose an Audio file (mp3 not supported):", type=["wav", 'ogg'])



if uploaded_file is not None:

    @st.cache
    def load_file():
        return librosa.load(uploaded_file)


    y, sr = load_file()


    @st.cache
    def generate_chromagram():
        return librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)


    raw_chromagram = generate_chromagram()


    @st.cache(suppress_st_warning=True)
    def run_phononet_model():
        song = torch.from_numpy(raw_chromagram).unsqueeze(0).unsqueeze(0).float()

        out = None
        chunk_confidences = []
        clear_vis = []
        with torch.no_grad():
            song_segs = song.split(1500, dim=3)
            for i in range(len(song_segs)):
                chunk = song_segs[i]  # get specific chunk
                if not chunk.shape[3] == 1500:  # hardcoded 1500 size
                    padding = torch.zeros(1, 1, chunk.size()[2], 1500 - chunk.size()[3])
                    chunk = torch.cat((chunk, padding), 3)
                if (out is None):
                    out = model.fc1(model(chunk.contiguous()))
                else:
                    out += model.fc1(model(chunk.contiguous()))
                chunk_confidences.append(torch.nn.functional.softmax(out, dim=1).numpy()[0] * 100.)
                _, saliency = vis_model(chunk.contiguous())
                clear_vis.append(saliency[0][0])
        clear_vis = np.concatenate(clear_vis, axis=1)
        return clear_vis, chunk_confidences, torch.nn.functional.softmax(out, dim=1).numpy()[0] * 100


    st.markdown('## Raga Predictions')
    clear_vis, chunk_confidences, confidences = run_phononet_model()
    prediction = np.argmax(confidences)

    fig = px.bar(x=ragas, y=confidences, labels={'x': 'Hindustani Raga', 'y': 'Prediction Confidence (%)'})
    fig.update_xaxes(tickangle=270)
    fig.update_yaxes(range=[0, 100])

    st.plotly_chart(fig)


    @st.cache
    def find_optimal_saliency():

        # Define a ten second window size
        x_size = 120
        max_x_start = -1
        max_accuracy_delta = 10000

        for x_start in range(0, raw_chromagram.shape[1], x_size):  # iterate through windows of x_size
            try:
                # Crop out segment of the chromagram
                new_vsong = np.array(raw_chromagram, copy=True)
                new_vsong[:, x_start: x_start + x_size] = 0

                # Run new system through network
                song_segs = torch.from_numpy(new_vsong).unsqueeze(0).unsqueeze(0).float().split(1500, dim=3)
                with torch.no_grad():
                    out = model.fc1(model(song_segs[x_start // 1500].contiguous()))

                confidences = torch.nn.functional.softmax(out, dim=1).numpy()[0] * 100.

                # Compute accuracy delta
                accuracy_delta = 100. * (
                        float(confidences[prediction]) - float(chunk_confidences[x_start // 1500][prediction])) / float(
                    chunk_confidences[x_start // 1500][prediction])

                if (accuracy_delta < max_accuracy_delta):
                    max_accuracy_delta = accuracy_delta
                    max_x_start = x_start
            except Exception:
                pass
        return x_size, max_x_start


    x_size, max_x_start = find_optimal_saliency()

    st.markdown('## Extracted Salient Segment')
    st.markdown(
        'The PhonoViz saliency evaluation algorithm finds the 10 second segment which contributes most to the system\'s raga prediction and visualizes the key segments of the chroma which contribute most to this prediction.')
    plt.subplot(2, 1, 1)
    librosa.display.specshow((raw_chromagram[:, max_x_start: max_x_start + x_size]), x_axis='time', hop_length=2048,
                             sr=sr,
                             cmap=get_cmap('magma'), y_axis='chroma')
    plt.colorbar()
    plt.title('Original Chromagram')
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    librosa.display.specshow((clear_vis[:, max_x_start: max_x_start + x_size]), x_axis='time', hop_length=2048, sr=sr,
                             cmap=get_cmap('magma'), y_axis='chroma')
    plt.colorbar()
    plt.title('PhonoViz Generated Chromagram')
    plt.tight_layout()
    st.pyplot()

    st.balloons()

# This is a test
