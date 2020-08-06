import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import streamlit as st

plt.style.use('seaborn')

st.title('Cao Vit Gibbon - Audio Recognition')

st.header('Introduction')

st.markdown("""
This project is a collaboration between CoderSchool and [Fauna & Flora International (FFI)](https://www.fauna-flora.org/countries/vietnam).
Our goal in this project is to apply Machine Learning to aid the preservation of the Cao Vit Gibbon - The second rarest species of primiate in the world.
At the moment, the task of monitoring the Cao Vit gibbon individuals is done manually. Experts have to take weekly/monthly trips to the forest and estimate the number of individuals using their signature call.
This is potentially a huge cost for FFI in term of time and money. With this project, we attempt to automate/semi-automate this task using Audio Classification techniques in Machine Learning.
""")

# Read audio data with librosa
st.header('Sample Audio')

path = 'data/' + \
    st.sidebar.selectbox('Select an Audio file:', os.listdir('data/'))
st.audio(path)

st.header('Visualize Audio Data')

st.markdown('We use Python library `librosa` to process audio data')

# Read audio data with librosa

with st.echo():
    signal, sr = librosa.load(path, sr=22050)
    n_fft = 2048
    hop_length = 512

# --Waveform--
st.subheader('Waveform')

with st.echo():
    plt.figure()

    librosa.display.waveplot(signal, sr=sr)

    plt.title('Waveform')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')

st.pyplot()

# --Spectrum--
st.subheader('Spectrum')

with st.echo():

    # Fast Fourier Transform -> Spectrum
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(fft))

    # Since the two parts of the strectrum are symmetrical,
    # We take the first half only
    left_frequency = frequency[:len(frequency)//2]
    left_magnitude = magnitude[:len(frequency)//2]

    # Plot frequency and magnitude
    plt.figure()

    plt.title('Spectrum')

    plt.plot(left_frequency, left_magnitude)

    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

st.pyplot()

# --Spectrogram--
st.subheader('Spectrogram')

with st.echo():

    # Short-time Fourier Transform -> Spectrogram
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(stft)

    # Convert magnitude to db for better visualization
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    # Display Spectrogram
    plt.figure()

    plt.title('Spectogram')

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)

    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.colorbar()

st.pyplot()

# --MFCC--
st.subheader('Mel-frequency cepstral coefficients (MFCCs)')

with st.echo():

    MFCCs = librosa.feature.mfcc(
        signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    # Display MFCCs
    plt.figure()

    plt.title('Mel-frequency cepstral coefficients (MFCCs)')

    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)

    plt.xlabel('Time')
    plt.ylabel('MFCCs')

    plt.colorbar()

st.pyplot()
