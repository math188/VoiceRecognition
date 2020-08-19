from flask import Flask,render_template,redirect,url_for,request,send_from_directory
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager
# from flask_app import routes
import os
import time
import random
import librosa
#from keras.models import load_model
import numpy as np

import collections
import contextlib
import sys
import wave
import webrtcvad
from flask_app.static.CRNN import CRNN_04 as model
import torch, librosa, torchaudio, torchaudio.transforms as transforms

app = Flask(__name__)

# app.config['SECRET_KEY'] = '1A37BbcCJh67'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# db = SQLAlchemy(app)

# login_manager = LoginManager()
# login_manager.init_app(app)
def load_model(model,model_path):
    model = model(6)
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

@app.route('/',methods=['GET','POST'])
def home():
    global model
    model_path = './flask_app/static/h5_file/CRNN_04_epochs70_adam_CE_batch1_lr1e-05.pth.tar'
    if request.method == 'POST':
        #members = ['one','four','five','two','three','six']
        members = ['five','three','six','four','one','two']
        real = ['Ryan', 'Rick', 'Yanbo', 'Hsiaoen', 'Kunyu', 'Joyee']
        #real = ['Kunyu','Hsiaoen','Ryan','Joyee','Rick','Yanbo']
       # index = random.randint(0,len(members)-1)
       # name = members[index]
        from flask_app.static.CRNN import CRNN_04 as model
        model = load_model(model,model_path)
        audio, sample_rate = read_wave('./flask_app/static/wav_file/predict.wav')
        vad = webrtcvad.Vad(1)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 300, vad, frames)
        for segment in segments:
            path = './flask_app/static/wav_file/after_vad.wav'
            write_wave(path, segment, sample_rate)
        noisy, sr = librosa.load('./flask_app/static/wav_file/after_vad.wav',sr=16000, mono=True)
        noisy = noisy/max(abs(noisy))
        MFCC_fea = transforms.MFCC(16000,melkwargs={'n_fft':512,'hop_length':160})(torch.from_numpy(noisy)).squeeze().t()
        MFCC_fea = MFCC_fea.unsqueeze(0)
        pred = model(MFCC_fea).max(1)[1].numpy()	

        # noisy,sr = librosa.load('./flask_app/static/wav_file/predict.wav', sr=16000, mono=True)
        #noisy = noisy[13000:26440]
        #mfcc = librosa.feature.mfcc(y=noisy, sr=sr,n_mfcc=40, dct_type=2, hop_length=256,  n_fft=512, center=False)
        #fea = mfcc.transpose()
        #test = np.reshape(fea,(1,51,-1,1))
        #model = load_model('./flask_app/static/h5_file/model.43-0.04.h5')
        #pre = model.predict(test)
        name = members[int(pred)]
        real_name = real[int(pred)]
        return render_template('home.html',name=name,real_name=real_name)
    return render_template('home.html',name='')

@app.route('/getcwd')
def getcwd():
    return os.getcwd()

@app.route('/testrecord', methods=['GET', 'POST'])
def test_record():
    # f = request.files['audio_data']
    # f.save('/static/wav_file/try.wav')
    # print('file uploaded successfully')
    # return render_template('index.html', ttttt="上傳成功")

    if request.method == "POST":
        f = request.files['audio_data']
        #name = './flask_app/static/wav_file/joyee/'+str(int(time.time()))+'.wav' 
        name = './flask_app/static/wav_file/predict.wav'
        f.save(name)
        return '錄音成功'
    else:
        return render_template("home.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        global model
        model_path = './flask_app/static/h5_file/CRNN_04_epochs70_adam_CE_batch1_lr1e-05.pth.tar'
        file = request.files['file']
        print(request.files)
        filename = file.filename
        if file and file.filename.endswith('.wav'):
            file.save('./flask_app/static/wav_file/'+filename)
            noisy, sr = librosa.load('./flask_app/static/wav_file/'+filename,sr=None)
            #members = ['one','four','five','two','three','six']
            #real = ['Kunyu','Hsiaoen','Ryan','Joyee','Rick','Yanbo']
            members = ['five','three','six','four','one','two']
            real = ['Ryan', 'Rick', 'Yanbo', 'Hsiaoen', 'Kunyu', 'Joyee']
            # index = random.randint(0,len(members)-1)
            # name = members[index]
            from flask_app.static.CRNN import CRNN_04 as model
            model = load_model(model,model_path)
            noisy,sr = librosa.load('./flask_app/static/wav_file/'+filename, sr=16000, mono=True)
            #noisy = noisy[7000:20440]
            #mfcc = librosa.feature.mfcc(y=noisy, sr=sr,n_mfcc=40, dct_type=2, hop_length=256,  n_fft=512, center=False)
            #fea = mfcc.transpose()
            #test = np.reshape(fea,(1,51,-1,1))
            #model = load_model('./flask_app/static/h5_file/CNN2.h5')
            #pre = model.predict(test)
            noisy = noisy/max(abs(noisy))
            MFCC_fea = transforms.MFCC(16000,melkwargs={'n_fft':512,'hop_length':160})(torch.from_numpy(noisy)).squeeze().t()
            MFCC_fea = MFCC_fea.unsqueeze(0)
            pred = model(MFCC_fea).max(1)[1].numpy()
            name = members[int(pred)]
            real_name = real[int(pred)]
            return render_template('home.html',name=name,real_name=real_name,filename=filename)
        return render_template('home.html',name='',condition='error')

if __name__ == "__main__":
    app.run(debug=True)


