from scipy.signal import spectrogram
from PIL import Image
import matplotlib.pyplot as plt
import io

def save_spectrogram(epoch, save_file, fs=64):
    f, t, Sxx = spectrogram(epoch, fs=fs, nperseg=128, noverlap=64)
    
    fig = plt.figure(figsize=(1, 0.8), dpi=100)
    plt.axis('off')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylim(0, 32)

    # ذخیره در بافر
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # تبدیل به تصویر PIL
    im = Image.open(buf).convert('RGB')
    im = im.crop((11, 14, 71, 90))  # crop به 76x60
    im.save(save_file)
    plt.close(fig)
