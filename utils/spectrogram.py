import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from PIL import Image

def save_spectrogram(epoch, save_path, fs=64):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import spectrogram
    from PIL import Image

    epoch = epoch - np.mean(epoch)
    epoch = epoch / (np.std(epoch) + 1e-8)

    f, t, Sxx = spectrogram(epoch, fs=fs, nperseg=128, noverlap=64)
    Sxx = np.log(Sxx + 1e-10)  

    fig = plt.figure(figsize=(1.3, 1.0), dpi=100)
    plt.axis('off')
    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis', vmin=np.min(Sxx), vmax=np.max(Sxx))
    plt.tight_layout(pad=0)
    fig.canvas.draw()

    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    img = Image.fromarray(img).resize((76, 60))
    img.save(save_path)

