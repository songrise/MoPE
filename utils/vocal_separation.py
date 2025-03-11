# command line extract .wav from videos
# convert stereo to mono: -ac; Resample to 22050: -ar
# for %f in (*.mp4) do ffmpeg -i "%f" -f wav -vn -ac 1 -ar 22050 "%~nf.wav"

##################
# reduce background noise
import numpy as np
import librosa
import argparse
import soundfile as sf


def vocal_separation(in_path):
	y, sr = librosa.load(in_path)
	# And compute the spectrogram magnitude and phase
	S_full, phase = librosa.magphase(librosa.stft(y))
	S_filter = librosa.decompose.nn_filter(S_full,
										   aggregate=np.median,
										   metric='cosine',
										   width=int(librosa.time_to_frames(2, sr=sr)))
	S_filter = np.minimum(S_full, S_filter)
	margin_i, margin_v = 2, 10
	power = 2
	mask_i = librosa.util.softmask(S_filter,
								   margin_i * (S_full - S_filter),
								   power=power)

	mask_v = librosa.util.softmask(S_full - S_filter,
								   margin_v * S_filter,
								   power=power)
	S_foreground = mask_v * S_full

	new_y = librosa.istft(S_foreground*phase)
	sf.write(in_path, new_y, sr, subtype='PCM_16')
	print(f"Successfully processed {in_path}")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", default="")
	args = parser.parse_args()
	vocal_separation(args.file)
	