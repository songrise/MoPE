cd /root/autodl-tmp/PromptDistill/data/mustard/utterances_final

# for f in *.mp4; do 
#   echo "$f"
# done

# for f in *.mp4; do
#   ffmpeg -i "$f" -f wav -vn "${f%.mp4}".wav
# done

for f in *.wav; do
  python /root/autodl-tmp/PromptDistill/utils/vocal_separation.py --file "$f"
done