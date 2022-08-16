# running text to speech through GTTS

import gtts
from playsound import playsound

def ConvertToAudio(text):
    tts = gtts.gTTS(text, lang='en', tld='co.uk')
    tts.save("GTTSDEMO.mp3")
    playsound("GTTSDEMO.mp3")


Text = "Hi, I m calling from Future Connect Training and Recruitment. Can I have your name please?"
ConvertToAudio(Text)

