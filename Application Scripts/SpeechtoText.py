import speech_recognition as sr




def AudioToText():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print('Please say Something...')
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("You have said: \n " + text)
            return text

        except Exception as e:
            print("Error : " + str(e))

AudioToText()