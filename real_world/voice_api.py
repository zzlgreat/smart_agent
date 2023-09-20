from elevenlabs import clone, generate, play, set_api_key,voices
from elevenlabs.api import History
import requests

set_api_key("9cbb901cd2dbd1ffab35d711781a2b58")
#print(voices)
voice = clone(
    name="202309231003",
    description="An usa woman, young ",
    files=["./sample1.mp3", "./sample2.mp3"],
)

audio = generate(text="Some very long text to be read by the voice", voice=voice)
#
# play(audio)
#
# history = History.from_api()
# print(history)
