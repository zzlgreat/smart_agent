from fastapi import FastAPI, Query
from starlette.responses import FileResponse
import datetime
from elevenlabs import generate, save, set_api_key

# Initialize FastAPI
app = FastAPI()

# Set API key
set_api_key("9cbb901cd2dbd1ffab35d711781a2b58")

@app.get("/get_voice/")
def get_voice_endpoint(text: str, voicename: str):
    """
    Generate voice from the given text and voicename.
    """
    audio = generate(
        text=text,
        voice=voicename,
        model='eleven_monolingual_v1'
    )
    datetimestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save(audio, './data/wav/'+datetimestr+'.wav')
    with open('./data/text/'+datetimestr+'.wav','a') as j:
        j.write(text)
    return FileResponse('./data/'+datetimestr+'.wav', headers={"Content-Type": "audio/wav"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9819)
