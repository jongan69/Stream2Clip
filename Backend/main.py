import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
from moviepy.editor import VideoFileClip
import librosa
from whisperTS.main import create_sentences_from_audio
from video import *
from utils import *

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Constants
HOST = "0.0.0.0"
PORT = 8080
GENERATING = False
LOCALVIDDIR = "../temp/"
COUNT = 0

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
          # Set global variable
        global GENERATING
        GENERATING = True
        COUNT+1
        
        # Clean
        clean_dir(LOCALVIDDIR)
        clean_dir("../subtitles/")
 
        # Parse JSON
        data = request.get_json()
        print(str(data))
        ai_model = data.get('aiModel')  # Get the AI model selected by the user
        video_url = data.get('videoUrl') # URL of Video -> Download Video to temp
        strokeColor = data.get('strokeColor') # URL of Video -> Download Video to temp
        fontOutline = data.get('fontOutline') # URL of Video -> Download Video to temp
        fontSize = int(data.get('fontSize')) # URL of Video -> Download Video to temp

        video_clips = []
        if video_url:
            # Process the provided YouTube video
            ydl_opts = {
                'format': 'best',
                'outtmpl': '../temp/%(id)s.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'keepvideo': True,  # Keep the video file after extracting audio
         }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get('id', None)
            audio_path = f"../temp/{video_id}.mp3"
            video_path = f"../temp/{video_id}.mp4"
        try:
            silent_segments = detect_silent_segments(audio_path)
                
        except PermissionError:
            print(f"Permission error when accessing {audio_path}. Retrying...")
            time.sleep(20)  # Wait a bit for the file to be released
            silent_segments = detect_silent_segments(audio_path)
        
        video_clips = trim_silences_from_video(video_path, silent_segments, strokeColor, fontOutline, fontSize)
        combined_video_path = combine_videos(video_clips, 500, 160)
        return jsonify({"status": "success", "message": "Video processed successfully.", "clips": combined_video_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def detect_silent_segments(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    print(y)
    non_silent_segments = librosa.effects.split(y, top_db=30)
    silent_segments = invert_segments(non_silent_segments, len(y), sr)
    return silent_segments

def invert_segments(segments, y_length, sr):
    silent_segments = []
    start = 0
    for seg in segments:
        silent_start = librosa.frames_to_time(start, sr=sr)
        silent_end = librosa.frames_to_time(seg[0], sr=sr)
        silent_segments.append((silent_start, silent_end))
        start = seg[1]
    if start < y_length:
        silent_end = librosa.frames_to_time(y_length, sr=sr)
        silent_segments.append((librosa.frames_to_time(start, sr=sr), silent_end))
    return silent_segments

def createClip(video_path, start, end, name, strokeColor, fontOutline, fontSize):
    
    # Define Stuff
    paths = []
    clip_path = f"{LOCALVIDDIR}{name}.mp4"
    captioned_clip_path = f"{LOCALVIDDIR}captioned_{name}.mp4"
    audio_path = f"{LOCALVIDDIR}{name}.mp3"
    
    # Save Clip
    video = VideoFileClip(video_path)
    video.subclip(start, end).write_videofile(clip_path, fps=30, codec="libx264")
    
    # Generate Pieces
    video.subclip(start, end).audio.write_audiofile(audio_path) 
    stt = (create_sentences_from_audio(audio_path)).split(". ")
    stt = list(filter(lambda x: x != "", stt))
    paths.append(audio_path)
    subtitles_path = generate_subtitles(audio_path, audio_clips=paths,sentences=stt)
    
    # Combine Clip Pieces
    generator = lambda txt: TextClip(
        txt,
        font="../fonts/bold_font.ttf",
        fontsize=fontSize,
        color=fontOutline,
        stroke_color=strokeColor,
        stroke_width=5,
    )

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, generator)
    result = CompositeVideoClip([
        VideoFileClip(clip_path),
        subtitles.set_pos(("center", "center"))
    ])

    # Add the audio
    audio = AudioFileClip(audio_path)
    result = result.set_audio(audio)

    result.write_videofile(captioned_clip_path, threads=2)
    return captioned_clip_path
    
def trim_silences_from_video(video_path, silent_segments, strokeColor, fontOutline, fontSize):
    video = VideoFileClip(video_path)
    video_duration = video.duration  # Get the duration of the video
    clips = []
    for start, end in silent_segments:
        # Ensure the start and end times are within the video's duration
        if start < video_duration and end < video_duration:
            final_video_path = createClip(
                video_path, 
                start, 
                end, 
                name=len(clips),
                strokeColor=strokeColor, 
                fontOutline=fontOutline, 
                fontSize=fontSize, 
                )
            clips.append(final_video_path)
        elif start < video_duration:
            # If the start is within the video but the end is not, create a subclip until the end of the video
            final_video_path = createClip(
                video_path, 
                start, 
                end, 
                name=len(clips),
                strokeColor=strokeColor, 
                fontOutline=fontOutline, 
                fontSize=fontSize,  
                
                )
            clips.append(final_video_path)
        elif end == video_duration:
            final_video_path = createClip(
                    video_path, 
                    start, 
                    end=(end - 1), 
                    name=len(clips),
                    strokeColor=strokeColor, 
                    fontOutline=fontOutline, 
                    fontSize=fontSize,            
                    )
            clips.append(final_video_path)
        elif end > video_duration:
            return clips
 
    video.close()  # Close the original video clip
    return clips



if __name__ == '__main__':
   # Run Flask App
    app.run(debug=True, host=HOST, port=PORT)
